#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>

namespace forcesmith {

FORCE_INLINE double ADPForceCalculator::quad_nu(const SymTens &M,
                                                const Vec3 &d) {
  return d.dot(M * d) - d.squaredNorm() / 3.0 * M.trace();
}

FORCE_INLINE Vec3 ADPForceCalculator::quad_xi(const SymTens &M, const Vec3 &d) {
  return M * d - (M.trace() / 3.0) * d;
}

std::optional<PairForce>
ADPForceCalculator::make_pair_force(const Atom &ai, const NeighborEntry &nb) {
  const Vec3 &d = nb.dist;
  const double r = d.norm();
  if (r < 1e-14) {
    return std::nullopt;
  }
  PairForce pf{&ai, nb.neighbor, d, r, 1.0 / r};
  pf.sites = nb.sites; // carry the spline-cache hints into the force stages
  return pf;
}

PairForce ADPForceCalculator::add_eam_force(PairForce &&pf) const {
  const Atom &ai = *pf.ai;
  const Atom &aj = *pf.aj;
  // Gate each radial table on its own cutoff (see in_range).
  const auto &phi_pot = pair[ai, aj];
  const auto &g_j = density[aj];
  const auto &g_i = density[ai];
  const auto [phi, dphi] = eval_deriv_gated(phi_pot, pf.sites[kSitePhi], pf.r);
  pf.phi = phi;
  const double drho_j = deriv_gated(g_j, pf.sites[kSiteGj], pf.r);
  const double drho_i = deriv_gated(g_i, pf.sites[kSiteGi], pf.r);
  pf.force += (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * pf.inv_r * pf.d;
  return std::move(pf);
}

PairForce ADPForceCalculator::add_dipole_force(PairForce &&pf) const {
  const Atom &ai = *pf.ai;
  const Atom &aj = *pf.aj;
  const auto &dip = dipole[ai, aj];
  const auto [u, du] = eval_deriv_gated(dip, pf.sites[kSiteDipole], pf.r);
  const double dot_i = ai.mu.dot(pf.d);
  const double dot_j = aj.mu.dot(pf.d);
  pf.force += (du * pf.inv_r * (dot_i - dot_j)) * pf.d + u * (ai.mu - aj.mu);
  return std::move(pf);
}

PairForce ADPForceCalculator::add_quadrupole_force(PairForce &&pf) const {
  const Atom &ai = *pf.ai;
  const Atom &aj = *pf.aj;
  const auto &quad = quadrupole[ai, aj];
  const auto [w, dw] = eval_deriv_gated(quad, pf.sites[kSiteQuad], pf.r);
  const double nu = quad_nu(ai.lambda, pf.d) + quad_nu(aj.lambda, pf.d);
  const Vec3 xi = quad_xi(ai.lambda, pf.d) + quad_xi(aj.lambda, pf.d);
  pf.force += (dw * pf.inv_r * nu) * pf.d + 2.0 * w * xi;
  return std::move(pf);
}

PairForce ADPForceCalculator::accumulate(Atom &ai, Configuration &cfg,
                                         PairForce &&pf) {
  ai.calc_force += pf.force;
  cfg.calc_energy += 0.5 * pf.phi;
  // Virial: bond ⊗ force-on-partner = d ⊗ (−fvec); 0.5 for the full list.
  cfg.calc_stress -= 0.5 * pf.d * pf.force.transpose();
  return std::move(pf);
}

std::size_t ADPForceCalculator::param_count() const {
  auto count_range = [](const auto &range) {
    return std::transform_reduce(range.begin(), range.end(), std::size_t{0},
                                 std::plus<>{},
                                 [](const auto &p) { return p.param_count(); });
  };
  return count_range(pair) + count_range(density) + count_range(embedding) +
         count_range(dipole) + count_range(quadrupole);
}

void ADPForceCalculator::gather_params(Eigen::VectorXd &dst,
                                       std::size_t off) const {
  gather_range(pair, dst, off);
  gather_range(density, dst, off);
  gather_range(embedding, dst, off);
  gather_range(dipole, dst, off);
  gather_range(quadrupole, dst, off);
}

void ADPForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                        std::size_t off) {
  scatter_range(pair, src, off);
  scatter_range(density, src, off);
  scatter_range(embedding, src, off);
  scatter_range(dipole, src, off);
  scatter_range(quadrupole, src, off);
}

void ADPForceCalculator::gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                                       std::size_t off) const {
  gather_bounds_range(pair, lo, hi, off);
  gather_bounds_range(density, lo, hi, off);
  gather_bounds_range(embedding, lo, hi, off);
  gather_bounds_range(dipole, lo, hi, off);
  gather_bounds_range(quadrupole, lo, hi, off);
}

double ADPForceCalculator::max_cutoff() const {
  auto max_cutoff = [](const auto &range) {
    return std::transform_reduce(
        range.begin(), range.end(), 0.0,
        [](double a, double b) { return std::max(a, b); },
        [](const auto &p) { return p.span().second; });
  };

  // Cover every radial table: dipole/quadrupole may reach farther than the
  // pair/density tables, and those neighbours must not be truncated (forcesmith
  // gates each contribution on its own per-table cutoff).
  return std::max({max_cutoff(pair), max_cutoff(density), max_cutoff(dipole),
                   max_cutoff(quadrupole)});
}

void ADPForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  // Zero scratch + output
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  cfg.calc_limit = 0.0;
  for (auto &atom : cfg.atoms) {
    atom.ZeroForce();
    atom.ZeroScratch();
  }

  // Pass 1: accumulate per-atom moments ρ_i, μ_i, λ_i
  std::ranges::for_each(cfg.atoms, [&](auto &ai) {
    double rho = 0.0;
    Vec3 mu = Vec3::Zero();
    SymTens lambda = SymTens::Zero();

    for (const auto &nb : ai.neighbors) {
      const auto &aj = *nb.neighbor;
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }

      // Gate each radial table on its own cutoff (see in_range): the neighbor
      // list spans the global max_cutoff(), so a shorter table would otherwise
      // extrapolate past its last knot here.
      const auto &g = density[aj];
      const auto &dip = dipole[ai, aj];
      const auto &quad = quadrupole[ai, aj];
      // eval_gated returns 0 outside each table's own cutoff (the += then no-ops).
      rho += eval_gated(g, nb.sites[kSiteGj], r);
      mu += eval_gated(dip, nb.sites[kSiteDipole], r) * nb.dist;
      lambda +=
          eval_gated(quad, nb.sites[kSiteQuad], r) * (nb.dist * nb.dist.transpose());
    }

    ai.rho += rho;
    ai.mu += mu;
    ai.lambda += lambda;
  });

  // After pass 1: embedding + ADP self-energies, cache gradF_i
  const double energy = std::transform_reduce(
      cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{}, [&](auto &ai) {
        const auto &emb = embedding[ai];

        // Clamp out-of-range ρ to the embedding table and punish the overshoot
        // (matches forcesmith's RESCALE branch, force_eam.c:334-358): F(ρ) is
        // evaluated at the clamped ρ, never extrapolated.
        const auto [rho_begin, rho_end] = emb.span();
        if (ai.rho > rho_end) {
          const double d = ai.rho - rho_end;
          cfg.calc_limit += kDummyWeight * 10.0 * d * d;
          ai.rho = rho_end;
        } else if (ai.rho < rho_begin) {
          const double d = rho_begin - ai.rho;
          cfg.calc_limit += kDummyWeight * 10.0 * d * d;
          ai.rho = rho_begin;
        }

        const auto [emb_F, emb_dF] = emb.eval_and_deriv(ai.rho);
        ai.gradF = emb_dF;

        double e = emb_F;
        e += 0.5 * ai.mu.squaredNorm();

        const double tr_lam = ai.lambda.trace();
        e += 0.5 * (ai.lambda.squaredNorm() - tr_lam * tr_lam / 3.0);

        return e;
      });

  cfg.calc_energy += energy;

  // Pass 2: forces
  // Run each i–j bond through the pipeline:
  //   geometry → EAM force → dipole force → quadrupole force → commit
  // std::optional short-circuits coincident atoms (make_pair_force), so each
  // step reads as one clear stage rather than a deeply nested loop body.
  for (Atom &ai : cfg.atoms) {
    for (const NeighborEntry &nb : ai.neighbors) {
      make_pair_force(ai, nb)
          .transform(std::bind_front(&ADPForceCalculator::add_eam_force, this))
          .transform(
              std::bind_front(&ADPForceCalculator::add_dipole_force, this))
          .transform(
              std::bind_front(&ADPForceCalculator::add_quadrupole_force, this))
          .transform(std::bind_front(&ADPForceCalculator::accumulate,
                                     std::ref(ai), std::ref(cfg)));
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(
      events::ForceEvalStats{conf_index, force_rms(cfg), cfg});
}

void ADPForceCalculator::prepare(std::span<Configuration> configs) const {
  // Single-threaded: prime the spline-cache hints for all five radial-table
  // roles a bond drives (φ, g_j, g_i, dipole u, quadrupole w).
  for (Configuration &cfg : configs) {
    build_neighbor_list(cfg, max_cutoff());
    for (Atom &ai : cfg.atoms) {
      const auto &g_i = density[ai];
      for (NeighborEntry &nb : ai.neighbors) {
        const Atom &aj = *nb.neighbor;
        const double r = nb.dist.norm();
        const auto &phi_pot = pair[ai, aj];
        const auto &g_j = density[aj];
        const auto &dip = dipole[ai, aj];
        const auto &quad = quadrupole[ai, aj];
        nb.sites[kSitePhi] =
            in_range(phi_pot, r) ? phi_pot.prepare_site(r) : SiteId{};
        nb.sites[kSiteGj] = in_range(g_j, r) ? g_j.prepare_site(r) : SiteId{};
        nb.sites[kSiteGi] = in_range(g_i, r) ? g_i.prepare_site(r) : SiteId{};
        nb.sites[kSiteDipole] = in_range(dip, r) ? dip.prepare_site(r) : SiteId{};
        nb.sites[kSiteQuad] = in_range(quad, r) ? quad.prepare_site(r) : SiteId{};
      }
    }
  }
}

} // namespace forcesmith
