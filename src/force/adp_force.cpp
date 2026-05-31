#include "potfit/force/adp_force.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <utility>

namespace potfit {
namespace {

// Helpers for quadrupole force terms (see header for derivation reference).
// nu(M, d) = d^T M d - r²/3 × tr(M)
FORCE_INLINE double quad_nu(const SymTens &M, const Vec3 &d) {
  return d.dot(M * d) - d.squaredNorm() / 3.0 * M.trace();
}
// xi(M, d) = M d - tr(M)/3 × d
FORCE_INLINE Vec3 quad_xi(const SymTens &M, const Vec3 &d) {
  return M * d - (M.trace() / 3.0) * d;
}

// ── i–j pair threaded through the force pipeline ────────────────────────────
// Each stage adds one physical contribution to `force` (the force on atom i)
// and passes the pair on. An empty std::optional means the two atoms are
// coincident and contribute nothing, so the chain's .transform short-circuits.
// Stage 1 — geometry. Empty for coincident atoms.
std::optional<PairForce> make_pair_force(const Atom &ai,
                                         const NeighborEntry &nb) {
  const Vec3 &d = nb.dist;
  const double r = d.norm();
  if (r < 1e-14) {
    return std::nullopt;
  }
  return PairForce{&ai, nb.neighbor, d, r, 1.0 / r};
}

// Stage 2 — EAM pair + embedding-gradient force (same as EAMForceCalculator):
//   F_eam = [dφ/dr + gradF_i×dg_{t(j)}/dr + gradF_j×dg_{t(i)}/dr] × r̂
auto add_eam_force(const PotentialPair &pair, const PotentialArray &density) {
  return [&pair, &density](PairForce &&pf) -> PairForce {
    const Atom &ai = *pf.ai;
    const Atom &aj = *pf.aj;
    // Gate each radial table on its own cutoff (see in_range).
    const auto &phi_pot = pair[ai, aj];
    const auto &g_j = density[aj];
    const auto &g_i = density[ai];
    pf.phi = in_range(phi_pot, pf.r) ? phi_pot.eval(pf.r) : 0.0;
    const double dphi = in_range(phi_pot, pf.r) ? phi_pot.deriv(pf.r) : 0.0;
    const double drho_j = in_range(g_j, pf.r) ? g_j.deriv(pf.r) : 0.0;
    const double drho_i = in_range(g_i, pf.r) ? g_i.deriv(pf.r) : 0.0;
    pf.force +=
        (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * pf.inv_r * pf.d;
    return std::move(pf);
  };
}

// Stage 3 — dipole force (Mishin 2005, derived via ∂E_dip/∂r_i):
//   F_dip = du/r × (μ_i·d − μ_j·d) × d + u × (μ_i − μ_j)
auto add_dipole_force(const PotentialPair &dipole) {
  return [&dipole](PairForce &&pf) -> PairForce {
    const Atom &ai = *pf.ai;
    const Atom &aj = *pf.aj;
    const auto &dip = dipole[ai, aj];
    const double u = in_range(dip, pf.r) ? dip.eval(pf.r) : 0.0;
    const double du = in_range(dip, pf.r) ? dip.deriv(pf.r) : 0.0;
    const double dot_i = ai.mu.dot(pf.d);
    const double dot_j = aj.mu.dot(pf.d);
    pf.force += (du * pf.inv_r * (dot_i - dot_j)) * pf.d + u * (ai.mu - aj.mu);
    return std::move(pf);
  };
}

// Stage 4 — quadrupole force (Mishin 2005, derived via ∂E_quad/∂r_i):
//   F_quad = dw/r × [ν(λ_i,d) + ν(λ_j,d)] × d + 2w × [ξ(λ_i,d) + ξ(λ_j,d)]
//   where ν(M,d)=d^T M d − r²/3 tr(M),  ξ(M,d)=Md − tr(M)/3 d
auto add_quadrupole_force(const PotentialPair &quadrupole) {
  return [&quadrupole](PairForce &&pf) -> PairForce {
    const Atom &ai = *pf.ai;
    const Atom &aj = *pf.aj;
    const auto &quad = quadrupole[ai, aj];
    const double w = in_range(quad, pf.r) ? quad.eval(pf.r) : 0.0;
    const double dw = in_range(quad, pf.r) ? quad.deriv(pf.r) : 0.0;
    const double nu = quad_nu(ai.lambda, pf.d) + quad_nu(aj.lambda, pf.d);
    const Vec3 xi = quad_xi(ai.lambda, pf.d) + quad_xi(aj.lambda, pf.d);
    pf.force += (dw * pf.inv_r * nu) * pf.d + 2.0 * w * xi;
    return std::move(pf);
  };
}

// Stage 5 — commit the pair's energy, force and virial to the configuration.
auto accumulate(Atom &ai, Configuration &cfg) {
  return [&ai, &cfg](PairForce &&pf) -> PairForce {
    ai.calc_force += pf.force;
    cfg.calc_energy += 0.5 * pf.phi;
    // Virial: bond ⊗ force-on-partner = d ⊗ (−fvec); 0.5 for the full list.
    cfg.calc_stress -= 0.5 * pf.d * pf.force.transpose();
    return std::move(pf);
  };
}

} // anonymous namespace

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

double ADPForceCalculator::max_cutoff() const {
  auto max_cutoff = [](const auto &range) {
    return std::transform_reduce(
        range.begin(), range.end(), 0.0,
        [](double a, double b) { return std::max(a, b); },
        [](const auto &p) { return p.span().second; });
  };

  // Cover every radial table: dipole/quadrupole may reach farther than the
  // pair/density tables, and those neighbours must not be truncated (potfit
  // gates each contribution on its own per-table cutoff).
  return std::max({max_cutoff(pair), max_cutoff(density), max_cutoff(dipole),
                   max_cutoff(quadrupole)});
}

void ADPForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  // ── Zero scratch + output ────────────────────────────────────────────────
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  for (auto &atom : cfg.atoms) {
    atom.ZeroForce();
    atom.ZeroScratch();
  }

  // ── Pass 1: accumulate per-atom moments ρ_i, μ_i, λ_i ────────────────────
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
      if (in_range(g, r)) {
        rho += g.eval(r);
      }
      if (in_range(dip, r)) {
        mu += dip.eval(r) * nb.dist;
      }
      if (in_range(quad, r)) {
        lambda += quad.eval(r) * (nb.dist * nb.dist.transpose());
      }
    }

    ai.rho += rho;
    ai.mu += mu;
    ai.lambda += lambda;
  });

  // ── After pass 1: embedding + ADP self-energies, cache gradF_i ───────────
  const double energy = std::transform_reduce(
      cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{}, [&](auto &ai) {
        const auto &emb = embedding[ai];
        ai.gradF = emb.deriv(ai.rho);

        double e = emb.eval(ai.rho);
        e += 0.5 * ai.mu.squaredNorm();

        const double tr_lam = ai.lambda.trace();
        e += 0.5 * (ai.lambda.squaredNorm() - tr_lam * tr_lam / 3.0);

        return e;
      });

  cfg.calc_energy += energy;

  // ── Pass 2: forces ───────────────────────────────────────────────────────
  // Run each i–j bond through the pipeline:
  //   geometry → EAM force → dipole force → quadrupole force → commit
  // std::optional short-circuits coincident atoms (make_pair_force), so each
  // step reads as one clear stage rather than a deeply nested loop body.
  for (Atom &ai : cfg.atoms) {
    for (const NeighborEntry &nb : ai.neighbors) {
      make_pair_force(ai, nb)
          .transform(add_eam_force(pair, density))
          .transform(add_dipole_force(dipole))
          .transform(add_quadrupole_force(quadrupole))
          .transform(accumulate(ai, cfg));
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg)});
}

} // namespace potfit
