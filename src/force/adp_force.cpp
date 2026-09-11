#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/force/eval_scope.hpp"
#include "forcesmith/force/kernels.hpp"
#include "forcesmith/force/param_jacobian.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

namespace forcesmith {

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
  force::accumulate_pair(cfg, ai, pf.d, pf.force, pf.phi);
  return std::move(pf);
}

void ADPForceCalculator::eval_forces(Configuration &cfg) const {
  force::with_eval_scope(cfg, max_cutoff(), conf_index, [&] {
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

        const auto &g = density[aj];
        const auto &dip = dipole[ai, aj];
        const auto &quad = quadrupole[ai, aj];
        rho += eval_gated(g, nb.sites[kSiteGj], r);
        mu += eval_gated(dip, nb.sites[kSiteDipole], r) * nb.dist;
        lambda += eval_gated(quad, nb.sites[kSiteQuad], r) *
                  (nb.dist * nb.dist.transpose());
      }

      ai.rho += rho;
      ai.mu += mu;
      ai.lambda += lambda;
    });

    const double energy = std::transform_reduce(
        cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{}, [&](auto &ai) {
          const auto &emb = embedding[ai];

          force::clamp_embedding(cfg, ai, emb);

          const auto [emb_F, emb_dF] = emb.eval_and_deriv(ai.rho);
          ai.gradF = emb_dF;

          double e = emb_F;
          e += 0.5 * ai.mu.squaredNorm();

          const double tr_lam = ai.lambda.trace();
          e += 0.5 * (ai.lambda.squaredNorm() - tr_lam * tr_lam / 3.0);

          return e;
        });

    cfg.calc_energy += energy;

    for (Atom &ai : cfg.atoms) {
      for (const NeighborEntry &nb : ai.neighbors) {
        make_pair_force(ai, nb)
            .transform(
                std::bind_front(&ADPForceCalculator::add_eam_force, this))
            .transform(
                std::bind_front(&ADPForceCalculator::add_dipole_force, this))
            .transform(std::bind_front(
                &ADPForceCalculator::add_quadrupole_force, this))
            .transform(std::bind_front(&ADPForceCalculator::accumulate,
                                       std::ref(ai), std::ref(cfg)));
      }
    }
  });
}

bool ADPForceCalculator::has_analytic_jacobian() const {
  return force::analytic_jacobian_available(*this);
}

void ADPForceCalculator::write_param_jacobian(Configuration &cfg, int row0,
                                              double energy_weight,
                                              double stress_weight,
                                              Eigen::MatrixXd &fjac) const {
  build_neighbor_list(cfg, max_cutoff());
  const auto cols = force::table_columns(*this);
  const force::JacRows rows(cfg, row0, stress_weight);
  const double inv_volume = 1.0 / bc_volume(cfg.bc);

  const std::size_t na = cfg.atoms.size();
  const int dens0 = cols.start[1], emb0 = cols.start[2];
  const int dip0 = cols.start[3], quad0 = cols.start[4];
  const auto ndens = static_cast<std::size_t>(cols.width(1));
  const auto nemb = static_cast<std::size_t>(cols.width(2));
  const auto ndip = static_cast<std::size_t>(cols.width(3));
  const auto nquad = static_cast<std::size_t>(cols.width(4));

  std::vector<double> rho(na, 0.0), drho(na * ndens, 0.0);
  std::vector<Vec3> mu(na, Vec3::Zero()), dmu(na * ndip, Vec3::Zero());
  std::vector<SymTens> lam(na, SymTens::Zero()),
      dlam(na * nquad, SymTens::Zero());
  std::vector<double> buf;
  force::Partials part;
  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }
      const Atom &aj = *nb.neighbor;
      const auto &g = density[aj];
      const auto &dip = dipole[ai, aj];
      const auto &quad = quadrupole[ai, aj];
      const SymTens dd = nb.dist * nb.dist.transpose();

      if (in_range(g, r)) {
        rho[i] += g.eval(r);
        if (const std::size_t n = g.param_count(); n > 0) {
          buf.resize(n);
          g.param_grad(r, buf);
          const auto base =
              static_cast<std::size_t>(cols.of[1][aj.type] - dens0) + i * ndens;
          for (std::size_t k = 0; k < n; ++k) {
            drho[base + k] += buf[k];
          }
        }
      }
      if (in_range(dip, r)) {
        mu[i] += dip.eval(r) * nb.dist;
        if (const std::size_t n = dip.param_count(); n > 0) {
          buf.resize(n);
          dip.param_grad(r, buf);
          const auto base =
              static_cast<std::size_t>(
                  cols.of[3][pair_ordinal(ai.type, aj.type, dipole.ntypes())] -
                  dip0) +
              i * ndip;
          for (std::size_t k = 0; k < n; ++k) {
            dmu[base + k] += buf[k] * nb.dist;
          }
        }
      }
      if (in_range(quad, r)) {
        lam[i] += quad.eval(r) * dd;
        if (const std::size_t n = quad.param_count(); n > 0) {
          buf.resize(n);
          quad.param_grad(r, buf);
          const auto base =
              static_cast<std::size_t>(
                  cols.of[4][pair_ordinal(ai.type, aj.type,
                                          quadrupole.ntypes())] -
                  quad0) +
              i * nquad;
          for (std::size_t k = 0; k < n; ++k) {
            dlam[base + k] += buf[k] * dd;
          }
        }
      }
    }
  }

  std::vector<double> gradF(na, 0.0), curvF(na, 0.0);
  std::vector<double> dgrad_demb(na * nemb, 0.0);
  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const auto &emb = embedding[ai];
    const auto [rho_begin, rho_end] = emb.span();
    double rho_bar = rho[i];
    double over = 0.0;
    if (rho_bar > rho_end) {
      over = 2.0 * kDummyWeight * 10.0 * (rho_bar - rho_end);
      rho_bar = rho_end;
    } else if (rho_bar < rho_begin) {
      over = -2.0 * kDummyWeight * 10.0 * (rho_begin - rho_bar);
      rho_bar = rho_begin;
    }
    const bool clamped = over != 0.0;
    gradF[i] = emb.deriv(rho_bar);
    curvF[i] = emb.deriv2(rho_bar);

    if (const std::size_t ne = emb.param_count(); ne > 0) {
      buf.resize(ne);
      emb.param_grad(rho_bar, buf);
      for (std::size_t k = 0; k < ne; ++k) {
        fjac(rows.energy, cols.of[2][ai.type] + static_cast<int>(k)) +=
            energy_weight * buf[k];
      }
      emb.dderiv_dparam(rho_bar, buf);
      const auto base =
          static_cast<std::size_t>(cols.of[2][ai.type] - emb0) + i * nemb;
      for (std::size_t k = 0; k < ne; ++k) {
        dgrad_demb[base + k] = buf[k];
      }
    }
    for (std::size_t m = 0; m < ndens; ++m) {
      const double d = drho[i * ndens + m];
      if (d == 0.0) {
        continue;
      }
      if (clamped) {
        fjac(rows.limit, dens0 + static_cast<int>(m)) += over * d;
      } else {
        fjac(rows.energy, dens0 + static_cast<int>(m)) +=
            energy_weight * gradF[i] * d;
      }
    }
    if (clamped) {
      std::ranges::fill(std::span(drho).subspan(i * ndens, ndens), 0.0);
    }

    for (std::size_t m = 0; m < ndip; ++m) {
      fjac(rows.energy, dip0 + static_cast<int>(m)) +=
          energy_weight * mu[i].dot(dmu[i * ndip + m]);
    }
    for (std::size_t m = 0; m < nquad; ++m) {
      const SymTens &dl = dlam[i * nquad + m];
      fjac(rows.energy, quad0 + static_cast<int>(m)) +=
          energy_weight *
          ((lam[i].cwiseProduct(dl)).sum() - lam[i].trace() * dl.trace() / 3.0);
    }
  }

  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const int atom_row = rows.force0 + 3 * static_cast<int>(i);
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }
      const Atom &aj = *nb.neighbor;
      const Vec3 &d = nb.dist;
      const double inv_r = 1.0 / r;
      const std::size_t j = static_cast<std::size_t>(&aj - cfg.atoms.data());

      const auto add_col = [&](int col, const Vec3 &df) {
        force::add_force_column(fjac, rows, atom_row, col, d, df, 0.0,
                                energy_weight, stress_weight, inv_volume);
      };

      const auto &phi_pot = pair[ai, aj];
      if (in_range(phi_pot, r)) {
        part.take(phi_pot, r);
        force::add_radial_bond(
            fjac, rows, atom_row,
            cols.of[0][pair_ordinal(ai.type, aj.type, pair.ntypes())], d,
            inv_r, part, 1.0, /*with_energy=*/true, energy_weight,
            stress_weight, inv_volume);
      }

      const auto &g_j = density[aj];
      const auto &g_i = density[ai];
      const bool gj_in = in_range(g_j, r);
      const bool gi_in = in_range(g_i, r);
      const double dg_j = gj_in ? g_j.deriv(r) : 0.0;
      const double dg_i = gi_in ? g_i.deriv(r) : 0.0;
      if (gj_in) {
        part.take(g_j, r);
        force::add_radial_bond(fjac, rows, atom_row, cols.of[1][aj.type], d,
                               inv_r, part, gradF[i], /*with_energy=*/false,
                               energy_weight, stress_weight, inv_volume);
      }
      if (gi_in) {
        part.take(g_i, r);
        force::add_radial_bond(fjac, rows, atom_row, cols.of[1][ai.type], d,
                               inv_r, part, gradF[j], /*with_energy=*/false,
                               energy_weight, stress_weight, inv_volume);
      }
      for (std::size_t m = 0; m < ndens; ++m) {
        const double chain = curvF[i] * drho[i * ndens + m] * dg_j +
                             curvF[j] * drho[j * ndens + m] * dg_i;
        if (chain != 0.0) {
          add_col(dens0 + static_cast<int>(m), (chain * inv_r) * d);
        }
      }
      for (std::size_t m = 0; m < nemb; ++m) {
        const double chain = dgrad_demb[i * nemb + m] * dg_j +
                             dgrad_demb[j * nemb + m] * dg_i;
        if (chain != 0.0) {
          add_col(emb0 + static_cast<int>(m), (chain * inv_r) * d);
        }
      }

      const auto &dip = dipole[ai, aj];
      if (in_range(dip, r)) {
        const auto [u, du] = dip.eval_and_deriv(r);
        const double dot = mu[i].dot(d) - mu[j].dot(d);
        part.take(dip, r);
        const int ucol =
            cols.of[3][pair_ordinal(ai.type, aj.type, dipole.ntypes())];
        for (const std::size_t k : part.slots()) {
          add_col(ucol + static_cast<int>(k),
                  (part.deriv[k] * inv_r * dot) * d +
                      part.value[k] * (mu[i] - mu[j]));
        }
        for (std::size_t m = 0; m < ndip; ++m) {
          const Vec3 dm = dmu[i * ndip + m] - dmu[j * ndip + m];
          if (!dm.isZero()) {
            add_col(dip0 + static_cast<int>(m),
                    (du * inv_r * dm.dot(d)) * d + u * dm);
          }
        }
      }

      const auto &quad = quadrupole[ai, aj];
      if (in_range(quad, r)) {
        const auto [w, dw] = quad.eval_and_deriv(r);
        const double nu = quad_nu(lam[i], d) + quad_nu(lam[j], d);
        const Vec3 xi = quad_xi(lam[i], d) + quad_xi(lam[j], d);
        part.take(quad, r);
        const int wcol =
            cols.of[4][pair_ordinal(ai.type, aj.type, quadrupole.ntypes())];
        for (const std::size_t k : part.slots()) {
          add_col(wcol + static_cast<int>(k),
                  (part.deriv[k] * inv_r * nu) * d + (2.0 * part.value[k]) * xi);
        }
        for (std::size_t m = 0; m < nquad; ++m) {
          const SymTens dl = dlam[i * nquad + m] + dlam[j * nquad + m];
          if (!dl.isZero()) {
            add_col(quad0 + static_cast<int>(m),
                    (dw * inv_r * quad_nu(dl, d)) * d + (2.0 * w) * quad_xi(dl, d));
          }
        }
      }
    }
  }
}

void ADPForceCalculator::prepare(std::span<Configuration> configs) const {
  build_all_neighbor_lists(configs, max_cutoff());
  // Phase 2 — single-threaded: prime the spline-cache hints for all five
  // radial-table roles a bond drives (φ, g_j, g_i, dipole u, quadrupole w).
  // prepare_site mutates shared spline objects, so it stays serial.
  for (Configuration &cfg : configs) {
    for (Atom &ai : cfg.atoms) {
      const auto &g_i = density[ai];
      for (NeighborEntry &nb : ai.neighbors) {
        const Atom &aj = *nb.neighbor;
        const double r = nb.dist.norm();
        nb.sites[kSitePhi] = force::prime_site(pair[ai, aj], r);
        nb.sites[kSiteGj] = force::prime_site(density[aj], r);
        nb.sites[kSiteGi] = force::prime_site(g_i, r);
        nb.sites[kSiteDipole] = force::prime_site(dipole[ai, aj], r);
        nb.sites[kSiteQuad] = force::prime_site(quadrupole[ai, aj], r);
      }
    }
  }
}

} // namespace forcesmith
