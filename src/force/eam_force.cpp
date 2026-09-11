#include "forcesmith/force/eam_force.hpp"
#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/force/eval_scope.hpp"
#include "forcesmith/force/kernels.hpp"
#include "forcesmith/force/param_jacobian.hpp"
#include "forcesmith/potentials/spline.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <ranges>

namespace forcesmith {

namespace {
FORCE_INLINE double eval_gated(const SplinePotential *sp,
                               const RadialPotential &p, SiteId site,
                               double r) {
  if (site.cacheable()) {
    return sp ? sp->eval_at(site.index()) : p.eval_at(site);
  }
  return in_range(p, r) ? p.eval(r) : 0.0;
}
FORCE_INLINE double deriv_gated(const SplinePotential *sp,
                                const RadialPotential &p, SiteId site,
                                double r) {
  if (site.cacheable()) {
    return sp ? sp->deriv_at(site.index()) : p.deriv_at(site);
  }
  return in_range(p, r) ? p.deriv(r) : 0.0;
}
FORCE_INLINE std::pair<double, double>
eval_deriv_gated(const SplinePotential *sp, const RadialPotential &p,
                 SiteId site, double r) {
  if (site.cacheable()) {
    return sp ? sp->eval_and_deriv_at(site.index()) : p.eval_and_deriv_at(site);
  }
  return in_range(p, r) ? p.eval_and_deriv(r) : std::pair{0.0, 0.0};
}
} // namespace

void EAMForceCalculator::eval_forces(Configuration &cfg) const {
  force::with_eval_scope(cfg, max_cutoff(), conf_index, [&] {
    TypeArray<const SplinePotential *> dens_sp;
    dens_sp.reserve(density.size());
    std::ranges::transform(
        density, std::back_inserter(dens_sp),
        [](const auto &g) { return g.template target<SplinePotential>(); });

    SymmetricMatrix<const SplinePotential *> pair_sp;
    pair_sp.reserve(pair.ntypes());
    std::ranges::transform(
        pair, std::back_inserter(pair_sp),
        [](const auto &p) { return p.template target<SplinePotential>(); });

    for (auto &ai : cfg.atoms) {
      for (const auto &nb : ai.neighbors) {
        const double r = nb.dist.norm();
        const auto &aj = *nb.neighbor;
        const auto &g = density[aj];
        ai.rho += eval_gated(dens_sp[aj], g, nb.sites[kSiteGj], r);
      }
    }

    for (auto &ai : cfg.atoms) {
      force::clamp_embedding(cfg, ai, embedding[ai]);
      const auto [emb_F, emb_dF] = embedding[ai].eval_and_deriv(ai.rho);
      cfg.calc_energy += emb_F;
      ai.gradF = emb_dF;
    }

    for (auto &ai : cfg.atoms) {
      for (const auto &nb : ai.neighbors) {
        const auto &aj = *nb.neighbor;
        const double r = nb.dist.norm();
        if (r < 1e-14) {
          continue;
        }
        const double inv_r = 1.0 / r;

        const auto &phi_pot = pair[ai, aj];
        const auto &g_j = density[aj];
        const auto &g_i = density[ai];
        const auto [phi, dphi] =
            eval_deriv_gated(pair_sp[ai, aj], phi_pot, nb.sites[kSitePhi], r);
        const double drho_j =
            deriv_gated(dens_sp[aj], g_j, nb.sites[kSiteGj], r);
        const double drho_i =
            deriv_gated(dens_sp[ai], g_i, nb.sites[kSiteGi], r);

        const double fscale =
            (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * inv_r;
        const Vec3 fvec = fscale * nb.dist;

        force::accumulate_pair(cfg, ai, nb.dist, fvec, phi);
      }
    }
  });
}

bool EAMForceCalculator::has_analytic_jacobian() const {
  return force::analytic_jacobian_available(*this);
}

void EAMForceCalculator::write_param_jacobian(Configuration &cfg, int row0,
                                              double energy_weight,
                                              double stress_weight,
                                              Eigen::MatrixXd &fjac) const {
  build_neighbor_list(cfg, max_cutoff());
  const auto cols = force::table_columns(*this);
  const force::JacRows rows(cfg, row0, stress_weight);
  const double inv_volume = 1.0 / bc_volume(cfg.bc);

  const std::size_t na = cfg.atoms.size();
  const int dens0 = cols.start[1];
  const auto ndens = static_cast<std::size_t>(cols.width(1));

  std::vector<double> rho(na, 0.0);
  std::vector<double> drho(na * ndens, 0.0);
  std::vector<double> buf;
  force::Partials part;
  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      const Atom &aj = *nb.neighbor;
      const auto &g = density[aj];
      if (!in_range(g, r)) {
        continue;
      }
      rho[i] += g.eval(r);
      const std::size_t n = g.param_count();
      if (n == 0) {
        continue;
      }
      buf.resize(n);
      g.param_grad(r, buf);
      const auto base =
          static_cast<std::size_t>(cols.of[1][aj.type] - dens0) + i * ndens;
      for (std::size_t k = 0; k < n; ++k) {
        drho[base + k] += buf[k];
      }
    }
  }

  std::vector<double> gradF(na, 0.0), curvF(na, 0.0);
  std::vector<double> dgrad_demb;
  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const auto &emb = embedding[ai];
    const auto [rho_begin, rho_end] = emb.span();
    double rho_bar = rho[i];
    double over = 0.0; // d(punishment)/d(rho), via the signed overshoot
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

    const std::size_t ne = emb.param_count();
    if (ne > 0) {
      buf.resize(ne);
      emb.param_grad(rho_bar, buf);
      for (std::size_t k = 0; k < ne; ++k) {
        fjac(rows.energy, cols.of[2][ai.type] + static_cast<int>(k)) +=
            energy_weight * buf[k];
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
    rho[i] = rho_bar; // the point every later embedding partial is taken at
  }

  const auto nemb = static_cast<std::size_t>(cols.width(2));
  dgrad_demb.assign(na * nemb, 0.0);
  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const auto &emb = embedding[ai];
    const std::size_t ne = emb.param_count();
    if (ne == 0) {
      continue;
    }
    buf.resize(ne);
    emb.dderiv_dparam(rho[i], buf);
    const auto base =
        static_cast<std::size_t>(cols.of[2][ai.type] - cols.start[2]) +
        i * nemb;
    for (std::size_t k = 0; k < ne; ++k) {
      dgrad_demb[base + k] = buf[k];
    }
  }

  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const int atom_row = rows.force0 + 3 * static_cast<int>(i);
    const auto add_col = [&](int col, double dfscale, const Vec3 &d,
                             double inv_r) {
      force::add_force_column(fjac, rows, atom_row, col, d,
                              (dfscale * inv_r) * d, 0.0, energy_weight,
                              stress_weight, inv_volume);
    };

    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }
      const Atom &aj = *nb.neighbor;
      const double inv_r = 1.0 / r;
      const std::size_t j = static_cast<std::size_t>(&aj - cfg.atoms.data());

      const auto &phi_pot = pair[ai, aj];
      if (in_range(phi_pot, r)) {
        part.take(phi_pot, r);
        force::add_radial_bond(
            fjac, rows, atom_row,
            cols.of[0][pair_ordinal(ai.type, aj.type, pair.ntypes())], nb.dist,
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
        force::add_radial_bond(fjac, rows, atom_row, cols.of[1][aj.type],
                               nb.dist, inv_r, part, gradF[i],
                               /*with_energy=*/false, energy_weight,
                               stress_weight, inv_volume);
      }
      if (gi_in) {
        part.take(g_i, r);
        force::add_radial_bond(fjac, rows, atom_row, cols.of[1][ai.type],
                               nb.dist, inv_r, part, gradF[j],
                               /*with_energy=*/false, energy_weight,
                               stress_weight, inv_volume);
      }

      for (std::size_t m = 0; m < ndens; ++m) {
        const double chain = curvF[i] * drho[i * ndens + m] * dg_j +
                             curvF[j] * drho[j * ndens + m] * dg_i;
        if (chain != 0.0) {
          add_col(dens0 + static_cast<int>(m), chain, nb.dist, inv_r);
        }
      }

      for (std::size_t m = 0; m < nemb; ++m) {
        const double chain = dgrad_demb[i * nemb + m] * dg_j +
                             dgrad_demb[j * nemb + m] * dg_i;
        if (chain != 0.0) {
          add_col(cols.start[2] + static_cast<int>(m), chain, nb.dist, inv_r);
        }
      }
    }
  }
}

void EAMForceCalculator::prepare(std::span<Configuration> configs) const {
  build_all_neighbor_lists(configs, max_cutoff());
  // Phase 2 — single-threaded (thread unsafe): prime the phi / g_j / g_i hints.
  for (Configuration &cfg : configs) {
    for (Atom &ai : cfg.atoms) {
      const auto &g_i = density[ai];
      for (NeighborEntry &nb : ai.neighbors) {
        const Atom &aj = *nb.neighbor;
        const double r = nb.dist.norm();
        nb.sites[kSitePhi] = force::prime_site(pair[ai, aj], r);
        nb.sites[kSiteGj] = force::prime_site(density[aj], r);
        nb.sites[kSiteGi] = force::prime_site(g_i, r);
      }
    }
  }
}

} // namespace forcesmith
