#include "forcesmith/force/angular_force.hpp"
#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/force/eval_scope.hpp"
#include "forcesmith/force/kernels.hpp"
#include "forcesmith/force/param_jacobian.hpp"

#include <cmath>
#include <numeric>
#include <ranges>
#include <vector>

namespace forcesmith {

void AngularForceCalculator::eval_forces(Configuration &cfg) const {
  force::with_eval_scope(cfg, max_cutoff(), conf_index, [&] {
    for (auto &ai : cfg.atoms) {
      for (const auto &nb : ai.neighbors) {
        const double r = nb.dist.norm();
        if (r < 1e-14) {
          continue;
        }
        const auto &phi_pot = pair[ai.type, nb.neighbor->type];
        if (!in_range(phi_pot, r)) {
          continue;
        }
        const auto [phi, dphi] = phi_pot.eval_and_deriv(r);
        const Vec3 fvec = (dphi / r) * nb.dist;
        force::accumulate_pair(cfg, ai, nb.dist, fvec, phi);
      }
    }

    std::ranges::for_each(cfg.atoms, [&](auto &ai) {
      const auto &nbs = ai.neighbors;
      const std::size_t nn = nbs.size();

      for (std::size_t jj = 0; jj < nn; ++jj) {
        const auto &nb_j = nbs[jj];
        const Vec3 &d1 = nb_j.dist;
        const double r1 = d1.norm();
        if (r1 < 1e-14) {
          continue;
        }
        const double inv_r1 = 1.0 / r1;

        const auto &rad_j = radial[ai.type, nb_j.neighbor->type];
        if (!in_range(rad_j, r1)) {
          continue;
        }
        const auto [f1, df1] = rad_j.eval_and_deriv(r1);

        for (std::size_t kk = jj + 1; kk < nn; ++kk) {
          const auto &nb_k = nbs[kk];
          const Vec3 &d2 = nb_k.dist;
          const double r2 = d2.norm();
          if (r2 < 1e-14) {
            continue;
          }
          const double inv_r2 = 1.0 / r2;

          const auto &rad_k = radial[ai.type, nb_k.neighbor->type];
          if (!in_range(rad_k, r2)) {
            continue;
          }
          const auto [f2, df2] = rad_k.eval_and_deriv(r2);

          const double c = d1.dot(d2) * inv_r1 * inv_r2;
          const auto [g, dg] = angular[ai.type].eval_and_deriv(c);

          cfg.calc_energy += f1 * f2 * g;

          force::accumulate_triplet(
              cfg, ai, const_cast<Atom &>(*nb_j.neighbor),
              const_cast<Atom &>(*nb_k.neighbor), d1, d2,
              force::three_body_forces(d1, d2, inv_r1, inv_r2, c, f1, df1, f2,
                                       df2, g, dg));
        }
      }
    });
  });
}

bool AngularForceCalculator::has_analytic_jacobian() const {
  return force::analytic_jacobian_available(*this);
}

void AngularForceCalculator::write_param_jacobian(Configuration &cfg, int row0,
                                                  double energy_weight,
                                                  double stress_weight,
                                                  Eigen::MatrixXd &fjac) const {
  build_neighbor_list(cfg, max_cutoff());
  const auto cols = force::table_columns(*this);
  const force::JacRows rows(cfg, row0, stress_weight);
  const double inv_volume = 1.0 / bc_volume(cfg.bc);
  const auto index_of = [&](const Atom &a) {
    return static_cast<int>(&a - cfg.atoms.data());
  };

  force::Partials part;

  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const int atom_row = rows.force0 + 3 * static_cast<int>(i);
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }
      const auto &phi_pot = pair[ai.type, nb.neighbor->type];
      if (!in_range(phi_pot, r)) {
        continue;
      }
      part.take(phi_pot, r);
      force::add_radial_bond(
          fjac, rows, atom_row,
          cols.of[0][pair_ordinal(ai.type, nb.neighbor->type, pair.ntypes())],
          nb.dist, 1.0 / r, part, 1.0, /*with_energy=*/true, energy_weight,
          stress_weight, inv_volume);
    }
  }

  for (const Atom &ai : cfg.atoms) {
    const auto &nbs = ai.neighbors;
    const int row_i = rows.force0 + 3 * index_of(ai);

    const auto add_triplet = [&](int col, const force::TripletForces &t,
                                 int row_j, int row_k, const Vec3 &d1,
                                 const Vec3 &d2, double denergy) {
      for (int a = 0; a < 3; ++a) {
        fjac(row_i + a, col) += t.fi[a];
        fjac(row_j + a, col) += t.fj[a];
        fjac(row_k + a, col) += t.fk[a];
      }
      fjac(rows.energy, col) += energy_weight * denergy;
      if (rows.stress0 >= 0) {
        int v = 0;
        for (const auto &[a, b] : kVoigt6) {
          fjac(rows.stress0 + v++, col) +=
              stress_weight * (d1[a] * t.fj[b] + d2[a] * t.fk[b]) * inv_volume;
        }
      }
    };

    for (std::size_t jj = 0; jj < nbs.size(); ++jj) {
      const auto &nb_j = nbs[jj];
      const Vec3 &d1 = nb_j.dist;
      const double r1 = d1.norm();
      if (r1 < 1e-14) {
        continue;
      }
      const double inv_r1 = 1.0 / r1;
      const auto &rad_j = radial[ai.type, nb_j.neighbor->type];
      if (!in_range(rad_j, r1)) {
        continue;
      }
      const auto [f1, df1] = rad_j.eval_and_deriv(r1);
      const int row_j = rows.force0 + 3 * index_of(*nb_j.neighbor);

      for (std::size_t kk = jj + 1; kk < nbs.size(); ++kk) {
        const auto &nb_k = nbs[kk];
        const Vec3 &d2 = nb_k.dist;
        const double r2 = d2.norm();
        if (r2 < 1e-14) {
          continue;
        }
        const double inv_r2 = 1.0 / r2;
        const auto &rad_k = radial[ai.type, nb_k.neighbor->type];
        if (!in_range(rad_k, r2)) {
          continue;
        }
        const auto [f2, df2] = rad_k.eval_and_deriv(r2);
        const double c = d1.dot(d2) * inv_r1 * inv_r2;
        const auto [g, dg] = angular[ai.type].eval_and_deriv(c);
        const int row_k = rows.force0 + 3 * index_of(*nb_k.neighbor);

        part.take(rad_j, r1);
        if (part) {
          const int col0 =
              cols.of[1][pair_ordinal(ai.type, nb_j.neighbor->type,
                                      radial.ntypes())];
          for (const std::size_t m : part.slots()) {
            add_triplet(col0 + static_cast<int>(m),
                        force::three_body_forces(d1, d2, inv_r1, inv_r2, c,
                                                 part.value[m], part.deriv[m], f2, df2, g,
                                                 dg),
                        row_j, row_k, d1, d2, part.value[m] * f2 * g);
          }
        }

        part.take(rad_k, r2);
        if (part) {
          const int col0 =
              cols.of[1][pair_ordinal(ai.type, nb_k.neighbor->type,
                                      radial.ntypes())];
          for (const std::size_t m : part.slots()) {
            add_triplet(col0 + static_cast<int>(m),
                        force::three_body_forces(d1, d2, inv_r1, inv_r2, c, f1,
                                                 df1, part.value[m], part.deriv[m], g, dg),
                        row_j, row_k, d1, d2,
                        f1 * part.value[m] * g);
          }
        }

        const auto &ang = angular[ai.type];
        part.take(ang, c);
        if (part) {
          const int col0 = cols.of[2][ai.type];
          for (const std::size_t m : part.slots()) {
            add_triplet(col0 + static_cast<int>(m),
                        force::three_body_forces(d1, d2, inv_r1, inv_r2, c, f1,
                                                 df1, f2, df2, part.value[m], part.deriv[m]),
                        row_j, row_k, d1, d2, f1 * f2 * part.value[m]);
          }
        }
      }
    }
  }
}

} // namespace forcesmith
