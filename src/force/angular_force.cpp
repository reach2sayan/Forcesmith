#include "potfit/force/angular_force.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <cmath>
#include <numeric>

namespace potfit {

std::size_t AngularForceCalculator::param_count() const {
  auto count_params = [](const auto &range) {
    return std::transform_reduce(range.begin(), range.end(), std::size_t{0},
                                 std::plus<>{},
                                 [](const auto &p) { return p.param_count(); });
  };

  return count_params(pair) + count_params(radial) + count_params(angular);
}

void AngularForceCalculator::gather_params(Eigen::VectorXd &dst,
                                           std::size_t off) const {
  gather_range(pair, dst, off);
  gather_range(radial, dst, off);
  gather_range(angular, dst, off);
}

void AngularForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                            std::size_t off) {
  scatter_range(pair, src, off);
  scatter_range(radial, src, off);
  scatter_range(angular, src, off);
}

double AngularForceCalculator::max_cutoff() const {
  auto max_range = [](const auto &range) {
    return std::transform_reduce(
        range.begin(), range.end(), 0.0,
        [](double a, double b) { return std::max(a, b); },
        [](const auto &p) { return p.span().second; });
  };
  return std::max(max_range(pair), max_range(radial));
}

void AngularForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  // ── Zero output ──────────────────────────────────────────────────────────
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  std::ranges::for_each(cfg.atoms,
                        [](auto &a) { a.calc_force = Vec3::Zero(); });

  // ── Pair force loop (factor 0.5 — full neighbor list counts each pair twice)
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }
      const double phi = pair[ai.type, nb.neighbor->type].eval(r);
      const double dphi = pair[ai.type, nb.neighbor->type].deriv(r);
      const Vec3 fvec = (dphi / r) * nb.dist;
      ai.calc_force += fvec;
      cfg.calc_energy += 0.5 * phi;
      // Virial: bond ⊗ force-on-partner = dist ⊗ (−fvec); 0.5 for the full
      // list.
      cfg.calc_stress -= 0.5 * nb.dist * fvec.transpose();
    }
  }

  // ── Three-body loop ──────────────────────────────────────────────────────
  // For each central atom i, iterate ordered pairs (j < k) from its neighbor
  // list and compute:
  //   E_ijk = f(r_ij) × f(r_ik) × g(cos θ_{jik})
  //
  // Gradient derivation (d₁ = r_j − r_i, d₂ = r_k − r_i, c = cos θ):
  //   A = c/r₁² − 1/(r₁r₂),   B = c/r₂² − 1/(r₁r₂)
  //
  //   F_i = (df1/r1 · f2 · g) d₁ + (f1 · df2/r2 · g) d₂
  //         − f1·f2·dg · (A·d₁ + B·d₂)
  //   F_j = −(df1/r1 · f2 · g) d₁
  //         − f1·f2·dg · (d₂/(r₁r₂) − c·d₁/r₁²)
  //   F_k = −(f1 · df2/r2 · g) d₂
  //         − f1·f2·dg · (d₁/(r₁r₂) − c·d₂/r₂²)
  //
  // F_i + F_j + F_k = 0  (Newton's 3rd law holds term-by-term).
  //
  // Forces on j and k are written via const_cast: cfg is non-const and the
  // pointers come from cfg.atoms, so this is well-defined.

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

      const double f1 = radial[ai.type, nb_j.neighbor->type].eval(r1);
      const double df1 = radial[ai.type, nb_j.neighbor->type].deriv(r1);

      for (std::size_t kk = jj + 1; kk < nn; ++kk) {
        const auto &nb_k = nbs[kk];
        const Vec3 &d2 = nb_k.dist;
        const double r2 = d2.norm();
        if (r2 < 1e-14) {
          continue;
        }
        const double inv_r2 = 1.0 / r2;

        const double f2 = radial[ai.type, nb_k.neighbor->type].eval(r2);
        const double df2 = radial[ai.type, nb_k.neighbor->type].deriv(r2);

        const double c = d1.dot(d2) * inv_r1 * inv_r2;
        // g indexed by the central atom's type (matches potfit force_ang.c).
        const double g = angular[ai.type].eval(c);
        const double dg = angular[ai.type].deriv(c);

        cfg.calc_energy += f1 * f2 * g;

        const double inv_r1r2 = inv_r1 * inv_r2;
        const double A = c * inv_r1 * inv_r1 - inv_r1r2;
        const double B = c * inv_r2 * inv_r2 - inv_r1r2;

        const Vec3 fi = (df1 * inv_r1 * f2 * g) * d1 +
                        (f1 * df2 * inv_r2 * g) * d2 -
                        (f1 * f2 * dg) * (A * d1 + B * d2);

        const Vec3 fj =
            -(df1 * inv_r1 * f2 * g) * d1 -
            (f1 * f2 * dg) * (inv_r1r2 * d2 - c * inv_r1 * inv_r1 * d1);

        const Vec3 fk =
            -(f1 * df2 * inv_r2 * g) * d2 -
            (f1 * f2 * dg) * (inv_r1r2 * d1 - c * inv_r2 * inv_r2 * d2);

        ai.calc_force += fi;
        const_cast<Atom &>(*nb_j.neighbor).calc_force += fj;
        const_cast<Atom &>(*nb_k.neighbor).calc_force += fk;

        // Virial: bond ⊗ force-on-partner (fj, fk are applied to atoms j, k).
        cfg.calc_stress += d1 * fj.transpose();
        cfg.calc_stress += d2 * fk.transpose();
      }
    }
  });

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg)});
}

} // namespace potfit
