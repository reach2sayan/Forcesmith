#include "potfit/force/adp_force.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace potfit {

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

// Helpers for quadrupole force terms (see header for derivation reference).
// nu(M, d) = d^T M d - r²/3 × tr(M)
static FORCE_INLINE double quad_nu(const SymTens &M, const Vec3 &d) {
  return d.dot(M * d) - d.squaredNorm() / 3.0 * M.trace();
}
// xi(M, d) = M d - tr(M)/3 × d
static FORCE_INLINE Vec3 quad_xi(const SymTens &M, const Vec3 &d) {
  return M * d - (M.trace() / 3.0) * d;
}

void ADPForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  // ── Zero scratch + output ────────────────────────────────────────────────
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  for (auto &atom : cfg.atoms) {
    atom.calc_force = Vec3::Zero();
    atom.rho = 0.0;
    atom.gradF = 0.0;
    atom.mu = Vec3::Zero();
    atom.lambda = SymTens::Zero();
  }

  // ── Pass 1: accumulate ρ_i, μ_i, λ_i ────────────────────────────────────
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const auto &aj = *nb.neighbor;
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }

      ai.rho += density[aj].eval(r);
      ai.mu += dipole[ai, aj].eval(r) * nb.dist;
      ai.lambda += quadrupole[ai, aj].eval(r) * (nb.dist * nb.dist.transpose());
    }
  }

  // ── After pass 1: embedding + ADP self-energies ──────────────────────────
  for (auto &ai : cfg.atoms) {
    cfg.calc_energy += embedding[ai].eval(ai.rho);
    ai.gradF = embedding[ai].deriv(ai.rho);
    cfg.calc_energy += 0.5 * ai.mu.squaredNorm();
    const double tr_lam = ai.lambda.trace();
    cfg.calc_energy += 0.5 * (ai.lambda.squaredNorm() - tr_lam * tr_lam / 3.0);
  }

  // ── Pass 2: forces ───────────────────────────────────────────────────────
  // Per-neighbor-pair force on atom i:
  //
  //   F = F_eam + F_dip + F_quad
  //
  // EAM pair + embedding-gradient (same as EAMForceCalculator):
  //   F_eam = [dφ/dr + gradF_i×dg_{t(j)}/dr + gradF_j×dg_{t(i)}/dr] × r̂
  //
  // Dipole (Mishin 2005, derived via ∂E_dip/∂r_i):
  //   F_dip = du/r × [(μ_i·d − μ_j·d)] × d + u × (μ_i − μ_j)
  //
  // Quadrupole (Mishin 2005, derived via ∂E_quad/∂r_i):
  //   F_quad = dw/r × [ν(λ_i,d) + ν(λ_j,d)] × d + 2w × [ξ(λ_i,d) + ξ(λ_j,d)]
  //   where ν(M,d)=d^T M d − r²/3 tr(M),  ξ(M,d)=Md − tr(M)/3 d
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const auto &aj = *nb.neighbor;
      const Vec3 &d = nb.dist;
      const double r = d.norm();
      if (r < 1e-14) {
        continue;
      }
      const double inv_r = 1.0 / r;

      // EAM pair + embedding terms
      const double phi = pair[ai, aj].eval(r);
      const double dphi = pair[ai, aj].deriv(r);
      const double drho_j = density[aj].deriv(r);
      const double drho_i = density[ai].deriv(r);
      Vec3 fvec = (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * inv_r * d;

      // Dipole force terms
      const double u = dipole[ai, aj].eval(r);
      const double du = dipole[ai, aj].deriv(r);
      const double dot_i = ai.mu.dot(d);
      const double dot_j = aj.mu.dot(d);
      fvec += (du * inv_r * (dot_i - dot_j)) * d + u * (ai.mu - aj.mu);

      // Quadrupole force terms
      const double w = quadrupole[ai, aj].eval(r);
      const double dw = quadrupole[ai, aj].deriv(r);
      const double nu = quad_nu(ai.lambda, d) + quad_nu(aj.lambda, d);
      const Vec3 xi = quad_xi(ai.lambda, d) + quad_xi(aj.lambda, d);
      fvec += (dw * inv_r * nu) * d + 2.0 * w * xi;

      ai.calc_force += fvec;
      cfg.calc_energy += 0.5 * phi;
      // Virial: bond ⊗ force-on-partner = d ⊗ (−fvec); 0.5 for the full list.
      cfg.calc_stress -= 0.5 * d * fvec.transpose();
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg)});
}

} // namespace potfit
