#include "potfit/force/eam_force.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <cmath>
#include <numeric>

namespace potfit {

std::size_t EAMForceCalculator::param_count() const {
  auto count_params = [](const auto &xs) {
    return std::transform_reduce(xs.begin(), xs.end(), std::size_t{0},
                                 std::plus<>{},
                                 [](const auto &p) { return p.param_count(); });
  };
  return count_params(pair) + count_params(density) + count_params(embedding);
}

void EAMForceCalculator::gather_params(Eigen::VectorXd &dst,
                                       std::size_t off) const {
  gather_range(pair, dst, off);
  gather_range(density, dst, off);
  gather_range(embedding, dst, off);
}

void EAMForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                        std::size_t off) {
  scatter_range(pair, src, off);
  scatter_range(density, src, off);
  scatter_range(embedding, src, off);
}

double EAMForceCalculator::max_cutoff() const {
  auto max_range = [](const auto &range) {
    return std::transform_reduce(
        range.begin(), range.end(), 0.0,
        [](double a, double b) { return std::max(a, b); },
        [](const auto &p) { return p.span().second; });
  };
  return std::max(max_range(pair), max_range(density));
}

void EAMForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  // ── Zero all scratch and output fields ──────────────────────────────────
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  for (auto &atom : cfg.atoms) {
    atom.calc_force = Vec3::Zero();
    atom.rho = 0.0;
    atom.gradF = 0.0;
  }

  // ── Pass 1: accumulate electron density ρ_i ─────────────────────────────
  // ρ_i = Σ_{j∈neighbors(i)} g_{t(j)}(r_ij)
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      ai.rho += density[*nb.neighbor].eval(r);
    }
  }

  // ── After pass 1: embedding energy + gradF_i = dF_i/dρ_i ────────────────
  for (auto &ai : cfg.atoms) {
    cfg.calc_energy += embedding[ai].eval(ai.rho);
    ai.gradF = embedding[ai].deriv(ai.rho);
  }

  // ── Pass 2: pair + embedding-gradient forces ─────────────────────────────
  // Force on atom i from neighbor j:
  //   F_ij = [dφ_{ij}/dr + gradF_i × dg_{t(j)}/dr + gradF_j × dg_{t(i)}/dr] ×
  //   r̂_{ij}
  //
  // Using the full neighbor list each pair (i,j) appears twice, so:
  //   - pair energy:     add 0.5 × φ per entry
  //   - embedding force: the cross-gradient term (gradF_j × ...) is
  //     self-consistent because gradF_j was computed in pass 1
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const auto &aj = *nb.neighbor;
      const double r = nb.dist.norm();
      if (r < 1e-14)
        continue;
      const double inv_r = 1.0 / r;

      const double phi = pair[ai, aj].eval(r);
      const double dphi = pair[ai, aj].deriv(r);
      const double drho_j = density[aj].deriv(r);
      const double drho_i = density[ai].deriv(r);

      const double fscale =
          (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * inv_r;
      const Vec3 fvec = fscale * nb.dist;

      ai.calc_force += fvec;
      cfg.calc_energy += 0.5 * phi;
      // Virial: bond ⊗ force-on-partner = dist ⊗ (−fvec); 0.5 for the full
      // list.
      cfg.calc_stress -= 0.5 * nb.dist * fvec.transpose();
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg)});
}

} // namespace potfit
