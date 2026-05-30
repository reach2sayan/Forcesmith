#include "potfit/force/eam_force.hpp"

#include <cmath>
#include <numeric>

namespace potfit {

void EAMForceCalculator::eval_forces(Configuration &cfg) const {
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
      ai.rho += density(nb.neighbor->type).eval(r);
    }
  }

  // ── After pass 1: embedding energy + gradF_i = dF_i/dρ_i ────────────────
  for (auto &ai : cfg.atoms) {
    cfg.calc_energy += embedding(ai.type).eval(ai.rho);
    ai.gradF = embedding(ai.type).deriv(ai.rho);
  }

  // ── Pass 2: pair + embedding-gradient forces ─────────────────────────────
  // Force on atom i from neighbor j:
  //   F_ij = [dφ_{ij}/dr + gradF_i × dg_{t(j)}/dr + gradF_j × dg_{t(i)}/dr] × r̂_{ij}
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

      const double phi    = pair(ai.type, aj.type).eval(r);
      const double dphi   = pair(ai.type, aj.type).deriv(r);
      const double drho_j = density(aj.type).deriv(r);
      const double drho_i = density(ai.type).deriv(r);

      const double fscale = (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * inv_r;
      const Vec3 fvec = fscale * nb.dist;

      ai.calc_force += fvec;
      cfg.calc_energy += 0.5 * phi;
      cfg.calc_stress += 0.5 * nb.dist * fvec.transpose();
    }
  }

  const double rms2 = std::transform_reduce(
      cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{},
      [](const auto &a) { return a.calc_force.squaredNorm(); });
  const double rms =
      cfg.atoms.empty()
          ? 0.0
          : std::sqrt(rms2 / static_cast<double>(cfg.atoms.size()));

  events::on_force_eval(events::ForceEvalStats{conf_index, rms});
}

} // namespace potfit
