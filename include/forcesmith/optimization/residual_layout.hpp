#pragma once

// Per-config rows: 3 per atom (forces), energy, 6 stress (kVoigt6, when
// fitted), rho punishment. Every residual and Jacobian writer must agree.

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/types.hpp"
#include "forcesmith/core/voigt.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <numeric>
#include <span>

namespace forcesmith::opt {

FORCE_INLINE std::size_t config_residual_count(const Configuration &cfg,
                                               double stress_weight) {
  return 3 * cfg.atoms.size() + 2 + (stress_weight > 0.0 ? kVoigtCount : 0);
}

FORCE_INLINE std::size_t count_residuals(std::span<Configuration> configs,
                                         double stress_weight) {
  return std::transform_reduce(configs.begin(), configs.end(), std::size_t{0},
                               std::plus<>{}, [&](const Configuration &cfg) {
                                 return config_residual_count(cfg,
                                                              stress_weight);
                               });
}

// Same arithmetic in the same order as before: the residual is bit-identical.
template <class Vec>
  requires requires(Vec &v, int i) {
    { v[i] } -> std::assignable_from<double>;
  }
FORCE_INLINE int write_config_residuals(const Configuration &cfg, Vec &fvec,
                                        int row, double energy_weight,
                                        double stress_weight) {
  for (const Atom &atom : cfg.atoms) {
    fvec[row++] = atom.calc_force[0] - atom.ref.force[0];
    fvec[row++] = atom.calc_force[1] - atom.ref.force[1];
    fvec[row++] = atom.calc_force[2] - atom.ref.force[2];
  }
  fvec[row++] = energy_weight * (cfg.calc_energy - cfg.ref.energy);
  if (stress_weight > 0.0) {
    for (const auto &[i, j] : kVoigt6) {
      fvec[row++] =
          stress_weight * (cfg.calc_stress(i, j) - cfg.ref.stress(i, j));
    }
  }
  fvec[row++] = cfg.calc_limit;
  return row;
}

} // namespace forcesmith::opt
