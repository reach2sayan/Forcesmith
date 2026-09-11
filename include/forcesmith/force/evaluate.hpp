#pragma once

#include "forcesmith/core/atom.hpp"              // Configuration
#include "forcesmith/core/types.hpp"             // Vec3, SymTens
#include "forcesmith/force/force_calculator.hpp" // ForceCalculator variant

#include <vector>

namespace forcesmith::force {

struct EvalResult {
  double energy = 0.0;              // calc_energy
  std::vector<Vec3> forces;         // per-atom, indexed parallel to cfg.atoms
  SymTens stress = SymTens::Zero(); // calc_stress (virial / volume)
  double limit = 0.0; // EAM/ADP out-of-range penalty (0 for others)
};

EvalResult evaluate(const ForceCalculator &calc, const Configuration &cfg);

} // namespace forcesmith::force
