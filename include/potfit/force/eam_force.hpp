#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/force/potential_table.hpp"

namespace potfit {

// EAM force calculator.
//
// Potential tables for ntypes element types:
//   pair      — φ_{ij}(r)  pair repulsion,  paircol = ntypes*(ntypes+1)/2 entries
//   density   — g_i(r)     electron density contributed by a type-i atom, ntypes entries
//   embedding — F_i(ρ)     cohesive energy at density ρ for a type-i atom, ntypes entries
struct EAMForceCalculator : ForceCalculatorBase<EAMForceCalculator> {
  PotentialPair pair;
  PotentialArray density;
  PotentialArray embedding;

  void eval_forces(Configuration &cfg) const;
};

static_assert(ForceCalculator<EAMForceCalculator>);

} // namespace potfit
