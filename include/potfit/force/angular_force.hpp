#pragma once

// Angular (pair + three-body) force calculator.
// Mirrors force_ang.c from the original potfit.
//
// Energy:
//   E = Σ_{i<j} φ(r_ij)
//     + Σ_i Σ_{j<k ∈ neigh(i)} f(r_ij) × f(r_ik) × g(cos θ_{jik})

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/force/potential_table.hpp"

namespace potfit {

// Potential tables for ntypes element types (paircol = ntypes*(ntypes+1)/2):
//   pair    — φ_{ij}(r)      pair repulsion,              paircol entries
//   radial  — f_{ij}(r)      radial three-body modulation, paircol entries
//   angular — g_{jk}(cos θ)  angular function, indexed by the two *neighbor*
//                             types tj and tk,              paircol entries
struct AngularForceCalculator : ForceCalculatorBase<AngularForceCalculator> {
  PotentialPair pair;
  PotentialPair radial;
  PotentialPair angular;
  void eval_forces(Configuration &cfg) const;
};

static_assert(ForceCalculatorModel<AngularForceCalculator>);

} // namespace potfit
