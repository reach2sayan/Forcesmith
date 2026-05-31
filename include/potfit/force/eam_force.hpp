#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <numeric>

namespace potfit {

// EAM force calculator.
//
// Potential tables for ntypes element types:
//   pair      — φ_{ij}(r)  pair repulsion,  paircol = ntypes*(ntypes+1)/2
//   entries density   — g_i(r)     electron density, ntypes entries embedding —
//   F_i(ρ)     cohesive energy,  ntypes entries
struct EAMForceCalculator : ForceCalculatorBase<EAMForceCalculator> {
  PotentialPair pair;
  PotentialArray density;
  PotentialArray embedding;

  void eval_forces(Configuration &cfg) const;

  int    param_count() const;
  void   gather_params(Eigen::VectorXd &dst, int off) const;
  void   scatter_params(const Eigen::VectorXd &src, int off);
  double max_cutoff() const;
};

static_assert(ForceCalculatorModel<EAMForceCalculator>);

} // namespace potfit
