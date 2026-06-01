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
struct EAMForceCalculator : ForceCalculatorBase<EAMForceCalculator>, WithGlobals {
  PotentialPair pair;
  PotentialArray density;
  PotentialArray embedding;

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  double max_cutoff() const;

  // Write each global's value into every potential slot it is linked to. Called
  // at the end of scatter_params (and once after parsing) so the linked, fixed
  // slots always hold the current shared value before eval_forces runs.
  void broadcast_globals();

  // One-time setup after `globals` is populated: mark every linked slot fixed
  // (so per-potential gather/scatter skip it) and broadcast the seed values.
  void finalize_globals();
};

static_assert(ForceCalculatorModel<EAMForceCalculator>);

} // namespace potfit
