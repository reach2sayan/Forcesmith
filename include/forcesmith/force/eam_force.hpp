#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <span>

namespace forcesmith {

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

  void prepare(std::span<Configuration> configs) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;
  void broadcast_globals();

  // One-time setup after `globals` is populated: mark every linked slot fixed
  // (so per-potential gather/scatter skip it) and broadcast the seed values.
  void finalize_globals();
};

static_assert(ForceCalculatorModel<EAMForceCalculator>);

} // namespace forcesmith
