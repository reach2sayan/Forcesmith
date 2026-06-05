#pragma once

// Angular (pair + three-body) force calculator.
// Mirrors force_ang.c from the original forcesmith.
//
// Energy:
//   E = Σ_{i<j} φ(r_ij)
//     + Σ_i Σ_{j<k ∈ neigh(i)} f(r_ij) × f(r_ik) × g(cos θ_{jik})

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace forcesmith {

// Potential tables for ntypes element types (paircol = ntypes*(ntypes+1)/2):
//   pair    — φ_{ij}(r)      pair repulsion,              paircol entries
//   radial  — f_{ij}(r)      radial three-body modulation, paircol entries
//   angular — g_i(cos θ)     angular function, indexed by the *central* atom
//                             type i (matches forcesmith's col = 2*paircol +
//                             typ_i),
//                                                          ntypes entries
struct AngularForceCalculator : ForceCalculatorBase<AngularForceCalculator>,
                                NoGlobals {
  PotentialPair pair;
  PotentialPair radial;
  PotentialArray angular;

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;
};

static_assert(ForceCalculatorModel<AngularForceCalculator>);

} // namespace forcesmith
