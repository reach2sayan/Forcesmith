#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace potfit {

// EAM force calculator.
//
// Potential tables for ntypes element types:
//   pair      — φ_{ij}(r)  pair repulsion,  paircol = ntypes*(ntypes+1)/2 entries
//   density   — g_i(r)     electron density, ntypes entries
//   embedding — F_i(ρ)     cohesive energy,  ntypes entries
struct EAMForceCalculator : ForceCalculatorBase<EAMForceCalculator> {
  PotentialPair  pair;
  PotentialArray density;
  PotentialArray embedding;

  void eval_forces(Configuration &cfg) const;

  int param_count() const {
    int count = 0;
    for (const auto& p : pair)      count += p.param_count();
    for (const auto& p : density)   count += p.param_count();
    for (const auto& p : embedding) count += p.param_count();
    return count;
  }

  void gather_params(Eigen::VectorXd& dst, int off) const {
    for (const auto& p : pair)      { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : density)   { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : embedding) { p.gather_params(dst, off); off += p.param_count(); }
  }

  void scatter_params(const Eigen::VectorXd& src, int off) {
    for (auto& p : pair)      { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : density)   { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : embedding) { p.scatter_params(src, off); off += p.param_count(); }
  }

  double max_cutoff() const {
    double rcut = 0.0;
    for (const auto& p : pair)    rcut = std::max(rcut, p.span().second);
    for (const auto& p : density) rcut = std::max(rcut, p.span().second);
    return rcut;
  }
};

static_assert(ForceCalculatorModel<EAMForceCalculator>);

} // namespace potfit
