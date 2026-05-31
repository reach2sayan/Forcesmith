#pragma once

// Angular (pair + three-body) force calculator.
// Mirrors force_ang.c from the original potfit.
//
// Energy:
//   E = Σ_{i<j} φ(r_ij)
//     + Σ_i Σ_{j<k ∈ neigh(i)} f(r_ij) × f(r_ik) × g(cos θ_{jik})

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>

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

  int param_count() const {
    int count = 0;
    for (const auto& p : pair)    count += p.param_count();
    for (const auto& p : radial)  count += p.param_count();
    for (const auto& p : angular) count += p.param_count();
    return count;
  }

  void gather_params(Eigen::VectorXd& dst, int off) const {
    for (const auto& p : pair)    { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : radial)  { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : angular) { p.gather_params(dst, off); off += p.param_count(); }
  }

  void scatter_params(const Eigen::VectorXd& src, int off) {
    for (auto& p : pair)    { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : radial)  { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : angular) { p.scatter_params(src, off); off += p.param_count(); }
  }

  double max_cutoff() const {
    double rcut = 0.0;
    for (const auto& p : pair)   rcut = std::max(rcut, p.span().second);
    for (const auto& p : radial) rcut = std::max(rcut, p.span().second);
    return rcut;
  }
};

static_assert(ForceCalculatorModel<AngularForceCalculator>);

} // namespace potfit
