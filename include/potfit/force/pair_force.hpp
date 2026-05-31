#pragma once

#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>

namespace potfit {

struct PairForceCalculator {
  int           ntypes     = 1;
  std::uint64_t conf_index = 0;
  PotentialPair pair; // owned pair potentials; eval_forces uses these directly

  // eval_forces rebuilds the neighbor list from pair and evaluates forces.
  void eval_forces(Configuration &cfg) const;

  int param_count() const {
    int count = 0;
    for (const auto& p : pair) count += p.param_count();
    return count;
  }

  void gather_params(Eigen::VectorXd& dst, int off) const {
    for (const auto& p : pair) { p.gather_params(dst, off); off += p.param_count(); }
  }

  void scatter_params(const Eigen::VectorXd& src, int off) {
    for (auto& p : pair) { p.scatter_params(src, off); off += p.param_count(); }
  }

  double max_cutoff() const {
    double rcut = 0.0;
    for (const auto& p : pair) rcut = std::max(rcut, p.span().second);
    return rcut;
  }
};

static_assert(ForceCalculatorModel<PairForceCalculator>);

} // namespace potfit
