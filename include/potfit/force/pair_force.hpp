#pragma once

#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"
#include <Eigen/Core>
#include <vector>

namespace potfit {

struct PairForceCalculator {
  std::size_t ntypes = 1;
  std::uint64_t conf_index = 0;
  PotentialPair pair; // owned pair potentials; eval_forces uses these directly

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void        gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void        scatter_params(const Eigen::VectorXd &src, std::size_t off);
  double      max_cutoff() const;
};

static_assert(ForceCalculatorModel<PairForceCalculator>);

// Build a PairForceCalculator that owns the given flat potential list.
// ntypes is inferred from paircol = ntypes*(ntypes+1)/2.
PairForceCalculator make_pair_force_calculator(std::vector<Potential> potentials);

} // namespace potfit
