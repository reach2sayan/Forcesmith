#pragma once

#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"
#include <Eigen/Core>

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

} // namespace potfit
