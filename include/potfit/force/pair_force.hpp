#pragma once

#include "potfit/force/force_calculator.hpp"

namespace potfit {

struct PairForceCalculator {
  std::uint64_t conf_index = 0;

  void eval_forces(Configuration &cfg) const;
};

static_assert(ForceCalculatorModel<PairForceCalculator>);

} // namespace potfit
