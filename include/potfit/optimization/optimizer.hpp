#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/optimization/solver.hpp"

#include <span>

namespace potfit {

struct OptimizerOptions {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;
  double energy_weight = 1.0;
  double stress_weight = 0.0;
};

int run_optimizer(std::span<Configuration> configs,
                  ForceCalculator &model,
                  const OptimizerOptions &opts = {});

int run_optimizer(std::span<Configuration> configs,
                  ForceCalculator &model,
                  const OptimizerOptions &opts,
                  const Solver &solver);

} // namespace potfit
