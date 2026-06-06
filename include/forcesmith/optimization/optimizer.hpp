#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/optimization/solver.hpp"

#include <span>

namespace forcesmith {

struct OptimizerOptions {
  double energy_weight = 1.0;
  double stress_weight = 0.0;
  double smooth_weight = 0.0; // Tikhonov curvature penalty on free knots
};

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts = {});

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts, const Solver &solver);

} // namespace forcesmith
