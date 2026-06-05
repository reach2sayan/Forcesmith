#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/optimization/solver.hpp"

#include <span>

namespace forcesmith {

// The objective the optimizer minimises — solver-agnostic. Every solver fits the
// same weighted residual vector; per-solver tuning (iteration caps, tolerances,
// seeds, DE parameters) lives inside the concrete Solver, not here. Select a
// solver by handing a fully-built one to Forcesmith::set_solver / run_optimizer;
// with none supplied the default Levenberg–Marquardt solver is used.
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
