#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/optimization/solver.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace potfit {

enum class Algorithm { LM, Powell, DE, LineSearch };

struct DEOptions {
  double mutation_factor = 0.65;
  double crossover_probability = 0.5;
  std::size_t NP_factor = 15;
  std::size_t max_generations = 1000;
  std::vector<double> lower_bounds; // empty → auto from current x
  std::vector<double> upper_bounds;
};

struct OptimizerOptions {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;
  double energy_weight = 1.0;
  double stress_weight = 0.0;
  double smooth_weight = 0.0; // Tikhonov curvature penalty on free knots
  Algorithm algorithm = Algorithm::LM;
  unsigned seed = 0; // 0 → std::random_device (used by DE)
  DEOptions de;
};

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts = {});

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts, const Solver &solver);

} // namespace potfit
