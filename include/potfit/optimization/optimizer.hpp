#pragma once

// Step 10: thin driver wrapping Eigen::LevenbergMarquardt.

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"

#include <span>
#include <vector>

namespace potfit {

struct OptimizerOptions {
  int max_iter = 500;
  double xtol = 1e-7; // relative step-size tolerance
  double ftol = 1e-7; // relative function-value tolerance
  double energy_weight =
      1.0; // w_E: weight of energy residuals vs. force residuals
};

// Returns Eigen::LevenbergMarquardtSpace::Status cast to int.
// Scatters optimized parameters back into potentials on return.
int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts = {});

} // namespace potfit
