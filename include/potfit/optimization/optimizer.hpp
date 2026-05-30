#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/optimization/solver.hpp"

#include <span>
#include <vector>

namespace potfit {

struct OptimizerOptions {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;
  double energy_weight = 1.0;
};

// Uses EigenLMSolver by default (backward-compatible).
int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts = {});

// Uses the provided solver — plug in any SolverImpl this way.
int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts,
                  const Solver &solver);

} // namespace potfit
