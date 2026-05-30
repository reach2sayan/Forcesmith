#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"

#include <unsupported/Eigen/NonLinearOptimization>

namespace potfit {

int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts) {
  PotfitFunctor functor(configs, potentials, opts.energy_weight);

  // Gather current parameter values into the initial x vector.
  Eigen::VectorXd x(functor.inputs());
  {
    int off = 0;
    for (const auto &p : potentials) {
      p.gather_params(x, off);
      off += p.param_count();
    }
  }

  Eigen::LevenbergMarquardt<PotfitFunctor> lm(functor);
  lm.parameters.maxfev = opts.max_iter;
  lm.parameters.xtol = opts.xtol;
  lm.parameters.ftol = opts.ftol;

  auto status = lm.minimize(x);

  // Scatter optimized parameters back into potentials.
  {
    int off = 0;
    for (auto &p : potentials) {
      p.scatter_params(x, off);
      off += p.param_count();
    }
  }

  return static_cast<int>(status);
}

} // namespace potfit
