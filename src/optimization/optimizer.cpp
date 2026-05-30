#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"


namespace potfit {

namespace {

int run_with_solver(std::span<Configuration> configs,
                    std::vector<Potential> &potentials,
                    const OptimizerOptions &opts, const Solver &solver) {
  PotfitFunctor functor(configs, potentials, opts.energy_weight);

  // Gather current parameter values into x.
  Eigen::VectorXd x(functor.inputs());
  {
    int off = 0;
    for (const auto &p : potentials) {
      p.gather_params(x, off);
      off += p.param_count();
    }
  }

  // Build a residual closure over the functor.
  auto residual_fn = [&functor](const Eigen::VectorXd &params) {
    Eigen::VectorXd fvec(functor.values());
    functor(params, fvec);
    return fvec;
  };

  const int status = solver.minimize(x, std::move(residual_fn), functor.values());

  // Scatter the final x back into potentials (functor already scatters on each
  // call, but we do it explicitly here too so callers see a consistent state).
  {
    int off = 0;
    for (auto &p : potentials) {
      p.scatter_params(x, off);
      off += p.param_count();
    }
  }

  return status;
}

} // namespace

int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts) {
  return run_with_solver(configs, potentials, opts,
                         make_default_solver(opts.max_iter, opts.xtol, opts.ftol));
}

int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts, const Solver &solver) {
  return run_with_solver(configs, potentials, opts, solver);
}

} // namespace potfit
