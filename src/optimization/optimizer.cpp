#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"

namespace potfit {

namespace {

int run_with_solver(std::span<Configuration> configs,
                    ForceCalculator &model,
                    const OptimizerOptions &opts, const Solver &solver) {
  PotfitFunctor functor(configs, model, opts.energy_weight, opts.stress_weight);

  Eigen::VectorXd x(functor.inputs());
  std::visit([&](const auto& m){ m.gather_params(x, std::size_t{0}); }, model);

  auto residual_fn = [&functor](const Eigen::VectorXd &params) {
    Eigen::VectorXd fvec(functor.values());
    functor(params, fvec);
    return fvec;
  };

  const int status = solver.minimize(x, std::move(residual_fn), functor.values());

  // Scatter final params back so caller sees consistent state.
  std::visit([&](auto& m){ m.scatter_params(x, std::size_t{0}); }, model);

  return status;
}

} // namespace

int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts) {
  return run_optimizer(configs, potentials, opts,
                       make_default_solver(opts.max_iter, opts.xtol, opts.ftol));
}

int run_optimizer(std::span<Configuration> configs,
                  std::vector<Potential> &potentials,
                  const OptimizerOptions &opts, const Solver &solver) {
  // Wrap the flat span in a PairForceCalculator for the generic path.
  PotfitFunctor functor(configs, std::span<Potential>{potentials}, opts.energy_weight, opts.stress_weight);

  Eigen::VectorXd x(functor.inputs());
  {
    std::size_t off = 0;
    for (const auto &p : potentials) { p.gather_params(x, off); off += p.param_count(); }
  }

  auto residual_fn = [&functor](const Eigen::VectorXd &params) {
    Eigen::VectorXd fvec(functor.values());
    functor(params, fvec);
    return fvec;
  };

  const int status = solver.minimize(x, std::move(residual_fn), functor.values());

  // Scatter final x back into the caller's potentials span.
  {
    std::size_t off = 0;
    for (auto &p : potentials) { p.scatter_params(x, off); off += p.param_count(); }
  }

  return status;
}

int run_optimizer(std::span<Configuration> configs,
                  ForceCalculator &model,
                  const OptimizerOptions &opts) {
  return run_optimizer(configs, model, opts,
                       make_default_solver(opts.max_iter, opts.xtol, opts.ftol));
}

int run_optimizer(std::span<Configuration> configs,
                  ForceCalculator &model,
                  const OptimizerOptions &opts, const Solver &solver) {
  return run_with_solver(configs, model, opts, solver);
}

} // namespace potfit
