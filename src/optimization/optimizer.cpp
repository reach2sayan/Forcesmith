#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"

namespace potfit {

namespace {

int run_with_solver(std::span<Configuration> configs, ForceCalculator &model,
                    const OptimizerOptions &opts, const Solver &solver) {
  PotfitFunctor functor(configs, model, opts.energy_weight, opts.stress_weight);

  Eigen::VectorXd x(functor.inputs());
  std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); }, model);

  auto residual_fn = [&functor](const Eigen::VectorXd &params) {
    Eigen::VectorXd fvec(functor.values());
    functor(params, fvec);
    return fvec;
  };

  const int status =
      solver.minimize(x, std::move(residual_fn), functor.values());

  // Scatter final params back so caller sees consistent state.
  std::visit([&](auto &m) { m.scatter_params(x, std::size_t{0}); }, model);

  return status;
}

Solver make_solver(const OptimizerOptions &opts) {
  switch (opts.algorithm) {
    case Algorithm::Powell:
      return Solver{EigenHybridSolver{opts.max_iter, opts.xtol}};
    case Algorithm::DE: {
      BoostDESolver s;
      s.mutation_factor       = opts.de.mutation_factor;
      s.crossover_probability = opts.de.crossover_probability;
      s.NP_factor             = opts.de.NP_factor;
      s.max_generations       = opts.de.max_generations;
      s.threads               = opts.de.threads;
      s.seed                  = opts.seed;
      s.lower_bounds          = opts.de.lower_bounds;
      s.upper_bounds          = opts.de.upper_bounds;
      return Solver{std::move(s)};
    }
    default: // Algorithm::LM
      return make_default_solver(opts.max_iter, opts.xtol, opts.ftol);
  }
}

} // namespace

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts) {
  return run_with_solver(configs, model, opts, make_solver(opts));
}

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts, const Solver &solver) {
  return run_with_solver(configs, model, opts, solver);
}

} // namespace potfit
