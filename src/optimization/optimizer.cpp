#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"

namespace potfit {

namespace {

int run_with_solver(std::span<Configuration> configs, ForceCalculator &model,
                    const OptimizerOptions &opts, const Solver &solver) {
  PotfitFunctor functor(configs, model, opts.energy_weight, opts.stress_weight,
                        opts.smooth_weight);

  Eigen::VectorXd x(functor.inputs());
  std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); }, model);

  auto residual_fn = [&functor](const Eigen::VectorXd &params) {
    Eigen::VectorXd fvec(functor.values());
    functor(params, fvec);
    return fvec;
  };

  // Analytic-interface Jacobian: the typed, config-parallel central FD on the
  // functor itself. Solvers that need a Jacobian (LM) use it; the rest ignore
  // it.
  auto jacobian_fn = [&functor](const Eigen::VectorXd &params,
                                Eigen::MatrixXd &fjac) {
    functor.df(params, fjac);
  };

  const int status = solver.minimize(x, std::move(residual_fn),
                                     std::move(jacobian_fn), functor.values());

  // Scatter final params back so caller sees consistent state.
  std::visit([&](auto &m) { m.scatter_params(x, std::size_t{0}); }, model);

  return status;
}

} // namespace

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts) {
  // No solver supplied: fit with the default Levenberg–Marquardt solver. Callers
  // wanting a different algorithm or non-default tuning build a Solver and pass
  // it to the overload below (e.g. via PotFit::set_solver).
  return run_with_solver(configs, model, opts, make_default_solver());
}

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts, const Solver &solver) {
  return run_with_solver(configs, model, opts, solver);
}

} // namespace potfit
