#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"

#include <iostream>

namespace potfit {

namespace {

int run_with_solver(std::span<Configuration> configs, ForceCalculator &model,
                    const OptimizerOptions &opts, const Solver &solver) {
  PotfitFunctor functor(configs, model, opts.energy_weight, opts.stress_weight,
                        opts.smooth_weight);

  Eigen::VectorXd x(functor.inputs());
  std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); }, model);

  // Per-parameter box constraints, aligned with x (±∞ where unbounded).
  Eigen::VectorXd lower(functor.inputs()), upper(functor.inputs());
  std::visit(
      [&](const auto &m) { m.gather_bounds(lower, upper, std::size_t{0}); },
      model);

  // Warn once if a real bound was supplied but the chosen solver ignores it.
  const bool has_finite_bound =
      lower.array().isFinite().any() || upper.array().isFinite().any();
  if (has_finite_bound && !solver.honors_bounds()) {
    std::cerr << "warning: parameter bounds are set but the selected solver "
                 "does not enforce them (use ipopt or de)\n";
  }

  auto residual_fn = [&functor](const Eigen::VectorXd &params) {
    Eigen::VectorXd fvec(functor.values());
    functor(params, fvec);
    return fvec;
  };

  auto jacobian_fn = [&functor](const Eigen::VectorXd &params,
                                Eigen::MatrixXd &fjac) {
    functor.df(params, fjac);
  };

  const int status =
      solver.minimize(x, std::move(residual_fn), std::move(jacobian_fn),
                      functor.values(), lower, upper);

  std::visit([&](auto &m) { m.scatter_params(x, std::size_t{0}); }, model);
  return status;
}

} // namespace

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts) {
  return run_with_solver(configs, model, opts, make_default_solver());
}

int run_optimizer(std::span<Configuration> configs, ForceCalculator &model,
                  const OptimizerOptions &opts, const Solver &solver) {
  return run_with_solver(configs, model, opts, solver);
}

} // namespace potfit
