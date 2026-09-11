#include "forcesmith/optimization/optimizer.hpp"
#include "forcesmith/optimization/forcesmith_functor.hpp"

#include <iostream>

namespace forcesmith {

namespace {

int run_with_solver(std::span<Configuration> configs, ForceCalculator &model,
                    const OptimizerOptions &opts, const Solver &solver) {
  const bool stream_jacobian = solver.one_shot() &&
                               model.has_param_jacobian() &&
                               !model.has_standardization();

  ForcesmithFunctor functor(configs, model, opts.energy_weight,
                            opts.stress_weight, opts.smooth_weight,
                            stream_jacobian);

  Eigen::VectorXd x(functor.inputs());
  model.gather_params(x, std::size_t{0});

  Eigen::VectorXd lower(functor.inputs()), upper(functor.inputs());
  model.gather_bounds(lower, upper, std::size_t{0});

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

  model.scatter_params(x, std::size_t{0});
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

} // namespace forcesmith
