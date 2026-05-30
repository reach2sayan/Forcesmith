#include "potfit/optimization/solver.hpp"

#include <unsupported/Eigen/NonLinearOptimization>

namespace potfit {

namespace {

// Adapts std::function<VectorXd(VectorXd)> to the Eigen DenseFunctor interface.
struct FunctorAdapter {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> fn;
  int n_inputs;
  int n_values;

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    fvec = fn(x);
    return 0;
  }

  // Central finite-difference Jacobian (Δ = 1e-5).
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
    constexpr double delta = 1e-5;
    Eigen::VectorXd fp, fm, xp = x;
    for (int j = 0; j < n_inputs; ++j) {
      xp[j] += delta;
      fp = fn(xp);
      xp[j] -= 2.0 * delta;
      fm = fn(xp);
      xp[j] += delta;
      fjac.col(j) = (fp - fm) / (2.0 * delta);
    }
    return 0;
  }

  int inputs() const { return n_inputs; }
  int values() const { return n_values; }
};

} // namespace

int EigenLMSolver::minimize(
    Eigen::VectorXd &x,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
    int n_vals) const {
  FunctorAdapter adapter{std::move(f), static_cast<int>(x.size()), n_vals};
  Eigen::LevenbergMarquardt<FunctorAdapter> lm(adapter);
  lm.parameters.maxfev = max_iter;
  lm.parameters.xtol = xtol;
  lm.parameters.ftol = ftol;
  return static_cast<int>(lm.minimize(x));
}

} // namespace potfit
