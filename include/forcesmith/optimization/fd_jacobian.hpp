#pragma once

// Central-difference Jacobian and gradient; arithmetic unchanged, bit-for-bit.

#include <Eigen/Core>

#include <concepts>
#include <utility>

namespace forcesmith::opt {

inline constexpr double kFDDelta = 1e-5;

namespace detail {
template <class F>
concept CResidualInto =
    std::invocable<F, const Eigen::VectorXd &, Eigen::VectorXd &>;

template <class F>
concept CResidual =
    CResidualInto<F> || std::invocable<F, const Eigen::VectorXd &>;

template <CResidual F>
void eval_residual(F &f, const Eigen::VectorXd &x, Eigen::VectorXd &out) {
  if constexpr (CResidualInto<F>) {
    f(x, out);
  } else {
    out = f(x);
  }
}
} // namespace detail

// J.col(j) = [f(x + h e_j) - f(x - h e_j)] / 2h. J must already be sized
// values x inputs; x is restored exactly (the same add/subtract sequence the
// call sites used, so the probe points are bit-identical).
template <detail::CResidual F>
void fd_jacobian(F &&f, const Eigen::VectorXd &x, Eigen::MatrixXd &J,
                 double h = kFDDelta) {
  Eigen::VectorXd fp, fm, xp = x;
  for (Eigen::Index j = 0; j < x.size(); ++j) {
    xp[j] += h;
    detail::eval_residual(f, xp, fp);
    xp[j] -= 2.0 * h;
    detail::eval_residual(f, xp, fm);
    xp[j] += h;
    J.col(j) = (fp - fm) / (2.0 * h);
  }
}

template <detail::CResidual F>
void fd_gradient(F &&f, const Eigen::VectorXd &x, Eigen::VectorXd &grad,
                 double h = kFDDelta) {
  Eigen::VectorXd f0, fp, fm, xp = x;
  detail::eval_residual(f, x, f0);
  grad.resize(x.size());
  for (Eigen::Index j = 0; j < x.size(); ++j) {
    xp[j] += h;
    detail::eval_residual(f, xp, fp);
    xp[j] -= 2.0 * h;
    detail::eval_residual(f, xp, fm);
    xp[j] += h;
    grad[j] = f0.dot((fp - fm) / (2.0 * h));
  }
}

} // namespace forcesmith::opt
