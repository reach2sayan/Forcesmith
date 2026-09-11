#pragma once

#include <Eigen/Core>
#include <functional>
#include <utility>

namespace forcesmith {

struct LinSearchOptions {
  double initial_step = 1.0;
  double max_step = 1e6;
  double tol = 1e-7;
  int max_iter = 200;
};

namespace detail {
struct linmin_fn {
  double
  operator()(Eigen::VectorXd &x, const Eigen::VectorXd &dir,
             const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &F,
             const LinSearchOptions &opts = {}) const;

private:
  static std::pair<double, double>
  bracket_minimum(const std::function<double(double)> &g, double initial_step,
                  double max_step, int max_iter);
};
} // namespace detail

inline constexpr detail::linmin_fn linmin{};

} // namespace forcesmith
