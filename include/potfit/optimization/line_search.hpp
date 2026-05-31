#pragma once

#include <Eigen/Core>
#include <functional>

namespace potfit {

struct LinSearchOptions {
  double initial_step = 1.0;
  double max_step = 1e6;
  double tol = 1e-7;
  int max_iter = 200;
};

// Minimises 0.5*||F(x + α·dir)||² over α ≥ 0 via golden-ratio bracketing +
// boost::math::tools::brent_find_minima. Updates x in place (x += α*·dir).
// Returns the optimal α found.
double linmin(Eigen::VectorXd &x, const Eigen::VectorXd &dir,
              const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &F,
              const LinSearchOptions &opts = {});

} // namespace potfit
