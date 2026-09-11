#pragma once

#include "forcesmith/optimization/solver.hpp"
#include <Eigen/Core>

namespace forcesmith {

struct IpoptSolver {
  int max_iter = 500;
  double tol = 1e-7;            // Ipopt "tol" (optimality tolerance)
  double acceptable_tol = 1e-6; // Ipopt "acceptable_tol"
  bool silent = true;           // print_level 0 when true

  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  constexpr bool honors_bounds() const { return true; }
  constexpr bool one_shot() const { return false; }
};

static_assert(CSolver<IpoptSolver>);

} // namespace forcesmith
