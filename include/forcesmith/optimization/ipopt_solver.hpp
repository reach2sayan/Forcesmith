#pragma once

#include "forcesmith/optimization/solver.hpp"
#include <Eigen/Core>

namespace forcesmith {

// Ipopt-backed solver for the least-squares objective φ(x) = ½‖F(x)‖².
// Satisfies SolverImpl, so it drops into the same Solver façade as
// EigenLMSolver and friends. The objective and its gradient are handed to Ipopt
// as
//   f(x)  = ½ F(x)ᵀ F(x)
//   ∇f(x) = J(x)ᵀ F(x)
// with no constraints and an L-BFGS (limited-memory) Hessian approximation. The
// Jacobian J comes from the supplied JacobianFn (the optimizer's parallel,
// finite-difference df); if none is supplied, minimize falls back to a local
// central-difference Jacobian.

struct IpoptSolver {
  int max_iter = 500;
  double tol = 1e-7;            // Ipopt "tol" (optimality tolerance)
  double acceptable_tol = 1e-6; // Ipopt "acceptable_tol"
  bool silent = true;           // print_level 0 when true

  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  constexpr bool honors_bounds() const { return true; }
};

static_assert(CSolver<IpoptSolver>);

} // namespace forcesmith
