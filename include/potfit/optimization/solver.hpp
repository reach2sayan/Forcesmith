#pragma once

#include "potfit/core/erased.hpp"

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace potfit {

// Residual map x ↦ F(x) and the (optional) Jacobian map (x, J) ↦ ∂F/∂x. An
// empty JacobianFn means "no analytic/parallel Jacobian supplied" — solvers
// that need a Jacobian fall back to their own finite differences.
using ResidualFn = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
using JacobianFn =
    std::function<void(const Eigen::VectorXd &, Eigen::MatrixXd &)>;

template <typename T>
concept CSolver = requires(const T &s, Eigen::VectorXd &x, ResidualFn f,
                              JacobianFn jac, int n_vals) {
  { s.minimize(x, f, jac, n_vals) } -> std::convertible_to<int>;
};

namespace detail {
struct SolverConcept {
  virtual ~SolverConcept() = default;
  virtual int minimize(Eigen::VectorXd &, ResidualFn, JacobianFn,
                       int) const = 0;
};
} // namespace detail

class Solver : private detail::ErasedMoveOnly<detail::SolverConcept> {
  template <typename T> struct Model final : detail::SolverConcept {
    T impl_;
    constexpr explicit Model(T t) : impl_(std::move(t)) {}
    int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
                 int n_vals) const override {
      return impl_.minimize(x, std::move(f), std::move(jac), n_vals);
    }
  };
  using Base = detail::ErasedMoveOnly<detail::SolverConcept>;
public:
  template <CSolver T>
    requires(!std::same_as<std::decay_t<T>, Solver>)
  constexpr explicit Solver(T impl)
      : Base(std::make_unique<Model<T>>(std::move(impl))) {}

  constexpr Solver(Solver &&) = default;
  Solver &operator=(Solver &&) = default;
  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
               int n_vals) const {
    return self_->minimize(x, std::move(f), std::move(jac), n_vals);
  }
};

struct EigenLMSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
               int n_vals) const;
};

static_assert(CSolver<EigenLMSolver>);

// Powell's dogleg via Eigen::HybridNonLinearSolver.
// Minimizes ||F||² by finding zeros of g(x)[j] = Fᵀ ∂F/∂xⱼ (central FD).
struct EigenHybridSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
               int n_vals) const;
};

static_assert(CSolver<EigenHybridSolver>);

// Differential evolution via boost::math::optimization::differential_evolution.
// mutation_factor (F) must be in (0, 1); values ≥ 1.0 throw std::domain_error.
struct BoostDESolver {
  std::vector<double> lower_bounds; // per-param; empty → auto from current x
  std::vector<double> upper_bounds;
  double mutation_factor = 0.65; // F ∈ (0, 1)
  double crossover_probability = 0.5;
  std::size_t NP_factor = 15; // NP = NP_factor × D
  std::size_t max_generations = 1000;
  // Population evaluation is intentionally serial — see
  // BoostDESolver::minimize. Parallelism comes one level down, from the
  // functor's per-config TBB loop.
  unsigned seed = 42;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
               int n_vals) const;
};

static_assert(CSolver<BoostDESolver>);

// Powell's direction-set method over 0.5·‖F‖², driven by linmin (golden-section
// bracketing + Brent). Derivative-free; each sweep line-minimizes along every
// direction (both ±, since linmin only searches α ≥ 0) and applies Powell's
// conjugate-direction replacement. Stops on ‖Δx‖ < xtol or max_iter sweeps.
struct LineSearchSolver {
  int max_iter = 200;
  double xtol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
               int n_vals) const;

private:
  // Powell's direction-set minimiser over φ(x) = ½‖F(x)‖², built around linmin.
  // A small stateful helper so minimize() reads as a sequence of named stages:
  // objective → directional line search → full sweep → conjugate-direction
  // update.
  struct PowellDirectionSet {
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &f;
    double phi(const Eigen::VectorXd &v) const {
      return 0.5 * f(v).squaredNorm();
    }
    // Line-minimise x along ±dir (linmin only searches α ≥ 0), never raising φ.
    void line_min(Eigen::VectorXd &x, const Eigen::VectorXd &dir) const;

    struct SweepResult {
      int ibig = 0;     // direction that gave the largest decrease
      double del = 0.0; // magnitude of that decrease
    };

    // One sweep: line-minimise along every direction in turn.
    SweepResult sweep(Eigen::VectorXd &x,
                      const std::vector<Eigen::VectorXd> &dirs) const;

    // Powell's test for adopting the net move p0→x as a new direction: the
    // extrapolated point 2x−p0 must improve on the sweep start and pass the
    // curvature test. Returns the direction to adopt, or nullopt to keep the set.
    std::optional<Eigen::VectorXd>
    conjugate_direction(const Eigen::VectorXd &p0, const Eigen::VectorXd &x,
                        double fp, double fret, double del) const;
  };
};

static_assert(CSolver<LineSearchSolver>);

// Factory for the default solver (used by run_optimizer when none is supplied).
Solver make_default_solver(int max_iter = 500, double xtol = 1e-7,
                           double ftol = 1e-7);

} // namespace potfit
