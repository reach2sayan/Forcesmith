#pragma once

#include "forcesmith/core/erased.hpp"

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace forcesmith {

using ResidualFn = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
using JacobianFn =
    std::function<void(const Eigen::VectorXd &, Eigen::MatrixXd &)>;

template <typename T>
concept CSolver =
    requires(const T &s, Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
             int n_vals, const Eigen::VectorXd &bounds) {
      {
        s.minimize(x, f, jac, n_vals, bounds, bounds)
      } -> std::convertible_to<int>;
    };

namespace detail {
struct SolverConcept {
  virtual ~SolverConcept() = default;
  virtual int minimize(Eigen::VectorXd &, ResidualFn, JacobianFn, int,
                       const Eigen::VectorXd &,
                       const Eigen::VectorXd &) const = 0;
  virtual bool honors_bounds() const = 0;
  virtual bool one_shot() const = 0;
};
} // namespace detail

class Solver : private detail::ErasedMoveOnly<detail::SolverConcept> {
  template <CSolver T> struct Model final : detail::SolverConcept {
    T impl_;
    constexpr explicit Model(T t) : impl_(std::move(t)) {}
    int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
                 const Eigen::VectorXd &lower,
                 const Eigen::VectorXd &upper) const override {
      return impl_.minimize(x, std::move(f), std::move(jac), n_vals, lower,
                            upper);
    }
    // Optional on the concrete solver: honored by Ipopt/DE, absent on the
    // gradient solvers (which ignore the box). Probe and default to false.
    bool honors_bounds() const override {
      if constexpr (requires(const T &t) { t.honors_bounds(); }) {
        return impl_.honors_bounds();
      } else {
        return false;
      }
    }
    // Optional: true for solvers that evaluate the residual + Jacobian once and
    // solve in closed form (NormalEquationsSolver). The optimizer uses this to
    // let the functor stream the Jacobian and skip the persistent descriptor
    // cache. Probe and default to false (iterative solvers reuse the cache).
    bool one_shot() const override {
      if constexpr (requires(const T &t) { t.one_shot(); }) {
        return impl_.one_shot();
      } else {
        return false;
      }
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
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const {
    return self_->minimize(x, std::move(f), std::move(jac), n_vals, lower,
                           upper);
  }
  // True iff the wrapped solver applies the box constraints (Ipopt / DE).
  bool honors_bounds() const { return self_->honors_bounds(); }
  // True iff the solver needs only a single residual+Jacobian evaluation
  // (closed-form least squares) — lets the optimizer stream the Jacobian.
  bool one_shot() const { return self_->one_shot(); }
};

struct EigenLMSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
};

static_assert(CSolver<EigenLMSolver>);

// Powell's dogleg via Eigen::HybridNonLinearSolver.
// Minimizes ||F||² by finding zeros of g(x)[j] = Fᵀ ∂F/∂xⱼ (central FD).
struct EigenHybridSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
};

static_assert(CSolver<EigenHybridSolver>);

// Differential evolution via boost::math::optimization::differential_evolution.
// mutation_factor (F) must be in (0, 1); values ≥ 1.0 throw std::domain_error.
// Honours the box constraints passed to minimize(); when a bound is ±∞ it
// auto-derives a ±half-width box from the current x
struct BoostDESolver {
  double mutation_factor = 0.65; // F ∈ (0, 1)
  double crossover_probability = 0.5;
  std::size_t NP_factor = 15; // NP = NP_factor × D
  std::size_t max_generations = 1000;
  // Population evaluation is intentionally serial
  unsigned seed = 42;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  bool honors_bounds() const { return true; }
};

static_assert(CSolver<BoostDESolver>);

// Powell's direction-set method over 0.5·‖F‖², applies Powell's
// conjugate-direction replacement.
struct LineSearchSolver {
  int max_iter = 200;
  double xtol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;

private:
  // Powell's direction-set minimiser over φ(x) = ½‖F(x)‖², built around linmin.
  // A small stateful helper so minimize() reads as a sequence of named stages:
  // objective → directional line search → full sweep → conjugate-direction
  // update. Stops on ‖Δx‖ < xtol or max_iter sweeps.
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

    // One sweep: line-minimize along every direction in turn.
    SweepResult sweep(Eigen::VectorXd &x,
                      const std::vector<Eigen::VectorXd> &dirs) const;

    // Powell's test for adopting the net move p0→x as a new direction: the
    // extrapolated point 2x−p0 must improve on the sweep start and pass the
    // curvature test. Returns the direction to adopt, or nullopt to keep the
    // set.
    std::optional<Eigen::VectorXd>
    conjugate_direction(const Eigen::VectorXd &p0, const Eigen::VectorXd &x,
                        double fp, double fret, double del) const;
  };
};

static_assert(CSolver<LineSearchSolver>);

// Closed-form (weighted) linear least-squares solver. For a model that is
// LINEAR in its parameters — every ML head a LinearHead — the residual is
// r(θ) = r(θ₀) + J·(θ−θ₀) with a CONSTANT Jacobian J, so the minimiser of
// ‖r‖² is the single Cholesky solve of the normal equations
//   (JᵀJ + λI)·Δ = −Jᵀr(θ₀),   θ = θ₀ + Δ.
// It evaluates f and jac exactly once each (hence one_shot()), so the optimizer
// streams the Jacobian per-config and never materialises the whole-dataset
// descriptor cache — the memory win for large SOAP/ACSF fits. A small ridge λ
// (Tikhonov) conditions the normal matrix and tames collinear/dead features;
// λ = 0 requests an auto tiny value derived from the diagonal.
//
// Intended for linear ML heads; on a nonlinear model it degrades to a single
// Gauss–Newton step (well-defined, but not a full fit).
struct NormalEquationsSolver {
  double ridge = 0.0; // Tikhonov λ; 0 → auto (1e-8 · mean|diag|)
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  bool one_shot() const { return true; }
};

static_assert(CSolver<NormalEquationsSolver>);

// Factory for the default solver (used by run_optimizer when none is supplied).
Solver make_default_solver(int max_iter = 500, double xtol = 1e-7,
                           double ftol = 1e-7);

} // namespace forcesmith
