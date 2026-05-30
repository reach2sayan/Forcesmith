#pragma once

#include <Eigen/Core>
#include <functional>
#include <memory>

namespace potfit {

// ── Concept
// ───────────────────────────────────────────────────────────────────
//
// A SolverImpl minimizes ||f(x)||² by modifying x in-place.
//   x      — initial parameter vector, overwritten with the minimizer
//   f      — residual function: maps parameter vector → residual vector
//   n_vals — dimension of the residual vector (f's output size)
//   returns an integer status code (≥1 = converged, ≤0 = failed/limit)
//
// Future impls (line-search, Brent, Powell, SA, DE) must satisfy this concept.

template <typename T>
concept SolverImpl =
    requires(const T &s, Eigen::VectorXd &x,
             std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
             int n_vals) {
      { s.minimize(x, f, n_vals) } -> std::convertible_to<int>;
    };

// ── Value-type-erased Solver
// ──────────────────────────────────────────────────

class Solver {
public:
  template <SolverImpl T>
    requires(!std::same_as<std::decay_t<T>, Solver>)
  explicit Solver(T impl)
      : impl_(std::make_unique<Model<T>>(std::move(impl))) {}

  Solver(Solver &&) = default;
  Solver &operator=(Solver &&) = default;
  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;

  int minimize(Eigen::VectorXd &x,
               std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
               int n_vals) const {
    return impl_->minimize(x, std::move(f), n_vals);
  }

private:
  struct Concept {
    virtual int
    minimize(Eigen::VectorXd &,
             std::function<Eigen::VectorXd(const Eigen::VectorXd &)>,
             int) const = 0;
    virtual ~Concept() = default;
  };

  template <typename T> struct Model final : Concept {
    T impl_;
    explicit Model(T t) : impl_(std::move(t)) {}
    int minimize(Eigen::VectorXd &x,
                 std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
                 int n_vals) const override {
      return impl_.minimize(x, std::move(f), n_vals);
    }
  };

  std::unique_ptr<Concept> impl_;
};

// ── Concrete: Eigen Levenberg-Marquardt
// ───────────────────────────────────────

struct EigenLMSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;

  // Wraps f in an Eigen dense-functor adapter and runs LM with central-FD
  // Jacobian. Returns Eigen::LevenbergMarquardtSpace::Status cast to int.
  int minimize(Eigen::VectorXd &x,
               std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
               int n_vals) const;
};

static_assert(SolverImpl<EigenLMSolver>);

// Factory for the default solver (used by run_optimizer when none is supplied).
inline Solver make_default_solver(int max_iter = 500, double xtol = 1e-7,
                                  double ftol = 1e-7) {
  return Solver(EigenLMSolver{max_iter, xtol, ftol});
}

} // namespace potfit
