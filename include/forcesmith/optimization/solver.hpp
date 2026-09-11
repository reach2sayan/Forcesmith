#pragma once

#include "forcesmith/core/erasure.hpp"

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
      { s.honors_bounds() } -> std::convertible_to<bool>;
      { s.one_shot() } -> std::convertible_to<bool>;
    };

namespace te_detail {
struct CSolverTE
    : boost::mpl::vector<
          te::MoveOnlyBuiltins,
          te::has_minimize<int(Eigen::VectorXd &, ResidualFn, JacobianFn, int,
                               const Eigen::VectorXd &, const Eigen::VectorXd &)
                               const>,
          te::has_honors_bounds<bool() const>, te::has_one_shot<bool() const>> {
};
} // namespace te_detail

class Solver {
  boost::type_erasure::any<te_detail::CSolverTE> a_;

public:
  template <class T>
    requires(!std::same_as<std::decay_t<T>, Solver> && CSolver<std::decay_t<T>>)
  explicit Solver(T &&impl) : a_(std::forward<T>(impl)) {}

  Solver(Solver &&) = default;
  Solver &operator=(Solver &&) = default;
  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;

  template <class T> const T *target() const noexcept {
    return boost::type_erasure::any_cast<const T *>(&a_);
  }

  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const {
    return a_.minimize(x, std::move(f), std::move(jac), n_vals, lower, upper);
  }
  bool honors_bounds() const { return a_.honors_bounds(); }
  bool one_shot() const { return a_.one_shot(); }
};

struct EigenLMSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  double ftol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  constexpr bool honors_bounds() const { return false; }
  constexpr bool one_shot() const { return false; }
};

static_assert(CSolver<EigenLMSolver>);

struct EigenHybridSolver {
  int max_iter = 500;
  double xtol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  constexpr bool honors_bounds() const { return false; }
  constexpr bool one_shot() const { return false; }
};

static_assert(CSolver<EigenHybridSolver>);

struct BoostDESolver {
  double mutation_factor = 0.65; // F ∈ (0, 1)
  double crossover_probability = 0.5;
  std::size_t NP_factor = 15; // NP = NP_factor × D
  std::size_t max_generations = 1000;
  unsigned seed = 42;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  bool honors_bounds() const { return true; }
  constexpr bool one_shot() const { return false; }
};

static_assert(CSolver<BoostDESolver>);

struct LineSearchSolver {
  int max_iter = 200;
  double xtol = 1e-7;
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  constexpr bool honors_bounds() const { return false; }
  constexpr bool one_shot() const { return false; }

private:
  struct PowellDirectionSet {
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &f;
    double phi(const Eigen::VectorXd &v) const {
      return 0.5 * f(v).squaredNorm();
    }
    void line_min(Eigen::VectorXd &x, const Eigen::VectorXd &dir) const;

    struct SweepResult {
      int ibig = 0;     // direction that gave the largest decrease
      double del = 0.0; // magnitude of that decrease
    };

    SweepResult sweep(Eigen::VectorXd &x,
                      const std::vector<Eigen::VectorXd> &dirs) const;

    std::optional<Eigen::VectorXd>
    conjugate_direction(const Eigen::VectorXd &p0, const Eigen::VectorXd &x,
                        double fp, double fret, double del) const;
  };
};

static_assert(CSolver<LineSearchSolver>);

struct NormalEquationsSolver {
  double ridge = 0.0; // Tikhonov λ; 0 → auto (1e-8 · mean|diag|)
  int minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
               const Eigen::VectorXd &lower,
               const Eigen::VectorXd &upper) const;
  bool one_shot() const { return true; }
  constexpr bool honors_bounds() const { return false; }
};

static_assert(CSolver<NormalEquationsSolver>);

Solver make_default_solver(int max_iter = 500, double xtol = 1e-7,
                           double ftol = 1e-7);

} // namespace forcesmith
