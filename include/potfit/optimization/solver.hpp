#pragma once

#include "potfit/core/erased.hpp"
#include <Eigen/Core>
#include <functional>
#include <memory>
#include <vector>

namespace potfit {

template <typename T>
concept SolverImpl =
    requires(const T &s, Eigen::VectorXd &x,
             std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
             int n_vals) {
        { s.minimize(x, f, n_vals) } -> std::convertible_to<int>;
    };

namespace detail {
struct SolverConcept {
    virtual ~SolverConcept() = default;
    virtual int minimize(Eigen::VectorXd &,
                         std::function<Eigen::VectorXd(const Eigen::VectorXd &)>,
                         int) const = 0;
};
} // namespace detail

class Solver : private detail::ErasedMoveOnly<detail::SolverConcept> {
    template <typename T> struct Model final : detail::SolverConcept {
        T impl_;
        explicit Model(T t) : impl_(std::move(t)) {}
        int minimize(Eigen::VectorXd &x,
                     std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
                     int n_vals) const override {
            return impl_.minimize(x, std::move(f), n_vals);
        }
    };

    using Base = detail::ErasedMoveOnly<detail::SolverConcept>;

public:
    template <SolverImpl T>
        requires(!std::same_as<std::decay_t<T>, Solver>)
    explicit Solver(T impl)
        : Base(std::make_unique<Model<T>>(std::move(impl))) {}

    Solver(Solver &&) = default;
    Solver &operator=(Solver &&) = default;
    Solver(const Solver &) = delete;
    Solver &operator=(const Solver &) = delete;

    int minimize(Eigen::VectorXd &x,
                 std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
                 int n_vals) const {
        return self_->minimize(x, std::move(f), n_vals);
    }
};

struct EigenLMSolver {
    int max_iter = 500;
    double xtol = 1e-7;
    double ftol = 1e-7;
    int minimize(Eigen::VectorXd &x,
                 std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
                 int n_vals) const;
};

static_assert(SolverImpl<EigenLMSolver>);

// Powell's dogleg via Eigen::HybridNonLinearSolver.
// Minimises ||F||² by finding zeros of g(x)[j] = Fᵀ ∂F/∂xⱼ (central FD).
struct EigenHybridSolver {
    int    max_iter = 500;
    double xtol     = 1e-7;
    int minimize(Eigen::VectorXd &x,
                 std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
                 int n_vals) const;
};

static_assert(SolverImpl<EigenHybridSolver>);

// Differential evolution via boost::math::optimization::differential_evolution.
// mutation_factor (F) must be in (0, 1); values ≥ 1.0 throw std::domain_error.
struct BoostDESolver {
    std::vector<double> lower_bounds;         // per-param; empty → auto from current x
    std::vector<double> upper_bounds;
    double      mutation_factor       = 0.65; // F ∈ (0, 1)
    double      crossover_probability = 0.5;
    std::size_t NP_factor             = 15;   // NP = NP_factor × D
    std::size_t max_generations       = 1000;
    unsigned    threads               = 0;    // 0 → hardware_concurrency
    unsigned    seed                  = 0;    // 0 → std::random_device
    int minimize(Eigen::VectorXd &x,
                 std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
                 int n_vals) const;
};

static_assert(SolverImpl<BoostDESolver>);

// Factory for the default solver (used by run_optimizer when none is supplied).
Solver make_default_solver(int max_iter = 500, double xtol = 1e-7,
                           double ftol = 1e-7);

} // namespace potfit
