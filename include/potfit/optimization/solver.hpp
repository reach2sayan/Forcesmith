#pragma once

#include "potfit/core/erased.hpp"
#include <Eigen/Core>
#include <functional>
#include <memory>

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

// Factory for the default solver (used by run_optimizer when none is supplied).
Solver make_default_solver(int max_iter = 500, double xtol = 1e-7,
                           double ftol = 1e-7);

} // namespace potfit
