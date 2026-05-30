#pragma once

#include "potfit/core/erased.hpp"
#include <Eigen/Core>
#include <concepts>
#include <memory>
#include <utility>

namespace potfit {

// Sean Parent value-type erasure.
// Any type satisfying the following concept can be stored in a Potential:
//   eval(double)->double, deriv(double)->double, span()->pair<double,double>
//   param_count()->int, gather_params(VectorXd&,int), scatter_params(const VectorXd&,int)

namespace detail {
struct PotentialConcept {
    virtual ~PotentialConcept() = default;
    virtual double eval(double r) const = 0;
    virtual double deriv(double r) const = 0;
    virtual std::pair<double, double> span() const = 0;
    virtual int param_count() const = 0;
    virtual void gather_params(Eigen::VectorXd &x, int off) const = 0;
    virtual void scatter_params(const Eigen::VectorXd &x, int off) = 0;
    virtual std::unique_ptr<PotentialConcept> clone() const = 0;
};
} // namespace detail

class Potential : private detail::ErasedValue<detail::PotentialConcept> {
    template <typename T> struct Model final : detail::PotentialConcept {
        T impl_;
        constexpr explicit Model(T t) : impl_(std::move(t)) {}
        constexpr double eval(double r) const override { return impl_.eval(r); }
        constexpr double deriv(double r) const override { return impl_.deriv(r); }
        constexpr std::pair<double, double> span() const override { return impl_.span(); }
        constexpr int param_count() const override { return impl_.param_count(); }
        constexpr void gather_params(Eigen::VectorXd &x, int off) const override {
            impl_.gather_params(x, off);
        }
        constexpr void scatter_params(const Eigen::VectorXd &x, int off) override {
            impl_.scatter_params(x, off);
        }
        std::unique_ptr<detail::PotentialConcept> clone() const override {
            return std::make_unique<Model>(*this);
        }
    };

    using Base = detail::ErasedValue<detail::PotentialConcept>;

public:
    template <typename T>
        requires(!std::same_as<std::decay_t<T>, Potential>)
    constexpr explicit Potential(T &&t)
        : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

    Potential(const Potential &) = default;
    Potential(Potential &&) noexcept = default;
    Potential &operator=(const Potential &) = default;
    Potential &operator=(Potential &&) noexcept = default;

    constexpr double eval(double r) const { return self_->eval(r); }
    constexpr double deriv(double r) const { return self_->deriv(r); }
    constexpr std::pair<double, double> span() const { return self_->span(); }
    constexpr int param_count() const { return self_->param_count(); }
    constexpr void gather_params(Eigen::VectorXd &x, int off) const {
        self_->gather_params(x, off);
    }
    constexpr void scatter_params(const Eigen::VectorXd &x, int off) {
        self_->scatter_params(x, off);
    }
};

} // namespace potfit
