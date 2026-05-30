#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/core/erased.hpp"
#include <concepts>
#include <cstdint>
#include <memory>

namespace potfit {

// Concept satisfied by any concrete force calculator type.
template <typename T>
concept ForceCalculatorModel = requires(T calc, Configuration &cfg) {
    { calc.eval_forces(cfg) } -> std::same_as<void>;
};

// CRTP base for multi-body force calculators that hold typed potential tables.
// Provides the shared fields ntypes and conf_index.
template <typename Derived> struct ForceCalculatorBase {
    int ntypes = 1;
    std::uint64_t conf_index = 0;
};

namespace detail {
struct ForceCalculatorConcept {
    virtual ~ForceCalculatorConcept() = default;
    virtual void eval_forces(Configuration &cfg) const = 0;
    virtual std::unique_ptr<ForceCalculatorConcept> clone() const = 0;
};
} // namespace detail

class ForceCalculator : private detail::ErasedValue<detail::ForceCalculatorConcept> {
    template <typename T> struct Model final : detail::ForceCalculatorConcept {
        T impl_;
        explicit Model(T t) : impl_(std::move(t)) {}
        constexpr void eval_forces(Configuration &cfg) const override {
            impl_.eval_forces(cfg);
        }
        std::unique_ptr<detail::ForceCalculatorConcept> clone() const override {
            return std::make_unique<Model>(*this);
        }
    };

    using Base = detail::ErasedValue<detail::ForceCalculatorConcept>;

public:
    template <typename T>
        requires(!std::same_as<std::decay_t<T>, ForceCalculator>)
    constexpr explicit ForceCalculator(T &&t)
        : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

    ForceCalculator(const ForceCalculator &) = default;
    ForceCalculator(ForceCalculator &&) noexcept = default;
    ForceCalculator &operator=(const ForceCalculator &) = default;
    ForceCalculator &operator=(ForceCalculator &&) noexcept = default;

    constexpr void eval_forces(Configuration &cfg) const {
        self_->eval_forces(cfg);
    }
};

static_assert(ForceCalculatorModel<ForceCalculator>);

} // namespace potfit
