#pragma once

#include "forcesmith/core/param.hpp"

#include <Eigen/Core>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

namespace forcesmith {

template <class R>
concept ParamRange =
    std::ranges::input_range<R> &&
    std::same_as<std::remove_cvref_t<std::ranges::range_reference_t<R>>, Param>;

template <class T>
concept CFittable =
    requires(T &t, const T &ct, Eigen::VectorXd &v, std::size_t off) {
      { ct.param_count() } -> std::convertible_to<std::size_t>;
      ct.gather_params(v, off);
      t.scatter_params(std::as_const(v), off);
      ct.gather_bounds(v, v, off);
    };

template <class Derived> struct NoBounds {
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    const auto n = static_cast<Eigen::Index>(
        static_cast<const Derived &>(*this).param_count());
    const auto o = static_cast<Eigen::Index>(off);
    lo.segment(o, n).setConstant(-std::numeric_limits<double>::infinity());
    hi.segment(o, n).setConstant(std::numeric_limits<double>::infinity());
  }
};

namespace detail {

constexpr std::size_t count_params_impl(ParamRange auto &&params) {
  return std::ranges::count_if(params, [](const Param &p) { return !p.fixed; });
}

constexpr void gather_params_impl(ParamRange auto &&params,
                                  Eigen::VectorXd &dst, std::size_t off) {
  for (const Param &p : params) {
    if (!p.fixed) {
      dst[off++] = p.value;
    }
  }
}

constexpr void scatter_params_impl(ParamRange auto &&params,
                                   const Eigen::VectorXd &src,
                                   std::size_t off) {
  for (Param &p : params) {
    if (!p.fixed) {
      p.value = src[off++];
    }
  }
}

constexpr void gather_bounds_impl(ParamRange auto &&params, Eigen::VectorXd &lo,
                                  Eigen::VectorXd &hi, std::size_t off) {
  for (const Param &p : params) {
    if (!p.fixed) {
      lo[off] = p.min;
      hi[off] = p.max;
      ++off;
    }
  }
}

void gather_params_impl(const std::ranges::input_range auto &values,
                        const std::ranges::input_range auto &fixed,
                        Eigen::VectorXd &dst, std::size_t off) {
  for (auto &&[v, f] : std::views::zip(values, fixed)) {
    if (!f) {
      dst[off++] = v;
    }
  }
}

void scatter_params_impl(std::ranges::input_range auto &values,
                         const std::ranges::input_range auto &fixed,
                         const Eigen::VectorXd &src, std::size_t off) {
  for (auto &&[v, f] : std::views::zip(values, fixed)) {
    if (!f) {
      v = src[off++];
    }
  }
}

struct FittableConcept {
  virtual ~FittableConcept() = default;
  virtual std::size_t param_count() const = 0;
  virtual void gather_params(Eigen::VectorXd &x, std::size_t off) const = 0;
  virtual void scatter_params(const Eigen::VectorXd &x, std::size_t off) = 0;
  virtual void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                             std::size_t off) const = 0;
};

template <class T, class Concept> struct FittableModel : Concept {
  T impl_;
  constexpr explicit FittableModel(T t) : impl_(std::move(t)) {}
  constexpr std::size_t param_count() const override {
    return impl_.param_count();
  }
  constexpr void gather_params(Eigen::VectorXd &x,
                               std::size_t off) const override {
    impl_.gather_params(x, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &x,
                                std::size_t off) override {
    impl_.scatter_params(x, off);
  }
  constexpr void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                               std::size_t off) const override {
    impl_.gather_bounds(lo, hi, off);
  }
};

template <class Derived, class T, class Concept>
struct CloningModel : FittableModel<T, Concept> {
  using FittableModel<T,
                      Concept>::FittableModel; // inherit impl_-forwarding ctor
  std::unique_ptr<Concept> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

} // namespace detail

template <class Derived> struct ParamSet {
  constexpr std::size_t param_count() const {
    return detail::count_params_impl(self().param_fields());
  }
  constexpr void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    detail::gather_params_impl(self().param_fields(), dst, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    detail::scatter_params_impl(self().param_fields(), src, off);
  }
  constexpr void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                               std::size_t off) const {
    detail::gather_bounds_impl(self().param_fields(), lo, hi, off);
  }

private:
  constexpr const Derived &self() const {
    return static_cast<const Derived &>(*this);
  }
  constexpr Derived &self() { return static_cast<Derived &>(*this); }
};

namespace detail {
struct ParamSetFittableProbe : ParamSet<ParamSetFittableProbe> {
  std::array<Param, 1> fields_{};
  consteval std::array<Param, 1> &param_fields() { return fields_; }
  consteval const std::array<Param, 1> &param_fields() const { return fields_; }
};
static_assert(CFittable<ParamSetFittableProbe>,
              "ParamSet must structurally model the FittableConcept interface");
} // namespace detail

} // namespace forcesmith
