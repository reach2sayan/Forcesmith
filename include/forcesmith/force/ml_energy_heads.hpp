#pragma once

#include "forcesmith/core/erased.hpp"
#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/param.hpp"
#include "forcesmith/core/types.hpp"

#include <Eigen/Core>
#include <nlohmann/json.hpp> // nlohmann::json (constructed/returned by value in inline to_json bodies)

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace forcesmith {

template <class T> nlohmann::json build_json_from_head(const T &) = delete;

namespace detail {
struct HeadConcept : FittableConcept {
  virtual double energy(const Eigen::VectorXd &D) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd &D) const = 0; // de/dD
  virtual bool constant_grad() const = 0; // ∂E/∂D and ∂(∂E/∂D)/∂θ const per

  virtual bool has_param_jacobian() const = 0;
  virtual Eigen::VectorXd
  param_grad(const Eigen::VectorXd &D) const = 0; // ∂E/∂θ  (1 x num_params)
  virtual Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D)
      const = 0; // ∂(∂E/∂D)/∂θ  (num_descritptor x num_params)

  virtual nlohmann::json to_json() const = 0;
  virtual Eigen::VectorXd all_values() const = 0;
  virtual void set_all_values(const Eigen::VectorXd &v) = 0;

  virtual std::unique_ptr<HeadConcept> clone() const = 0;
  virtual std::unique_ptr<HeadConcept>
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const = 0;
};
} // namespace detail

template <class T>
concept CHead = CFittable<T> &&
                requires(const T &t, T &mut, const Eigen::VectorXd &D,
                         const Eigen::VectorXd &v,
                         const std::vector<std::optional<Eigen::Index>> &map) {
                  { t.energy(D) } -> std::convertible_to<double>;
                  { t.grad(D) } -> std::convertible_to<Eigen::VectorXd>;
                  { t.constant_grad() } -> std::convertible_to<bool>;
                  { t.has_param_jacobian() } -> std::convertible_to<bool>;
                  { t.param_grad(D) } -> std::convertible_to<Eigen::VectorXd>;
                  { t.dgrad_dparam(D) } -> std::convertible_to<Eigen::MatrixXd>;
                  { t.all_values() } -> std::convertible_to<Eigen::VectorXd>;
                  mut.set_all_values(v);
                  { t.remapped(map) } -> std::convertible_to<T>;
                };

template <typename Derived> struct HeadParams : ParamSet<HeadParams<Derived>> {
  auto param_fields();
  auto param_fields() const;

  Eigen::VectorXd all_values() const;
  void set_all_values(const Eigen::VectorXd &v);

private:
  constexpr const Derived &self() const;
  constexpr Derived &self();
};

template <typename Derived> auto HeadParams<Derived>::param_fields() {
  return self().param_range();
}

template <typename Derived> auto HeadParams<Derived>::param_fields() const {
  return self().param_range();
}

template <typename Derived>
Eigen::VectorXd HeadParams<Derived>::all_values() const {
  auto f = self().param_range();
  Eigen::VectorXd v(static_cast<Eigen::Index>(std::ranges::size(f)));
  std::ranges::transform(f, v.data(),
                         [](const Param &p) { return p.value; });
  return v;
}

template <typename Derived>
void HeadParams<Derived>::set_all_values(const Eigen::VectorXd &v) {
  auto f = self().param_range();
  assert(v.size() == static_cast<Eigen::Index>(std::ranges::size(f)));
  for (auto &&[field, value] : std::views::zip(
           f, std::span(v.data(), static_cast<std::size_t>(v.size())))) {
    field.value = value;
  }
}

template <typename Derived>
constexpr const Derived &HeadParams<Derived>::self() const {
  return static_cast<const Derived &>(*this);
}

template <typename Derived> constexpr Derived &HeadParams<Derived>::self() {
  return static_cast<Derived &>(*this);
}

class EnergyHead : private detail::ErasedValue<detail::HeadConcept> {
  template <CHead T>
  struct Model final : detail::CloningModel<Model<T>, T, detail::HeadConcept> {
    using Base = detail::CloningModel<Model<T>, T, detail::HeadConcept>;
    using Base::Base;  // inherit the impl_-forwarding constructor
    using Base::impl_; // bring impl_ into scope for the bodies below
    double energy(const Eigen::VectorXd &D) const override {
      return impl_.energy(D);
    }
    Eigen::VectorXd grad(const Eigen::VectorXd &D) const override {
      return impl_.grad(D);
    }
    bool constant_grad() const override { return impl_.constant_grad(); }
    bool has_param_jacobian() const override {
      return impl_.has_param_jacobian();
    }
    Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const override {
      return impl_.param_grad(D);
    }
    Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const override {
      return impl_.dgrad_dparam(D);
    }
    nlohmann::json to_json() const override {
      return build_json_from_head(impl_); // ADL → concrete head's overload
    }
    Eigen::VectorXd all_values() const override { return impl_.all_values(); }
    void set_all_values(const Eigen::VectorXd &v) override {
      impl_.set_all_values(v);
    }
    std::unique_ptr<detail::HeadConcept> remapped(
        const std::vector<std::optional<Eigen::Index>> &map) const override {
      return std::make_unique<Model>(impl_.remapped(map));
    }
  };

  using Base = detail::ErasedValue<detail::HeadConcept>;

  explicit EnergyHead(std::unique_ptr<detail::HeadConcept> p)
      : Base(std::move(p)) {}

public:
  template <typename T>
    requires(!std::same_as<std::decay_t<T>, EnergyHead> &&
             CHead<std::decay_t<T>>)
  explicit EnergyHead(T &&t)
      : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

  EnergyHead(const EnergyHead &) = default;
  EnergyHead(EnergyHead &&) noexcept = default;
  EnergyHead &operator=(const EnergyHead &) = default;
  EnergyHead &operator=(EnergyHead &&) noexcept = default;

  FORCE_INLINE double energy(const Eigen::VectorXd &D) const {
    return self_->energy(D);
  }
  FORCE_INLINE Eigen::VectorXd grad(const Eigen::VectorXd &D) const {
    return self_->grad(D);
  }
  FORCE_INLINE bool constant_grad() const { return self_->constant_grad(); }
  FORCE_INLINE bool has_param_jacobian() const {
    return self_->has_param_jacobian();
  }
  FORCE_INLINE Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const {
    return self_->param_grad(D);
  }
  FORCE_INLINE Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const {
    return self_->dgrad_dparam(D);
  }
  FORCE_INLINE std::size_t param_count() const { return self_->param_count(); }
  FORCE_INLINE void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  FORCE_INLINE void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
  FORCE_INLINE void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                                  std::size_t off) const {
    self_->gather_bounds(lo, hi, off);
  }
  FORCE_INLINE nlohmann::json to_json() const { return self_->to_json(); }
  FORCE_INLINE Eigen::VectorXd all_values() const {
    return self_->all_values();
  }
  FORCE_INLINE void set_all_values(const Eigen::VectorXd &v) {
    self_->set_all_values(v);
  }
  [[nodiscard]] FORCE_INLINE EnergyHead
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const {
    return EnergyHead(self_->remapped(map));
  }
};

} // namespace forcesmith
