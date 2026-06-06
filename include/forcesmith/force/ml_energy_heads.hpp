#pragma once

// Type-erased "head" infrastructure for ML potentials.
//
// A head maps a per-atom descriptor vector D_i to a per-atom energy E_i.
// Concrete heads (e.g. LinearHead in ml_force.hpp) implement the math;
// EnergyHead is the value-erased holder (Sean Parent style, like
// RadialPotential) so the ML force calculator can store one head per element
// type without templating on the concrete head type.
//
//   * HeadConcept  — the abstract interface the erasure forwards to (inherits
//                    the optimizer param-plumbing virtuals from
//                    FittableConcept).
//   * HeadParams   — CRTP base concrete heads derive from for default
//                    (de)serialization over their Param fields.
//   * EnergyHead   — the public value type; copyable/assignable, holds any
//   head.
//
// Member bodies are defined out-of-line below each class (kept in the header
// because they are templates or must be inline).

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

// Head JSON customization point (ADL, like curvature_count/write_curvature):
// every concrete head — including ones defined by external clients — provides
// `nlohmann::json build_json_from_head(const TheHead&)` in its own namespace,
// and the erased Model<T>::to_json() calls it unqualified so ADL resolves it.
// No generic default: a head without an overload is a hard compile error (JSON
// has no sensible default shape).
template <class T> nlohmann::json build_json_from_head(const T &) = delete;

namespace detail {
// Inherits the optimizer param-plumbing virtuals (param_count, gather_params,
// scatter_params, gather_bounds) from FittableConcept (core/fit_params.hpp).
struct HeadConcept : FittableConcept {
  virtual double energy(const Eigen::VectorXd &D) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd &D) const = 0; // de/dD
  virtual bool constant_grad() const = 0; // ∂E/∂D and ∂(∂E/∂D)/∂θ const per
                                          // elem

  virtual bool has_param_jacobian() const = 0;
  virtual Eigen::VectorXd
  param_grad(const Eigen::VectorXd &D) const = 0; // ∂E/∂θ  (1 x num_params)
  virtual Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D)
      const = 0; // ∂(∂E/∂D)/∂θ  (num_descritptor x num_params)

  virtual nlohmann::json to_json() const = 0;
  virtual Eigen::VectorXd all_values() const = 0;
  virtual void set_all_values(const Eigen::VectorXd &v) = 0;

  virtual std::unique_ptr<HeadConcept> clone() const = 0;
  // Re-rank support (species count change): remapped() re-lays-out the head to
  // map.size() descriptor features, pulling coeff k from old index map[k]
  // (nullopt ⇒ 0), preserving the bias and per-coeff fixed flags. Carrying over
  // learned coeffs is the only re-rank op that needs the concrete head, so it
  // is the only one on this interface; fresh new-element heads are built
  // in-place by the caller (see MLBase::remap).
  virtual std::unique_ptr<HeadConcept>
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const = 0;
};
} // namespace detail

// The leaf-side contract for a type stored in an EnergyHead: the descriptor→
// energy maths plus the param-Jacobian hooks EnergyHead::Model forwards to
// unconditionally. Subsumes CFittable (every head supplies the full optimizer
// param surface, gather_bounds included, via HeadParams/ParamSet). remapped()
// returns the concrete head itself (Derived). The JSON CPO build_json_from_head
// is intentionally omitted — it is resolved by ADL at the wrap site.
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

  // Default (de)serialization over field_ptrs() — ALL params, free and fixed.
  // Derived must still supply type_tag() and architecture(). Heads that do not
  // store their parameters as Param* may override these.
  Eigen::VectorXd all_values() const;
  void set_all_values(const Eigen::VectorXd &v);

private:
  constexpr const Derived &self() const;
  constexpr Derived &self();
};

template <typename Derived> auto HeadParams<Derived>::param_fields() {
  return self().field_ptrs() |
         std::views::transform([](Param *p) -> Param & { return *p; });
}

template <typename Derived> auto HeadParams<Derived>::param_fields() const {
  return self().field_ptrs() |
         std::views::transform(
             [](const Param *p) -> const Param & { return *p; });
}

template <typename Derived>
Eigen::VectorXd HeadParams<Derived>::all_values() const {
  auto f = self().field_ptrs();
  Eigen::VectorXd v(static_cast<Eigen::Index>(f.size()));
  std::ranges::transform(f, v.data(), [](const auto *p) { return p->value; });
  return v;
}

template <typename Derived>
void HeadParams<Derived>::set_all_values(const Eigen::VectorXd &v) {
  auto f = self().field_ptrs();
  assert(v.size() == static_cast<Eigen::Index>(f.size()));
  for (auto [field, value] : std::views::zip(
           f, std::span(v.data(), static_cast<std::size_t>(v.size())))) {
    field->value = value;
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
    double energy(const Eigen::VectorXd &D) const override;
    Eigen::VectorXd grad(const Eigen::VectorXd &D) const override;
    bool constant_grad() const override;
    bool has_param_jacobian() const override;
    Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const override;
    Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const override;
    nlohmann::json to_json() const override;
    Eigen::VectorXd all_values() const override;
    void set_all_values(const Eigen::VectorXd &v) override;
    // clone() supplied by CloningModel.
    std::unique_ptr<detail::HeadConcept> remapped(
        const std::vector<std::optional<Eigen::Index>> &map) const override;
  };

  using Base = detail::ErasedValue<detail::HeadConcept>;

  // Wrap an already-built concept (used by remapped()).
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

  double energy(const Eigen::VectorXd &D) const;
  Eigen::VectorXd grad(const Eigen::VectorXd &D) const;
  bool constant_grad() const;
  bool has_param_jacobian() const;
  Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const;
  Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const;
  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &x, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &x, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  nlohmann::json to_json() const;
  Eigen::VectorXd all_values() const;
  void set_all_values(const Eigen::VectorXd &v);
  [[nodiscard]] EnergyHead
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const;
};

template <CHead T>
double EnergyHead::Model<T>::energy(const Eigen::VectorXd &D) const {
  return impl_.energy(D);
}
template <CHead T>
Eigen::VectorXd EnergyHead::Model<T>::grad(const Eigen::VectorXd &D) const {
  return impl_.grad(D);
}
template <CHead T> bool EnergyHead::Model<T>::constant_grad() const {
  return impl_.constant_grad();
}
template <CHead T> bool EnergyHead::Model<T>::has_param_jacobian() const {
  return impl_.has_param_jacobian();
}
template <CHead T>
Eigen::VectorXd
EnergyHead::Model<T>::param_grad(const Eigen::VectorXd &D) const {
  return impl_.param_grad(D);
}
template <CHead T>
Eigen::MatrixXd
EnergyHead::Model<T>::dgrad_dparam(const Eigen::VectorXd &D) const {
  return impl_.dgrad_dparam(D);
}
template <CHead T> nlohmann::json EnergyHead::Model<T>::to_json() const {
  return build_json_from_head(impl_); // ADL → concrete head's overload
}
template <CHead T> Eigen::VectorXd EnergyHead::Model<T>::all_values() const {
  return impl_.all_values();
}
template <CHead T>
void EnergyHead::Model<T>::set_all_values(const Eigen::VectorXd &v) {
  impl_.set_all_values(v);
}
template <CHead T>
std::unique_ptr<detail::HeadConcept> EnergyHead::Model<T>::remapped(
    const std::vector<std::optional<Eigen::Index>> &map) const {
  return std::make_unique<Model>(impl_.remapped(map));
}

FORCE_INLINE double EnergyHead::energy(const Eigen::VectorXd &D) const {
  return self_->energy(D);
}
FORCE_INLINE Eigen::VectorXd EnergyHead::grad(const Eigen::VectorXd &D) const {
  return self_->grad(D);
}
FORCE_INLINE bool EnergyHead::constant_grad() const {
  return self_->constant_grad();
}
FORCE_INLINE bool EnergyHead::has_param_jacobian() const {
  return self_->has_param_jacobian();
}
FORCE_INLINE Eigen::VectorXd
EnergyHead::param_grad(const Eigen::VectorXd &D) const {
  return self_->param_grad(D);
}
FORCE_INLINE Eigen::MatrixXd
EnergyHead::dgrad_dparam(const Eigen::VectorXd &D) const {
  return self_->dgrad_dparam(D);
}
FORCE_INLINE std::size_t EnergyHead::param_count() const {
  return self_->param_count();
}
FORCE_INLINE void EnergyHead::gather_params(Eigen::VectorXd &x,
                                            std::size_t off) const {
  self_->gather_params(x, off);
}
FORCE_INLINE void EnergyHead::scatter_params(const Eigen::VectorXd &x,
                                             std::size_t off) {
  self_->scatter_params(x, off);
}
FORCE_INLINE void EnergyHead::gather_bounds(Eigen::VectorXd &lo,
                                            Eigen::VectorXd &hi,
                                            std::size_t off) const {
  self_->gather_bounds(lo, hi, off);
}
FORCE_INLINE nlohmann::json EnergyHead::to_json() const {
  return self_->to_json();
}
FORCE_INLINE Eigen::VectorXd EnergyHead::all_values() const {
  return self_->all_values();
}
FORCE_INLINE void EnergyHead::set_all_values(const Eigen::VectorXd &v) {
  self_->set_all_values(v);
}
FORCE_INLINE EnergyHead EnergyHead::remapped(
    const std::vector<std::optional<Eigen::Index>> &map) const {
  return EnergyHead(self_->remapped(map));
}

} // namespace forcesmith
