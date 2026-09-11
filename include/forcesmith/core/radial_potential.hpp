#pragma once

#include "forcesmith/core/erased.hpp"
#include "forcesmith/core/erasure.hpp" // te::, for the wrappers that use it
#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/site_id.hpp"
#include "forcesmith/core/types.hpp" // FORCE_INLINE
#include "forcesmith/potentials/curvature.hpp"
#include <Eigen/Core>
#include <boost/leaf/result.hpp>
#include <concepts>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace forcesmith {

template <class T>
concept CRadialPotential =
    CFittable<T> && requires(T &mut, const T &t, double r, int site,
                             std::size_t i, Eigen::VectorXd &v) {
      { t.eval(r) } -> std::convertible_to<double>;
      { t.deriv(r) } -> std::convertible_to<double>;
      { t.span() } -> std::convertible_to<std::pair<double, double>>;
      { t.prepare_site(r) } -> std::convertible_to<int>;
      { t.eval_at(site) } -> std::convertible_to<double>;
      { t.deriv_at(site) } -> std::convertible_to<double>;
      { t.eval_and_deriv(r) } -> std::convertible_to<std::pair<double, double>>;
      {
        t.eval_and_deriv_at(site)
      } -> std::convertible_to<std::pair<double, double>>;
      mut.set_param(i, r);
      mut.set_fixed(i, true);
      mut.set_bounds(i, r, r);
      { curvature_count(t) } -> std::convertible_to<std::size_t>;
      write_curvature(t, v, i, r);
      { T::has_param_jacobian() } -> std::convertible_to<bool>;
      { t.deriv2(r) } -> std::convertible_to<double>;
    };

template <typename Derived> struct NoSiteCache {
  constexpr int prepare_site(double) const noexcept { return -1; }
  double eval_at(int) const { std::unreachable(); }
  double deriv_at(int) const { std::unreachable(); }
  std::pair<double, double> eval_and_deriv_at(int) const { std::unreachable(); }
  constexpr std::pair<double, double> eval_and_deriv(double r) const {
    const Derived &self = static_cast<const Derived &>(*this);
    return {self.eval(r), self.deriv(r)};
  }
};

struct NoParamJacobian {
  static constexpr bool has_param_jacobian() noexcept { return false; }
  void param_grad(double, std::span<double>) const { std::unreachable(); }
  void dderiv_dparam(double, std::span<double>) const { std::unreachable(); }
  double deriv2(double) const { std::unreachable(); }
};

// A potential with no raw indexed parameter access (the silent no-op the
// erasure used to supply).
struct NoRawParamAccess {
  constexpr void set_param(std::size_t, double) noexcept {}
  constexpr void set_fixed(std::size_t, bool) noexcept {}
  constexpr void set_bounds(std::size_t, double, double) noexcept {}
};

namespace detail {
struct RadialPotentialConcept : FittableConcept {
  virtual double eval(double r) const = 0;
  virtual double deriv(double r) const = 0;
  virtual int prepare_site(double r) const = 0;
  virtual double eval_at(int site) const = 0;
  virtual double deriv_at(int site) const = 0;
  virtual std::pair<double, double> eval_and_deriv(double r) const = 0;
  virtual std::pair<double, double> eval_and_deriv_at(int site) const = 0;
  virtual std::pair<double, double> span() const = 0;
  virtual void set_param(std::size_t i, double v) = 0;
  virtual void set_fixed(std::size_t i, bool f) = 0;
  virtual void set_bounds(std::size_t i, double lo, double hi) = 0;
  virtual bool has_param_jacobian() const = 0;
  virtual void param_grad(double r, std::span<double> out) const = 0;
  virtual void dderiv_dparam(double r, std::span<double> out) const = 0;
  virtual double deriv2(double r) const = 0;
  virtual std::size_t smoothness_count() const = 0;
  virtual void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                                double weight) const = 0;
  virtual std::unique_ptr<RadialPotentialConcept> clone() const = 0;
};
} // namespace detail

class RadialPotential
    : private detail::ErasedValue<detail::RadialPotentialConcept> {
  template <CRadialPotential T>
  struct Model final
      : detail::CloningModel<Model<T>, T, detail::RadialPotentialConcept> {
    using Base =
        detail::CloningModel<Model<T>, T, detail::RadialPotentialConcept>;
    using Base::Base;
    using Base::impl_;
    double eval(double r) const override { return impl_.eval(r); }
    double deriv(double r) const override { return impl_.deriv(r); }
    int prepare_site(double r) const override { return impl_.prepare_site(r); }
    double eval_at(int site) const override { return impl_.eval_at(site); }
    double deriv_at(int site) const override { return impl_.deriv_at(site); }
    std::pair<double, double> eval_and_deriv(double r) const override {
      return impl_.eval_and_deriv(r);
    }
    std::pair<double, double> eval_and_deriv_at(int site) const override {
      return impl_.eval_and_deriv_at(site);
    }
    std::pair<double, double> span() const override { return impl_.span(); }
    void set_param(std::size_t i, double v) override { impl_.set_param(i, v); }
    void set_fixed(std::size_t i, bool f) override { impl_.set_fixed(i, f); }
    void set_bounds(std::size_t i, double lo, double hi) override {
      impl_.set_bounds(i, lo, hi);
    }
    bool has_param_jacobian() const override {
      return T::has_param_jacobian();
    }
    void param_grad(double r, std::span<double> out) const override {
      impl_.param_grad(r, out);
    }
    void dderiv_dparam(double r, std::span<double> out) const override {
      impl_.dderiv_dparam(r, out);
    }
    double deriv2(double r) const override { return impl_.deriv2(r); }
    std::size_t smoothness_count() const override {
      return curvature_count(impl_);
    }
    void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                          double weight) const override {
      write_curvature(impl_, x, off, weight);
    }
  };

  using Base = detail::ErasedValue<detail::RadialPotentialConcept>;

public:
  template <typename T>
    requires(!std::same_as<std::decay_t<T>, RadialPotential> &&
             CRadialPotential<std::decay_t<T>>)
  explicit RadialPotential(T &&t)
      : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

  RadialPotential(const RadialPotential &) = default;
  RadialPotential(RadialPotential &&) noexcept = default;
  RadialPotential &operator=(const RadialPotential &) = default;
  RadialPotential &operator=(RadialPotential &&) noexcept = default;

  template <class T> const T *target() const noexcept {
    auto *m = dynamic_cast<const Model<T> *>(self_.get());
    return m ? &m->impl_ : nullptr;
  }

  FORCE_INLINE double eval(double r) const { return self_->eval(r); }
  FORCE_INLINE double deriv(double r) const { return self_->deriv(r); }

  // prepare_site mutates the site table: call it single-threaded (prepare()).
  // The eval/deriv_at family is read-only and safe in the parallel Jacobian.
  SiteId prepare_site(double r) const { return SiteId{self_->prepare_site(r)}; }
  FORCE_INLINE double eval_at(SiteId site) const {
    return self_->eval_at(site.index());
  }
  FORCE_INLINE double deriv_at(SiteId site) const {
    return self_->deriv_at(site.index());
  }
  FORCE_INLINE std::pair<double, double> eval_and_deriv(double r) const {
    return self_->eval_and_deriv(r);
  }
  FORCE_INLINE std::pair<double, double> eval_and_deriv_at(SiteId site) const {
    return self_->eval_and_deriv_at(site.index());
  }
  std::pair<double, double> span() const { return self_->span(); }

  std::size_t param_count() const { return self_->param_count(); }
  void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    self_->gather_bounds(lo, hi, off);
  }
  void set_param(std::size_t i, double v) { self_->set_param(i, v); }
  void set_fixed(std::size_t i, bool f) { self_->set_fixed(i, f); }
  void set_bounds(std::size_t i, double lo, double hi) {
    self_->set_bounds(i, lo, hi);
  }
  [[nodiscard]] bool has_param_jacobian() const {
    return self_->has_param_jacobian();
  }
  FORCE_INLINE void param_grad(double r, std::span<double> out) const {
    self_->param_grad(r, out);
  }
  FORCE_INLINE void dderiv_dparam(double r, std::span<double> out) const {
    self_->dderiv_dparam(r, out);
  }
  FORCE_INLINE double deriv2(double r) const { return self_->deriv2(r); }

  std::size_t smoothness_count() const { return self_->smoothness_count(); }
  void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                        double weight) const {
    self_->write_smoothness(x, off, weight);
  }

  [[nodiscard]] static boost::leaf::result<RadialPotential>
  from_text(std::string_view json_spec);
  [[nodiscard]] static boost::leaf::result<RadialPotential>
  from_file(const std::filesystem::path &path);
};

inline double eval_cached(const RadialPotential &p, SiteId site, double r) {
  return site.cacheable() ? p.eval_at(site) : p.eval(r);
}
inline double deriv_cached(const RadialPotential &p, SiteId site, double r) {
  return site.cacheable() ? p.deriv_at(site) : p.deriv(r);
}
inline std::pair<double, double> eval_deriv_cached(const RadialPotential &p,
                                                   SiteId site, double r) {
  return site.cacheable() ? p.eval_and_deriv_at(site) : p.eval_and_deriv(r);
}

} // namespace forcesmith
