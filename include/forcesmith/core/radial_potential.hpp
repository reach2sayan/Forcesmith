#pragma once

#include "forcesmith/core/erased.hpp"
#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/site_id.hpp"
#include "forcesmith/potentials/curvature.hpp"
#include <Eigen/Core>
#include <boost/leaf/result.hpp>
#include <concepts>
#include <filesystem>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>

namespace forcesmith {

// The leaf-side contract for a type stored in a RadialPotential: the CFittable
// optimizer trio plus the eval/deriv/span maths Model forwards to
// unconditionally. gather_bounds is supplied-by-default (see CFittable), so it
// is not demanded here. The remaining Model<T> methods stay probed for a real
// reason rather than boilerplate-avoidance: set_param/fixed/bounds and the
// prepare_site/eval_at/eval_and_deriv site-cache family are a genuine
// capability split (analytic potentials have no site cache; tabulated ones do),
// so they are intentionally NOT part of this contract.
template <class T>
concept CRadialPotential = CFittable<T> && requires(const T &t, double r) {
  { t.eval(r) } -> std::convertible_to<double>;
  { t.deriv(r) } -> std::convertible_to<double>;
  { t.span() } -> std::convertible_to<std::pair<double, double>>;
};

namespace detail {
// Inherits the optimizer param-plumbing virtuals (param_count, gather_params,
// scatter_params, gather_bounds) from FittableConcept (core/fit_params.hpp).
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
    using Base::Base;  // inherit the impl_-forwarding constructor
    using Base::impl_; // bring impl_ into scope for the bodies below
    // clone() supplied by CloningModel.
    constexpr double eval(double r) const override { return impl_.eval(r); }
    constexpr double deriv(double r) const override { return impl_.deriv(r); }

    int prepare_site(double r) const override {
      if constexpr (requires(const T &t, double rr) { t.prepare_site(rr); }) {
        return impl_.prepare_site(r);
      } else {
        return -1;
      }
    }
    double eval_at(int site) const override {
      if constexpr (requires(const T &t, int s) { t.eval_at(s); }) {
        return impl_.eval_at(site);
      } else {
        std::unreachable();
        return 0.0; // unreachable: prepare_site returned -1 for this type
      }
    }
    double deriv_at(int site) const override {
      if constexpr (requires(const T &t, int s) { t.deriv_at(s); }) {
        return impl_.deriv_at(site);
      } else {
        std::unreachable();
        return 0.0; // unreachable: prepare_site returned -1 for this type
      }
    }

    std::pair<double, double> eval_and_deriv(double r) const override {
      if constexpr (requires(const T &t, double rr) { t.eval_and_deriv(rr); }) {
        return impl_.eval_and_deriv(r);
      } else {
        return {impl_.eval(r), impl_.deriv(r)};
      }
    }
    std::pair<double, double> eval_and_deriv_at(int site) const override {
      if constexpr (requires(const T &t, int s) { t.eval_and_deriv_at(s); }) {
        return impl_.eval_and_deriv_at(site);
      } else {
        std::unreachable();
        return {0.0, 0.0};
      }
    }
    constexpr std::pair<double, double> span() const override {
      return impl_.span();
    }
    constexpr void set_param(std::size_t i, double v) override {
      if constexpr (requires(T &t, std::size_t j, double w) {
                      t.set_param(j, w);
                    }) {
        impl_.set_param(i, v);
      }
    }
    constexpr void set_fixed(std::size_t i, bool f) override {
      if constexpr (requires(T &t, std::size_t j, bool g) {
                      t.set_fixed(j, g);
                    }) {
        impl_.set_fixed(i, f);
      }
    }
    constexpr void set_bounds(std::size_t i, double lo, double hi) override {
      if constexpr (requires(T &t, std::size_t j, double a, double b) {
                      t.set_bounds(j, a, b);
                    }) {
        impl_.set_bounds(i, lo, hi);
      }
    }
    constexpr std::size_t smoothness_count() const override {
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
  constexpr explicit RadialPotential(T &&t)
      : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

  RadialPotential(const RadialPotential &) = default;
  RadialPotential(RadialPotential &&) noexcept = default;
  RadialPotential &operator=(const RadialPotential &) = default;
  RadialPotential &operator=(RadialPotential &&) noexcept = default;

  template <class T> const T *target() const noexcept {
    auto *m = dynamic_cast<const Model<T> *>(self_.get());
    return m ? &m->impl_ : nullptr;
  }

  constexpr double eval(double r) const { return self_->eval(r); }
  constexpr double deriv(double r) const { return self_->deriv(r); }

  // Public fit-cache API is typed: prepare_site hands back an opaque SiteId
  // (cacheable() for spline potentials, "none" for analytic ones — the caller
  // then uses eval/deriv(r)), and the eval/deriv_at family consume it. The raw
  // int index stays an implementation detail of the erased Model<>/concrete
  // potential (SiteId::index()). prepare_site mutates an internal site table —
  // call it single-threaded (e.g. ForceCalculator prepare()); the eval/deriv_at
  // family are read-only and safe in the parallel Jacobian.
  constexpr SiteId prepare_site(double r) const {
    return SiteId{self_->prepare_site(r)};
  }
  constexpr double eval_at(SiteId site) const {
    return self_->eval_at(site.index());
  }
  constexpr double deriv_at(SiteId site) const {
    return self_->deriv_at(site.index());
  }
  constexpr std::pair<double, double> eval_and_deriv(double r) const {
    return self_->eval_and_deriv(r);
  }
  constexpr std::pair<double, double> eval_and_deriv_at(SiteId site) const {
    return self_->eval_and_deriv_at(site.index());
  }
  constexpr std::pair<double, double> span() const { return self_->span(); }
  constexpr std::size_t param_count() const { return self_->param_count(); }
  constexpr void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
  constexpr void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                               std::size_t off) const {
    self_->gather_bounds(lo, hi, off);
  }
  constexpr void set_param(std::size_t i, double v) { self_->set_param(i, v); }
  constexpr void set_fixed(std::size_t i, bool f) { self_->set_fixed(i, f); }
  constexpr void set_bounds(std::size_t i, double lo, double hi) {
    self_->set_bounds(i, lo, hi);
  }
  constexpr std::size_t smoothness_count() const {
    return self_->smoothness_count();
  }
  constexpr void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                                  double weight) const {
    self_->write_smoothness(x, off, weight);
  }

  // ── parsing factories (the object owns its parsing) ───────────────────────
  // Build ONE potential from a single JSON spec. The spec is self-describing:
  // an object with a "type" key is analytic (looked up in the maker registry);
  // an object with a "knots" array is a tabulated SplinePotential. Defined in
  // src/io/potential_reader.cpp (keeps nlohmann + the registry out of core).
  // Leaf-returning, not a throwing ctor — see feedback_error_handling.
  [[nodiscard]] static boost::leaf::result<RadialPotential>
  from_text(std::string_view json_spec);
  [[nodiscard]] static boost::leaf::result<RadialPotential>
  from_file(const std::filesystem::path &path);
};

// Spline-cache dispatch used by the force calculators in their per-bond hot
// loops: when a bond was primed by prepare() (site >= 0) evaluate via the
// cached site — no binary search; otherwise (rescale/tests that never call
// prepare()) fall back to a direct evaluation at r.
inline double eval_cached(const RadialPotential &p, SiteId site, double r) {
  return site.cacheable() ? p.eval_at(site) : p.eval(r);
}
inline double deriv_cached(const RadialPotential &p, SiteId site, double r) {
  return site.cacheable() ? p.deriv_at(site) : p.deriv(r);
}
// Fused value+derivative variant of the above: one dispatch returns both,
// reusing the cached site (or the single interval search on the fallback path).
inline std::pair<double, double> eval_deriv_cached(const RadialPotential &p,
                                                   SiteId site, double r) {
  return site.cacheable() ? p.eval_and_deriv_at(site) : p.eval_and_deriv(r);
}

} // namespace forcesmith
