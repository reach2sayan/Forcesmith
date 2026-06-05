#pragma once

#include "forcesmith/core/erased.hpp"
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

// Any type satisfying the following concept can be stored in a Potential:
//   eval(double)->double, deriv(double)->double, span()->pair<double,double>
//   param_count()->int, gather_params(VectorXd&,int), scatter_params(const
//   VectorXd&,int), gather_bounds(VectorXd&,VectorXd&,int)

namespace detail {
struct PotentialConcept {
  virtual ~PotentialConcept() = default;
  virtual double eval(double r) const = 0;
  virtual double deriv(double r) const = 0;
  virtual std::pair<double, double> span() const = 0;
  virtual std::size_t param_count() const = 0;
  virtual void gather_params(Eigen::VectorXd &x, std::size_t off) const = 0;
  virtual void scatter_params(const Eigen::VectorXd &x, std::size_t off) = 0;
  virtual void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                             std::size_t off) const = 0;
  virtual void set_param(std::size_t i, double v) = 0;
  virtual void set_fixed(std::size_t i, bool f) = 0;
  virtual void set_bounds(std::size_t i, double lo, double hi) = 0;
  virtual std::size_t smoothness_count() const = 0;
  virtual void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                                double weight) const = 0;
  virtual std::unique_ptr<PotentialConcept> clone() const = 0;
};
} // namespace detail

class Potential : private detail::ErasedValue<detail::PotentialConcept> {
  template <typename T> struct Model final : detail::PotentialConcept {
    T impl_;
    constexpr explicit Model(T t) : impl_(std::move(t)) {}
    constexpr double eval(double r) const override { return impl_.eval(r); }
    constexpr double deriv(double r) const override { return impl_.deriv(r); }
    constexpr std::pair<double, double> span() const override {
      return impl_.span();
    }
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
      if constexpr (requires(const T &t, Eigen::VectorXd &v, std::size_t o) {
                      t.gather_bounds(v, v, o);
                    }) {
        impl_.gather_bounds(lo, hi, off);
      } else {
        // Type carries no bound metadata → all free params unbounded.
        const std::size_t n = static_cast<std::size_t>(impl_.param_count());
        lo.segment(off, n).setConstant(-std::numeric_limits<double>::infinity());
        hi.segment(off, n).setConstant( std::numeric_limits<double>::infinity());
      }
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
  [[nodiscard]] static boost::leaf::result<Potential>
  from_text(std::string_view json_spec);
  [[nodiscard]] static boost::leaf::result<Potential>
  from_file(const std::filesystem::path &path);
};

} // namespace forcesmith
