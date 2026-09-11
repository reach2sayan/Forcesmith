#pragma once

#include "forcesmith/core/radial_potential.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <utility>

namespace forcesmith {

template <typename Derived>
struct DelegatingPotential : NoSiteCache<Derived>, NoRawParamAccess,
                             NoParamJacobian {
  std::size_t param_count() const { return self().delegate().param_count(); }
  void gather_params(Eigen::VectorXd &v, std::size_t off) const {
    self().delegate().gather_params(v, off);
  }
  void scatter_params(const Eigen::VectorXd &v, std::size_t off) {
    mut().delegate().scatter_params(v, off);
  }
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    self().delegate().gather_bounds(lo, hi, off);
  }
  std::pair<double, double> span() const { return self().delegate().span(); }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }
  Derived &mut() { return static_cast<Derived &>(*this); }
};

struct LinearAdjustedPotential : DelegatingPotential<LinearAdjustedPotential> {
  RadialPotential base;
  double slope = 0.0;
  double intercept = 0.0;

  LinearAdjustedPotential(RadialPotential b, double s, double i)
      : base(std::move(b)), slope(s), intercept(i) {}

  const RadialPotential &delegate() const { return base; }
  RadialPotential &delegate() { return base; }

  double eval(double x) const { return base.eval(x) - slope * x - intercept; }
  double deriv(double x) const { return base.deriv(x) - slope; }
};

struct ScaledOutputPotential : DelegatingPotential<ScaledOutputPotential> {
  RadialPotential base;
  double a = 1.0;

  ScaledOutputPotential(RadialPotential b, double scale)
      : base(std::move(b)), a(scale) {}

  const RadialPotential &delegate() const { return base; }
  RadialPotential &delegate() { return base; }

  double eval(double r) const { return a * base.eval(r); }
  double deriv(double r) const { return a * base.deriv(r); }
};

struct ScaledArgPotential : DelegatingPotential<ScaledArgPotential> {
  RadialPotential base;
  double a = 1.0;

  ScaledArgPotential(RadialPotential b, double scale)
      : base(std::move(b)), a(scale) {}

  const RadialPotential &delegate() const { return base; }
  RadialPotential &delegate() { return base; }

  double eval(double rho) const { return base.eval(rho / a); }
  double deriv(double rho) const { return base.deriv(rho / a) / a; }
  std::pair<double, double> span() const {
    auto [lo, hi] = base.span();
    return {a * lo, a * hi};
  }
};

struct CompensatedPairPotential
    : DelegatingPotential<CompensatedPairPotential> {
  RadialPotential phi;
  RadialPotential g_alpha;
  RadialPotential g_beta;
  double coeff_alpha = 0.0;
  double coeff_beta = 0.0;

  CompensatedPairPotential(RadialPotential p, RadialPotential ga,
                           RadialPotential gb, double ca, double cb)
      : phi(std::move(p)), g_alpha(std::move(ga)), g_beta(std::move(gb)),
        coeff_alpha(ca), coeff_beta(cb) {}

  const RadialPotential &delegate() const { return phi; }
  RadialPotential &delegate() { return phi; }

  double eval(double r) const {
    return phi.eval(r) + coeff_alpha * g_beta.eval(r) +
           coeff_beta * g_alpha.eval(r);
  }
  double deriv(double r) const {
    return phi.deriv(r) + coeff_alpha * g_beta.deriv(r) +
           coeff_beta * g_alpha.deriv(r);
  }
};

} // namespace forcesmith
