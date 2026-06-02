#pragma once

#include "potfit/core/potential_base.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <utility>

namespace potfit {

// Wrapper: eval(x) = base.eval(x) − slope × x − intercept.
// gather/scatter delegate to base so optimizer sees unchanged parameter layout.
struct LinearAdjustedPotential {
  Potential base;
  double slope = 0.0;
  double intercept = 0.0;

  constexpr double eval(double x) const {
    return base.eval(x) - slope * x - intercept;
  }
  constexpr double deriv(double x) const { return base.deriv(x) - slope; }
  constexpr std::pair<double, double> span() const { return base.span(); }
  constexpr std::size_t param_count() const { return base.param_count(); }
  constexpr void gather_params(Eigen::VectorXd &v, int off) const {
    base.gather_params(v, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &v, int off) {
    base.scatter_params(v, off);
  }
};

// Wrapper: scales the OUTPUT by a — density g(r) → a·g(r). Used for the
// rho-axis stretch. gather/scatter delegate to base (knot params unchanged).
struct ScaledOutputPotential {
  Potential base;
  double a = 1.0;

  constexpr double eval(double r) const { return a * base.eval(r); }
  constexpr double deriv(double r) const { return a * base.deriv(r); }
  constexpr std::pair<double, double> span() const { return base.span(); }
  constexpr std::size_t param_count() const { return base.param_count(); }
  constexpr void gather_params(Eigen::VectorXd &v, int off) const {
    base.gather_params(v, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &v, int off) {
    base.scatter_params(v, off);
  }
};

// Wrapper: scales the ARGUMENT by 1/a — embedding F(ρ) → F(ρ/a), so the stretch
// is energy-preserving (F_new(a·ρ_old) = F_old(ρ_old)). The span scales by a.
struct ScaledArgPotential {
  Potential base;
  double a = 1.0;

  constexpr double eval(double rho) const { return base.eval(rho / a); }
  constexpr double deriv(double rho) const { return base.deriv(rho / a) / a; }
  constexpr std::pair<double, double> span() const {
    auto [lo, hi] = base.span();
    return {a * lo, a * hi};
  }
  constexpr std::size_t param_count() const { return base.param_count(); }
  constexpr void gather_params(Eigen::VectorXd &v, int off) const {
    base.gather_params(v, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &v, int off) {
    base.scatter_params(v, off);
  }
};

// Wrapper: eval(r) = phi.eval(r) + coeff_alpha × g_beta.eval(r) + coeff_beta ×
// g_alpha.eval(r). Implements the EAM gauge compensation for a linear F shift.
// gather/scatter delegate to phi; density copies are fixed at rescale time.
struct CompensatedPairPotential {
  Potential phi;
  Potential g_alpha;
  Potential g_beta;
  double coeff_alpha = 0.0;
  double coeff_beta = 0.0;

  constexpr double eval(double r) const {
    return phi.eval(r) + coeff_alpha * g_beta.eval(r) +
           coeff_beta * g_alpha.eval(r);
  }
  constexpr double deriv(double r) const {
    return phi.deriv(r) + coeff_alpha * g_beta.deriv(r) +
           coeff_beta * g_alpha.deriv(r);
  }
  constexpr std::pair<double, double> span() const { return phi.span(); }
  constexpr int param_count() const { return phi.param_count(); }
  void gather_params(Eigen::VectorXd &v, int off) const {
    phi.gather_params(v, off);
  }
  void scatter_params(const Eigen::VectorXd &v, int off) {
    phi.scatter_params(v, off);
  }
};

} // namespace potfit
