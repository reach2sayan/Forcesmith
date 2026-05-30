#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <utility>

namespace potfit {

// CRTP base: stores params + [rmin, rmax] and forwards eval/deriv to Derived.
// Derived must implement eval_impl(double)->double and
// deriv_impl(double)->double.
template <typename Derived, std::size_t N> struct AnalyticBase {
  std::array<double, N> params;
  double rmin, rmax;

  constexpr double eval(double r) const {
    return static_cast<const Derived &>(*this).eval_impl(r);
  }
  constexpr double deriv(double r) const {
    return static_cast<const Derived &>(*this).deriv_impl(r);
  }
  constexpr std::pair<double, double> span() const { return {rmin, rmax}; }
  constexpr int param_count() const { return static_cast<int>(N); }
  constexpr void gather_params(Eigen::VectorXd &dst, int offset) const {
    std::ranges::copy(params, dst.data() + offset);
  }

  constexpr void scatter_params(const Eigen::VectorXd &src, int offset) {
    std::copy_n(src.data() + offset, N, params.begin());
  }

protected:
  AnalyticBase(std::array<double, N> p, double lo, double hi) noexcept
      : params(p), rmin(lo), rmax(hi) {}
};

// ── Lennard-Jones: V(r) = 4ε[(σ/r)^12 − (σ/r)^6] ──────────────────────────
// params: {epsilon, sigma}
struct LennardJones : AnalyticBase<LennardJones, 2> {
  LennardJones(double epsilon, double sigma, double lo, double hi)
      : AnalyticBase({epsilon, sigma}, lo, hi) {}

  constexpr double eval_impl(double r) const {
    const auto [ep, sig] = params;
    const double sr6 = std::pow(sig / r, 6);
    return 4.0 * ep * (sr6 * sr6 - sr6);
  }
  constexpr double deriv_impl(double r) const {
    const auto [ep, sig] = params;
    const double sr6 = std::pow(sig / r, 6);
    return 4.0 * ep * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;
  }
};

// ── Morse: V(r) = D_e[(1−e^{−a(r−r_e)})^2 − 1] ────────────────────────────
// params: {D_e, a, r_e}
struct Morse : AnalyticBase<Morse, 3> {
  Morse(double De, double a, double re, double lo, double hi)
      : AnalyticBase({De, a, re}, lo, hi) {}

  constexpr double eval_impl(double r) const {
    const auto [De, a, re] = params;
    const double e = std::exp(-a * (r - re));
    return De * (1.0 - e) * (1.0 - e) - De;
  }
  constexpr double deriv_impl(double r) const {
    const auto [De, a, re] = params;
    const double e = std::exp(-a * (r - re));
    return 2.0 * De * a * e * (1.0 - e);
  }
};

} // namespace potfit
