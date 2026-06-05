#pragma once

#include "forcesmith/core/param.hpp"
#include "forcesmith/core/types.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>

namespace forcesmith {

// CRTP base for analytic potentials: holds the parameter array + range and
// generates the optimizer param-plumbing
// (param_count/gather/scatter/set_param). Each Derived supplies the maths
// directly — eval(double) and deriv(double) — defined inline below so the
// compiler inlines them at the Potential/SmoothCutoff call sites.
template <typename Derived, std::size_t N> struct AnalyticParams {
  static constexpr std::size_t num_params = N;

protected:
  std::array<Param, N> params;
  double rmin, rmax;

  constexpr AnalyticParams(std::array<double, N> vals, double lo,
                           double hi) noexcept
      : rmin(lo), rmax(hi) {
    std::ranges::transform(vals, params.begin(),
                           [](auto v) { return Param{v}; });
  }

public:
  constexpr std::pair<double, double> span() const { return {rmin, rmax}; }

  // Mark parameter i as fixed (excluded from optimizer) or free.
  constexpr void set_fixed(std::size_t i, bool f) { params[i].fixed = f; }
  constexpr bool is_fixed(std::size_t i) const { return params[i].fixed; }

  // Write parameter i directly, bypassing the fixed flag. Used to broadcast a
  // shared global parameter (e.g. a smooth-cutoff h) into this potential's
  // slot, which is held fixed so gather/scatter skip it.
  constexpr void set_param(std::size_t i, double v) { params[i].value = v; }

  constexpr std::size_t param_count() const {
    return std::ranges::count_if(params,
                                 [](const auto &p) { return !p.fixed; });
  }
  constexpr void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    for (const auto &p : params) {
      if (!p.fixed) {
        dst[off++] = p.value;
      }
    }
  }
  constexpr void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    for (auto &p : params) {
      if (!p.fixed) {
        p.value = src[off++];
      }
    }
  }
  // Per-free-param box constraints, in the same order as gather_params so the
  // bound vectors align element-for-element with the gathered x.
  constexpr void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                               std::size_t off) const {
    for (const auto &p : params) {
      if (!p.fixed) {
        lo[off] = p.min;
        hi[off] = p.max;
        ++off;
      }
    }
  }
  // Set parameter i's [min, max] box (raw index, bypassing the fixed flag).
  constexpr void set_bounds(std::size_t i, double lo, double hi) {
    params[i].min = lo;
    params[i].max = hi;
  }
};

namespace detail {

// Tersoff / SW smooth cosine cutoff and its derivative.
FORCE_INLINE constexpr double fc(double r, double R, double S) noexcept {
  if (r <= R) {
    return 1.0;
  }
  if (r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return 0.5 + 0.5 * std::cos(x);
}

constexpr FORCE_INLINE double dfc(double r, double R, double S) noexcept {
  if (r <= R || r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return -0.5 * std::numbers::pi / (S - R) * std::sin(x);
}

// forcesmith's smooth-cutoff switching factor (the `_sc` function variants):
//   apot_cutoff(r, r0, h) = u⁴/(1+u⁴),  u = (r − r0)/h,  and 0 for r ≥ r0.
// r0 is the function's cutoff (rmax). Value AND derivative → 0 as r → r0, so a
// _sc potential is continuous at the cutoff (unlike a hard truncation). This is
// a DIFFERENT function from the cosine fc above (used by Tersoff/SW); it
// matches forcesmith's apot_cutoff exactly (functions.c:322) for apple-to-apple
// parity.
constexpr FORCE_INLINE double apot_cutoff(double r, double r0,
                                          double h) noexcept {
  if (r >= r0) {
    return 0.0;
  }
  const double u = (r - r0) / h;
  const double u4 = (u * u) * (u * u);
  return u4 / (1.0 + u4);
}

// d/dr of apot_cutoff: with u=(r−r0)/h, c=u⁴/(1+u⁴) ⇒ dc/dr = (4u³/h)/(1+u⁴)².
constexpr FORCE_INLINE double apot_cutoff_deriv(double r, double r0,
                                                double h) noexcept {
  if (r >= r0) {
    return 0.0;
  }
  const double u = (r - r0) / h;
  const double u3 = u * u * u;
  const double denom = 1.0 + u3 * u;
  return (4.0 * u3 / h) / (denom * denom);
}

} // namespace detail

// Lennard-Jones: V(r) = 4ε[(σ/r)^12 − (σ/r)^6]
// params: {epsilon, sigma}
struct LennardJones : AnalyticParams<LennardJones, 2> {
  constexpr LennardJones(double epsilon, double sigma, double lo, double hi)
      : AnalyticParams({epsilon, sigma}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [ep, sig] = params;
    const double sr6 = std::pow(sig / r, 6);
    return 4.0 * ep * (sr6 * sr6 - sr6);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [ep, sig] = params;
    const double sr6 = std::pow(sig / r, 6);
    return 4.0 * ep * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;
  }
};

// Morse: V(r) = D_e[(1−e^{−a(r−r_e)})^2 − 1]
// params: {D_e, a, r_e}
struct Morse : AnalyticParams<Morse, 3> {
  constexpr Morse(double De, double a, double re, double lo, double hi)
      : AnalyticParams({De, a, re}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [De, a, re] = params;
    const double e = std::exp(-a * (r - re));
    return De * (1.0 - e) * (1.0 - e) - De;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [De, a, re] = params;
    const double e = std::exp(-a * (r - re));
    return 2.0 * De * a * e * (1.0 - e);
  }
};

// ── Buckingham: V = A exp(−r/ρ) − C·ρ^6/r^6 ─────────────────────────────────
// Matches forcesmith buck_value: the dispersion term carries a ρ^6 factor.
// params: {A, rho, C}
struct Buckingham : AnalyticParams<Buckingham, 3> {
  constexpr Buckingham(double A, double rho, double C, double lo, double hi)
      : AnalyticParams({A, rho, C}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, rho, C] = params;
    const double x = (rho * rho) / (r * r);
    return A * std::exp(-r / rho) - C * x * x * x; // C·(ρ²/r²)³ = C·ρ⁶/r⁶
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, rho, C] = params;
    const double rho6 = std::pow(rho, 6);
    return -(A / rho) * std::exp(-r / rho) + 6.0 * C * rho6 / std::pow(r, 7);
  }
};

// ── Born: V = A exp((C−r)/B) − D/r^6 + E/r^8 ────────────────────────────────
// Matches forcesmith born_value. params: {A, B, C, D, E}
//   A: amplitude, B: range, C: offset inside exponent, D: r^6, E: r^8.
struct Born : AnalyticParams<Born, 5> {
  constexpr Born(double A, double B, double C, double D, double E, double lo,
                 double hi)
      : AnalyticParams({A, B, C, D, E}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, C, D, E] = params;
    const double r2 = r * r, r6 = r2 * r2 * r2, r8 = r6 * r2;
    return A * std::exp((C - r) / B) - D / r6 + E / r8;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, C, D, E] = params;
    return -(A / B) * std::exp((C - r) / B) + 6.0 * D / std::pow(r, 7) -
           8.0 * E / std::pow(r, 9);
  }
};

// ── PowerDecay: V = A/r^n ────────────────────────────────────────────────────
// params: {A, n}
struct PowerDecay : AnalyticParams<PowerDecay, 2> {
  constexpr PowerDecay(double A, double n, double lo, double hi)
      : AnalyticParams({A, n}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, n] = params;
    return A / std::pow(r, n);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, n] = params;
    return -A * n / std::pow(r, n + 1.0);
  }
};

// ── ExpDecay: V = A exp(−Br) ─────────────────────────────────────────────────
// params: {A, B}
struct ExpDecay : AnalyticParams<ExpDecay, 2> {
  constexpr ExpDecay(double A, double B, double lo, double hi)
      : AnalyticParams({A, B}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B] = params;
    return A * std::exp(-B * r);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B] = params;
    return -A * B * std::exp(-B * r);
  }
};

// ── MexpDecay, matches forcesmith mexp_decay_value: V = A·exp(−B·(r − r0)) ───────
// params order follows forcesmith p[]: {A, B, r0}
struct MexpDecay : AnalyticParams<MexpDecay, 3> {
  constexpr MexpDecay(double A, double B, double r0, double lo, double hi)
      : AnalyticParams({A, B, r0}, lo, hi) {}
  FORCE_INLINE double eval(double r) const {
    const auto [A, B, r0] = params;
    return A * std::exp(-B * (r - r0));
  }
  FORCE_INLINE double deriv(double r) const {
    const auto [A, B, r0] = params;
    return -A * B * std::exp(-B * (r - r0));
  }
};

// ── Harmonic: V = k(r−r0)² ───────────────────────────────────────────────────
// params: {k, r0}
struct Harmonic : AnalyticParams<Harmonic, 2> {
  constexpr Harmonic(double k, double r0, double lo, double hi)
      : AnalyticParams({k, r0}, lo, hi) {}
  constexpr double eval(double r) const {
    const auto [k, r0] = params;
    return k * (r - r0) * (r - r0);
  }
  constexpr double deriv(double r) const {
    const auto [k, r0] = params;
    return 2.0 * k * (r - r0);
  }
};

// ── Universal embedding function, matches forcesmith universal_value:
//   V = E0·(b/(b−a)·r^a − a/(b−a)·r^b) + c·r
// params order follows forcesmith p[]: {E0, a, b, c}
struct Universal : AnalyticParams<Universal, 4> {
  Universal(double E0, double a, double b, double c, double lo, double hi)
      : AnalyticParams({E0, a, b, c}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [E0, a, b, c] = params;
    return E0 * (b / (b - a) * std::pow(r, a) - a / (b - a) * std::pow(r, b)) +
           c * r;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [E0, a, b, c] = params;
    return E0 * (a * b / (b - a)) *
               (std::pow(r, a - 1.0) - std::pow(r, b - 1.0)) +
           c;
  }
};

// ── Eopp (empirical oscillating pair), matches forcesmith eopp_value:
//   V = A/r^n + (B/r^m)·cos(k·r + φ)
// params order follows forcesmith p[]: {A, n, B, m, k, phi}
struct Eopp : AnalyticParams<Eopp, 6> {
  constexpr Eopp(double A, double n, double B, double m, double k, double phi,
                 double lo, double hi)
      : AnalyticParams({A, n, B, m, k, phi}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, n, B, m, k, phi] = params;
    return A / std::pow(r, n) + (B / std::pow(r, m)) * std::cos(k * r + phi);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, n, B, m, k, phi] = params;
    const double c = std::cos(k * r + phi), s = std::sin(k * r + phi);
    return -A * n / std::pow(r, n + 1.0) - B * m / std::pow(r, m + 1.0) * c -
           B * k / std::pow(r, m) * s;
  }
};

// Generic `_sc` smooth-cutoff wrapper. Wraps ANY base analytic potential and
// multiplies its value by detail::apot_cutoff(r, rmax, h); the derivative uses
// the product rule (base'·c + base·c'). This mirrors forcesmith's generic `_sc`
// mechanism exactly: a name ending in `_sc` appends ONE parameter h (the
// switching width) after the base potential's parameters, and the cutoff radius
// is the potential's rmax. apot_cutoff takes value AND derivative to zero at
// the cutoff, so the base must NOT itself early-cut at rmax (it already zeroes
// there). Replaces the per-function LjSC/MorseSC/ExpDecaySC/EoppSC structs;
// any base becomes `_sc` for free by registering SmoothCutoff(Base{...}, h).
template <typename Base> struct SmoothCutoff {
  Base base;
  Param h;
  constexpr SmoothCutoff(Base b, double h_val) : base(std::move(b)), h(h_val) {}
  constexpr FORCE_INLINE double eval(double r) const {
    return base.eval(r) * detail::apot_cutoff(r, base.span().second, h.value);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const double r0 = base.span().second;
    return base.deriv(r) * detail::apot_cutoff(r, r0, h.value) +
           base.eval(r) * detail::apot_cutoff_deriv(r, r0, h.value);
  }
  constexpr std::pair<double, double> span() const { return base.span(); }
  constexpr std::size_t param_count() const {
    return base.param_count() + (h.fixed ? 0 : 1);
  }
  constexpr void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    base.gather_params(dst, off);
    if (!h.fixed) {
      dst[off + base.param_count()] = h.value;
    }
  }
  constexpr void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    base.scatter_params(src, off);
    if (!h.fixed) {
      h.value = src[off + base.param_count()];
    }
  }
  constexpr void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                               std::size_t off) const {
    base.gather_bounds(lo, hi, off);
    if (!h.fixed) {
      lo[off + base.param_count()] = h.min;
      hi[off + base.param_count()] = h.max;
    }
  }
  // Raw-index parameter access (used to broadcast a shared global h). Indices
  // 0..N-1 address the base; index N addresses the appended cutoff width h.
  constexpr void set_param(std::size_t i, double v) {
    if (i < Base::num_params) {
      base.set_param(i, v);
    } else {
      h.value = v;
    }
  }
  constexpr void set_fixed(std::size_t i, bool f) {
    if (i < Base::num_params) {
      base.set_fixed(i, f);
    } else {
      h.fixed = f;
    }
  }
  constexpr void set_bounds(std::size_t i, double lo, double hi) {
    if (i < Base::num_params) {
      base.set_bounds(i, lo, hi);
    } else {
      h.min = lo;
      h.max = hi;
    }
  }
  constexpr bool is_fixed(std::size_t i) const {
    return i < Base::num_params ? base.is_fixed(i) : h.fixed;
  }
};

// ── EoppExp, matches forcesmith eopp_exp_value:
//   V = A·exp(−B·r) + (C/r^m)·cos(k·r + φ)
// params order follows forcesmith p[]: {A, B, C, m, k, phi}
struct EoppExp : AnalyticParams<EoppExp, 6> {
  constexpr EoppExp(double A, double B, double C, double m, double k,
                    double phi, double lo, double hi)
      : AnalyticParams({A, B, C, m, k, phi}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, C, m, k, phi] = params;
    return A * std::exp(-B * r) + (C / std::pow(r, m)) * std::cos(k * r + phi);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, C, m, k, phi] = params;
    const double cc = std::cos(k * r + phi), s = std::sin(k * r + phi);
    return -A * B * std::exp(-B * r) - C * m / std::pow(r, m + 1.0) * cc -
           C * k / std::pow(r, m) * s;
  }
};

// ── Meopp (modified eopp), matches forcesmith meopp_value:
//   V = A/(r−r0)^n + (B/r^m)·cos(k·r + φ)
// params order follows forcesmith p[]: {A, n, B, m, k, phi, r0}
struct Meopp : AnalyticParams<Meopp, 7> {
  constexpr Meopp(double A, double n, double B, double m, double k, double phi,
                  double r0, double lo, double hi)
      : AnalyticParams({A, n, B, m, k, phi, r0}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, n, B, m, k, phi, r0] = params;
    return A / std::pow(r - r0, n) +
           (B / std::pow(r, m)) * std::cos(k * r + phi);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, n, B, m, k, phi, r0] = params;
    const double c = std::cos(k * r + phi), s = std::sin(k * r + phi);
    return -A * n / std::pow(r - r0, n + 1.0) -
           B * m / std::pow(r, m + 1.0) * c - B * k / std::pow(r, m) * s;
  }
};

// ── GenLJ (generalized Lennard-Jones), matches forcesmith gen_lj_value:
//   x = r/r0;  V = A/(m−n)·(m·x^{−n} − n·x^{−m}) + B
// params order follows forcesmith p[]: {A, n, m, r0, B}
struct GenLJ : AnalyticParams<GenLJ, 5> {
  constexpr GenLJ(double A, double n, double m, double r0, double B, double lo,
                  double hi)
      : AnalyticParams({A, n, m, r0, B}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, n, m, r0, B] = params;
    const double x = r / r0;
    return A / (m - n) * (m * std::pow(x, -n) - n * std::pow(x, -m)) + B;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, n, m, r0, B] = params;
    const double x = r / r0;
    // dV/dr = A/(m−n)·(1/r0)·(−mn·x^{−n−1} + nm·x^{−m−1})
    return A / (m - n) / r0 * m * n *
           (std::pow(x, -m - 1.0) - std::pow(x, -n - 1.0));
  }
};

// ── DoubleMorse: sum of two Morse terms + constant offset ────────────────────
// params: {D1, a1, r1, D2, a2, r2, C}
struct DoubleMorse : AnalyticParams<DoubleMorse, 7> {
  constexpr DoubleMorse(double D1, double a1, double r1, double D2, double a2,
                        double r2, double C, double lo, double hi)
      : AnalyticParams({D1, a1, r1, D2, a2, r2, C}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [D1, a1, r1, D2, a2, r2, C] = params;
    const double e1 = std::exp(-a1 * (r - r1));
    const double e2 = std::exp(-a2 * (r - r2));
    return D1 * ((1.0 - e1) * (1.0 - e1) - 1.0) +
           D2 * ((1.0 - e2) * (1.0 - e2) - 1.0) + C;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [D1, a1, r1, D2, a2, r2, C] = params;
    const double e1 = std::exp(-a1 * (r - r1));
    const double e2 = std::exp(-a2 * (r - r2));
    return 2.0 * D1 * a1 * e1 * (1.0 - e1) + 2.0 * D2 * a2 * e2 * (1.0 - e2);
  }
};

// ── DoubleExp, matches forcesmith double_exp_value:
//   V = A·exp(−B·(r − r1)²) + exp(−C·(r − r2))   (2nd term has no prefactor)
// params order follows forcesmith p[]: {A, B, r1, C, r2}
struct DoubleExp : AnalyticParams<DoubleExp, 5> {
  constexpr DoubleExp(double A, double B, double r1, double C, double r2,
                      double lo, double hi)
      : AnalyticParams({A, B, r1, C, r2}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, r1, C, r2] = params;
    const double dr = r - r1;
    return A * std::exp(-B * dr * dr) + std::exp(-C * (r - r2));
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, r1, C, r2] = params;
    const double dr = r - r1;
    return -2.0 * A * B * dr * std::exp(-B * dr * dr) -
           C * std::exp(-C * (r - r2));
  }
};

// ── Mishin, matches forcesmith mishin_value:
//   z = r − r0;  e = exp(−d·z);  V = A·z^n·e·(1 + B·e) + C
// params order follows forcesmith p[]: {A, B, C, r0, n, d}
struct Mishin : AnalyticParams<Mishin, 6> {
  constexpr Mishin(double A, double B, double C, double r0, double n, double d,
                   double lo, double hi)
      : AnalyticParams({A, B, C, r0, n, d}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, C, r0, n, d] = params;
    const double z = r - r0;
    const double e = std::exp(-d * z);
    return A * std::pow(z, n) * e * (1.0 + B * e) + C;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, C, r0, n, d] = params;
    const double z = r - r0;
    const double e = std::exp(-d * z);
    const double zn1 = std::pow(z, n - 1.0);
    // d/dr[A z^n e]      = A z^{n-1} e (n − d z)
    // d/dr[A B z^n e^2]  = A B z^{n-1} e^2 (n − 2 d z)
    return A * zn1 * e * (n - d * z) + A * B * zn1 * e * e * (n - 2.0 * d * z);
  }
};

// SqrtFunc: V = A sqrt(r / B)
// Matches forcesmith sqrt_value. params: {A, B}
struct SqrtFunc : AnalyticParams<SqrtFunc, 2> {
  constexpr SqrtFunc(double A, double B, double lo, double hi)
      : AnalyticParams({A, B}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B] = params;
    return A * std::sqrt(r / B);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B] = params;
    return A / (2.0 * B * std::sqrt(r / B));
  }
};

// ConstFunc: V = C
// // params: {C}
struct ConstFunc : AnalyticParams<ConstFunc, 1> {
  constexpr explicit ConstFunc(double C, double lo, double hi)
      : AnalyticParams({C}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double) const { return params[0]; }
  constexpr FORCE_INLINE double deriv(double) const { return 0.0; }
};

// ── Parabola: V = Ar² + Br + C ───────────────────────────────────────────────
// params: {A, B, C}
struct Parabola : AnalyticParams<Parabola, 3> {
  constexpr Parabola(double A, double B, double C, double lo, double hi)
      : AnalyticParams({A, B, C}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, C] = params;
    return A * r * r + B * r + C;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, C] = params;
    return 2.0 * A * r + B;
  }
};

// ── Poly5, matches forcesmith poly_5_value (expansion about r = 1):
//   s = r − 1;  V = a0 + 0.5·a1·s² + a2·s³ + a3·s⁴ + a4·s⁵
// params: {a0, a1, a2, a3, a4}
struct Poly5 : AnalyticParams<Poly5, 5> {
  constexpr Poly5(double a0, double a1, double a2, double a3, double a4,
                  double lo, double hi)
      : AnalyticParams({a0, a1, a2, a3, a4}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [a0, a1, a2, a3, a4] = params;
    const double s = r - 1.0, s2 = s * s;
    return a0 + 0.5 * a1 * s2 + a2 * s * s2 + a3 * s2 * s2 + a4 * s2 * s2 * s;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [a0, a1, a2, a3, a4] = params;
    const double s = r - 1.0, s2 = s * s;
    return a1 * s + 3.0 * a2 * s2 + 4.0 * a3 * s2 * s + 5.0 * a4 * s2 * s2;
  }
};

// ── StiwWeb2 (Stillinger-Weber pair), matches forcesmith stiweb_2_value:
//   V = (A·r^{−p} − B·r^{−q})·exp(δ/(r − rc))
// params order follows forcesmith p[]: {A, B, p, q, delta, rc}
struct StiwWeb2 : AnalyticParams<StiwWeb2, 6> {
  constexpr StiwWeb2(double A, double B, double p, double q, double delta,
                     double rc, double lo, double hi)
      : AnalyticParams({A, B, p, q, delta, rc}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, p, q, delta, rc] = params;
    if (r >= rc)
      return 0.0; // exp pole at r = rc; SW pair vanishes beyond
    const double poly = A * std::pow(r, -p) - B * std::pow(r, -q);
    return poly * std::exp(delta / (r - rc));
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, p, q, delta, rc] = params;
    if (r >= rc)
      return 0.0;
    const double d = r - rc;
    const double g = std::exp(delta / d);
    const double poly = A * std::pow(r, -p) - B * std::pow(r, -q);
    const double dpoly =
        -A * p * std::pow(r, -p - 1.0) + B * q * std::pow(r, -q - 1.0);
    return dpoly * g + poly * g * (-delta / (d * d));
  }
};

// ── StiwWeb3: h(r) = exp(γ/(r−a)),  r < a  [SW 3-body radial function] ───────
// params: {gamma, a}   (γ = γ_SW × σ and a = a_SW × σ, pre-multiplied)
struct StiwWeb3 : AnalyticParams<StiwWeb3, 2> {
  constexpr StiwWeb3(double gamma, double a, double lo, double hi)
      : AnalyticParams({gamma, a}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [gamma, a] = params;
    if (r >= a)
      return 0.0;
    return std::exp(gamma / (r - a));
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [gamma, a] = params;
    if (r >= a)
      return 0.0;
    const double d = r - a;
    const double h = std::exp(gamma / d);
    if (h == 0.0)
      return 0.0;
    return h * (-gamma / (d * d));
  }
};

// ── TersoffPot: V = fc(r)[A exp(−λr) − B exp(−μr)] ──────────────────────────
// Bond-order params β, n, c, d, h are stored but not used in pure pair eval.
// params: {A, B, lambda, mu, beta, n, c, d, h, R, S}
struct TersoffPot : AnalyticParams<TersoffPot, 11> {
  constexpr TersoffPot(double A, double B, double lam, double mu, double beta,
                       double n, double c, double d, double h, double R,
                       double S, double lo, double hi)
      : AnalyticParams({A, B, lam, mu, beta, n, c, d, h, R, S}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const double A = params[0], B = params[1], lam = params[2], mu = params[3];
    const double R = params[9], S = params[10];
    return detail::fc(r, R, S) *
           (A * std::exp(-lam * r) - B * std::exp(-mu * r));
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const double A = params[0], B = params[1], lam = params[2], mu = params[3];
    const double R = params[9], S = params[10];
    const double f = detail::fc(r, R, S);
    const double df = detail::dfc(r, R, S);
    const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
    const double dpair =
        -lam * A * std::exp(-lam * r) + mu * B * std::exp(-mu * r);
    return df * pair + f * dpair;
  }
};

// ── TersoffMix: mixing correction V = χ exp(−ω r) ────────────────────────────
// params: {chi, omega}
struct TersoffMix : AnalyticParams<TersoffMix, 2> {
  constexpr TersoffMix(double chi, double omega, double lo, double hi)
      : AnalyticParams({chi, omega}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [chi, omega] = params;
    return chi * std::exp(-omega * r);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [chi, omega] = params;
    return -chi * omega * std::exp(-omega * r);
  }
};

// ── TersoffModPot: modified Tersoff pair (16 params) ─────────────────────────
// Extended pair: fc(r)[A exp(−λr) − B exp(−μr)] × polynomial correction.
// Extra params c1..c5 (indices 11-15) provide a polynomial correction.
// params: {A, B, lambda, mu, beta, n, c, d, h, R, S, c1, c2, c3, c4, c5}
struct TersoffModPot : AnalyticParams<TersoffModPot, 16> {
  constexpr TersoffModPot(std::array<double, 16> p, double lo, double hi)
      : AnalyticParams(p, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const double A = params[0], B = params[1];
    const double lam = params[2], mu = params[3];
    const double R = params[9], S = params[10];
    const double c1 = params[11], c2 = params[12];
    const double c3 = params[13], c4 = params[14], c5 = params[15];
    const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
    const double corr = 1.0 + c1 * r + c2 * r * r + c3 * r * r * r +
                        c4 * r * r * r * r + c5 * r * r * r * r * r;
    return detail::fc(r, R, S) * pair * corr;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const double A = params[0], B = params[1];
    const double lam = params[2], mu = params[3];
    const double R = params[9], S = params[10];
    const double c1 = params[11], c2 = params[12];
    const double c3 = params[13], c4 = params[14], c5 = params[15];
    const double f = detail::fc(r, R, S);
    const double df = detail::dfc(r, R, S);
    const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
    const double dpair =
        -lam * A * std::exp(-lam * r) + mu * B * std::exp(-mu * r);
    const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
    const double corr = 1.0 + c1 * r + c2 * r2 + c3 * r3 + c4 * r4 + c5 * r5;
    const double dcorr =
        c1 + 2.0 * c2 * r + 3.0 * c3 * r2 + 4.0 * c4 * r3 + 5.0 * c5 * r4;
    return (df * pair + f * dpair) * corr + f * pair * dcorr;
  }
};

// ── Kawamura ionic potential, matches forcesmith kawamura_value:
//   V = p0·p1/r + p2·(p5+p6)·exp((p3+p4−r)/(p5+p6)) − p7·p8/r^6
// params follow forcesmith p[] (charges/sums multiply): {p0..p8}
struct Kawamura : AnalyticParams<Kawamura, 9> {
  constexpr Kawamura(double p0, double p1, double p2, double p3, double p4,
                     double p5, double p6, double p7, double p8, double lo,
                     double hi)
      : AnalyticParams({p0, p1, p2, p3, p4, p5, p6, p7, p8}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [p0, p1, p2, p3, p4, p5, p6, p7, p8] = params;
    const double s = p5 + p6, t = p3 + p4;
    const double r6 = std::pow(r, 6);
    return p0 * p1 / r + p2 * s * std::exp((t - r) / s) - p7 * p8 / r6;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [p0, p1, p2, p3, p4, p5, p6, p7, p8] = params;
    const double s = p5 + p6, t = p3 + p4;
    return -p0 * p1 / (r * r) - p2 * std::exp((t - r) / s) +
           6.0 * p7 * p8 / std::pow(r, 7);
  }
};

// ── KawamuraMix, matches forcesmith kawamura_mix_value:
//   V = kawamura(p0..p8) + p2·p9·(exp(−2·p10·(r−p11)) − 2·exp(−p10·(r−p11)))
// params follow forcesmith p[]: {p0..p11}
struct KawamuraMix : AnalyticParams<KawamuraMix, 12> {
  constexpr KawamuraMix(std::array<double, 12> p, double lo, double hi)
      : AnalyticParams(p, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const double p0 = params[0], p1 = params[1], p2 = params[2], p3 = params[3];
    const double p4 = params[4], p5 = params[5], p6 = params[6], p7 = params[7],
                 p8 = params[8];
    const double p9 = params[9], p10 = params[10], p11 = params[11];
    const double s = p5 + p6, t = p3 + p4, r6 = std::pow(r, 6), w = r - p11;
    return p0 * p1 / r + p2 * s * std::exp((t - r) / s) - p7 * p8 / r6 +
           p2 * p9 * (std::exp(-2.0 * p10 * w) - 2.0 * std::exp(-p10 * w));
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const double p0 = params[0], p1 = params[1], p2 = params[2], p3 = params[3];
    const double p4 = params[4], p5 = params[5], p6 = params[6], p7 = params[7],
                 p8 = params[8];
    const double p9 = params[9], p10 = params[10], p11 = params[11];
    const double s = p5 + p6, t = p3 + p4, w = r - p11;
    return -p0 * p1 / (r * r) - p2 * std::exp((t - r) / s) +
           6.0 * p7 * p8 / std::pow(r, 7) +
           2.0 * p2 * p9 * p10 *
               (std::exp(-p10 * w) - std::exp(-2.0 * p10 * w));
  }
};

// ── Softshell: V = (A/r)^n (soft-core repulsion) ─────────────────────────────
// Matches forcesmith softshell_value: the whole ratio A/r is raised to n.
// params: {A, n}
struct Softshell : AnalyticParams<Softshell, 2> {
  constexpr Softshell(double A, double n, double lo, double hi)
      : AnalyticParams({A, n}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, n] = params;
    return std::pow(A / r, n);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, n] = params;
    return -n * std::pow(A / r, n) / r; // d/dr (A/r)^n = -(n/r)(A/r)^n
  }
};

// ── ExpPlus: V = A exp(−Br) + C ──────────────────────────────────────────────
// params: {A, B, C}
struct ExpPlus : AnalyticParams<ExpPlus, 3> {
  constexpr ExpPlus(double A, double B, double C, double lo, double hi)
      : AnalyticParams({A, B, C}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, C] = params;
    return A * std::exp(-B * r) + C;
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, C] = params;
    return -A * B * std::exp(-B * r);
  }
};

// ── Strmm (Streitz-Mintmire), matches forcesmith strmm_value:
//   s = r − r0;  V = 2A·exp(−B/2·s) − C·(1 + D·s)·exp(−D·s)
// params order follows forcesmith p[]: {A, B, C, D, r0}
struct Strmm : AnalyticParams<Strmm, 5> {
  constexpr Strmm(double A, double B, double C, double D, double r0, double lo,
                  double hi)
      : AnalyticParams({A, B, C, D, r0}, lo, hi) {}
  constexpr FORCE_INLINE double eval(double r) const {
    const auto [A, B, C, D, r0] = params;
    const double s = r - r0;
    return 2.0 * A * std::exp(-B / 2.0 * s) -
           C * (1.0 + D * s) * std::exp(-D * s);
  }
  constexpr FORCE_INLINE double deriv(double r) const {
    const auto [A, B, C, D, r0] = params;
    const double s = r - r0;
    return -A * B * std::exp(-B / 2.0 * s) + C * D * D * s * std::exp(-D * s);
  }
};

} // namespace forcesmith
