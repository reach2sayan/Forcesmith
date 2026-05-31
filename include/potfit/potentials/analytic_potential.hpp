#pragma once

#include "potfit/core/param.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <utility>

namespace potfit {

// CRTP base for analytic potentials. Each Derived supplies an eval_impl(double)
// and deriv_impl(double); their bodies live in analytic_potential.cpp.
template <typename Derived, std::size_t N> struct AnalyticBase {
  std::array<Param, N> params;
  double rmin, rmax;

  double eval(double r) const {
    return static_cast<const Derived &>(*this).eval_impl(r);
  }
  double deriv(double r) const {
    return static_cast<const Derived &>(*this).deriv_impl(r);
  }
  constexpr std::pair<double, double> span() const { return {rmin, rmax}; }

  // Mark parameter i as fixed (excluded from optimizer) or free.
  constexpr void set_fixed(std::size_t i, bool f) { params[i].fixed = f; }
  constexpr bool is_fixed(std::size_t i) const { return params[i].fixed; }

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

protected:
  AnalyticBase(std::array<double, N> vals, double lo, double hi) noexcept
      : rmin(lo), rmax(hi) {
    std::ranges::transform(vals, params.begin(),
                           [](auto v) { return Param{v}; });
  }
};

// Each concrete potential below declares eval_impl/deriv_impl; the formulas are
// defined out-of-line in analytic_potential.cpp.
#define POTFIT_ANALYTIC_METHODS                                                \
  double eval_impl(double r) const;                                            \
  double deriv_impl(double r) const;

// ── Lennard-Jones: V(r) = 4ε[(σ/r)^12 − (σ/r)^6] ───────────────────────────
// params: {epsilon, sigma}
struct LennardJones : AnalyticBase<LennardJones, 2> {
  LennardJones(double epsilon, double sigma, double lo, double hi)
      : AnalyticBase({epsilon, sigma}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Morse: V(r) = D_e[(1−e^{−a(r−r_e)})^2 − 1] ─────────────────────────────
// params: {D_e, a, r_e}
struct Morse : AnalyticBase<Morse, 3> {
  Morse(double De, double a, double re, double lo, double hi)
      : AnalyticBase({De, a, re}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Buckingham: V = A exp(−r/ρ) − C·ρ^6/r^6 ─────────────────────────────────
// Matches potfit buck_value: the dispersion term carries a ρ^6 factor.
// params: {A, rho, C}
struct Buckingham : AnalyticBase<Buckingham, 3> {
  Buckingham(double A, double rho, double C, double lo, double hi)
      : AnalyticBase({A, rho, C}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Born: V = A exp((C−r)/B) − D/r^6 + E/r^8 ────────────────────────────────
// Matches potfit born_value. params: {A, B, C, D, E}
//   A: amplitude, B: range, C: offset inside exponent, D: r^6, E: r^8.
struct Born : AnalyticBase<Born, 5> {
  Born(double A, double B, double C, double D, double E, double lo, double hi)
      : AnalyticBase({A, B, C, D, E}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── PowerDecay: V = A/r^n ────────────────────────────────────────────────────
// params: {A, n}
struct PowerDecay : AnalyticBase<PowerDecay, 2> {
  PowerDecay(double A, double n, double lo, double hi)
      : AnalyticBase({A, n}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── ExpDecay: V = A exp(−Br) ─────────────────────────────────────────────────
// params: {A, B}
struct ExpDecay : AnalyticBase<ExpDecay, 2> {
  ExpDecay(double A, double B, double lo, double hi)
      : AnalyticBase({A, B}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── MexpDecay, matches potfit mexp_decay_value: V = A·exp(−B·(r − r0)) ───────
// params order follows potfit p[]: {A, B, r0}
struct MexpDecay : AnalyticBase<MexpDecay, 3> {
  MexpDecay(double A, double B, double r0, double lo, double hi)
      : AnalyticBase({A, B, r0}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Harmonic: V = k(r−r0)² ───────────────────────────────────────────────────
// params: {k, r0}
struct Harmonic : AnalyticBase<Harmonic, 2> {
  Harmonic(double k, double r0, double lo, double hi)
      : AnalyticBase({k, r0}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Universal embedding function, matches potfit universal_value:
//   V = E0·(b/(b−a)·r^a − a/(b−a)·r^b) + c·r
// params order follows potfit p[]: {E0, a, b, c}
struct Universal : AnalyticBase<Universal, 4> {
  Universal(double E0, double a, double b, double c, double lo, double hi)
      : AnalyticBase({E0, a, b, c}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Eopp (empirical oscillating pair), matches potfit eopp_value:
//   V = A/r^n + (B/r^m)·cos(k·r + φ)
// params order follows potfit p[]: {A, n, B, m, k, phi}
struct Eopp : AnalyticBase<Eopp, 6> {
  Eopp(double A, double n, double B, double m, double k, double phi, double lo,
       double hi)
      : AnalyticBase({A, n, B, m, k, phi}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── EoppExp, matches potfit eopp_exp_value:
//   V = A·exp(−B·r) + (C/r^m)·cos(k·r + φ)
// params order follows potfit p[]: {A, B, C, m, k, phi}
struct EoppExp : AnalyticBase<EoppExp, 6> {
  EoppExp(double A, double B, double C, double m, double k, double phi,
          double lo, double hi)
      : AnalyticBase({A, B, C, m, k, phi}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Meopp (modified eopp), matches potfit meopp_value:
//   V = A/(r−r0)^n + (B/r^m)·cos(k·r + φ)
// params order follows potfit p[]: {A, n, B, m, k, phi, r0}
struct Meopp : AnalyticBase<Meopp, 7> {
  Meopp(double A, double n, double B, double m, double k, double phi, double r0,
        double lo, double hi)
      : AnalyticBase({A, n, B, m, k, phi, r0}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── GenLJ (generalized Lennard-Jones), matches potfit gen_lj_value:
//   x = r/r0;  V = A/(m−n)·(m·x^{−n} − n·x^{−m}) + B
// params order follows potfit p[]: {A, n, m, r0, B}
struct GenLJ : AnalyticBase<GenLJ, 5> {
  GenLJ(double A, double n, double m, double r0, double B, double lo, double hi)
      : AnalyticBase({A, n, m, r0, B}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── DoubleMorse: sum of two Morse terms + constant offset ────────────────────
// params: {D1, a1, r1, D2, a2, r2, C}
struct DoubleMorse : AnalyticBase<DoubleMorse, 7> {
  DoubleMorse(double D1, double a1, double r1, double D2, double a2, double r2,
              double C, double lo, double hi)
      : AnalyticBase({D1, a1, r1, D2, a2, r2, C}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── DoubleExp, matches potfit double_exp_value:
//   V = A·exp(−B·(r − r1)²) + exp(−C·(r − r2))   (2nd term has no prefactor)
// params order follows potfit p[]: {A, B, r1, C, r2}
struct DoubleExp : AnalyticBase<DoubleExp, 5> {
  DoubleExp(double A, double B, double r1, double C, double r2, double lo,
            double hi)
      : AnalyticBase({A, B, r1, C, r2}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Mishin, matches potfit mishin_value:
//   z = r − r0;  e = exp(−d·z);  V = A·z^n·e·(1 + B·e) + C
// params order follows potfit p[]: {A, B, C, r0, n, d}
struct Mishin : AnalyticBase<Mishin, 6> {
  Mishin(double A, double B, double C, double r0, double n, double d, double lo,
         double hi)
      : AnalyticBase({A, B, C, r0, n, d}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── SqrtFunc: V = A sqrt(r / B) ──────────────────────────────────────────────
// Matches potfit sqrt_value. params: {A, B}
struct SqrtFunc : AnalyticBase<SqrtFunc, 2> {
  SqrtFunc(double A, double B, double lo, double hi)
      : AnalyticBase({A, B}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── ConstFunc: V = C ─────────────────────────────────────────────────────────
// params: {C}
struct ConstFunc : AnalyticBase<ConstFunc, 1> {
  explicit ConstFunc(double C, double lo, double hi)
      : AnalyticBase({C}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Parabola: V = Ar² + Br + C ───────────────────────────────────────────────
// params: {A, B, C}
struct Parabola : AnalyticBase<Parabola, 3> {
  Parabola(double A, double B, double C, double lo, double hi)
      : AnalyticBase({A, B, C}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Poly5, matches potfit poly_5_value (expansion about r = 1):
//   s = r − 1;  V = a0 + 0.5·a1·s² + a2·s³ + a3·s⁴ + a4·s⁵
// params: {a0, a1, a2, a3, a4}
struct Poly5 : AnalyticBase<Poly5, 5> {
  Poly5(double a0, double a1, double a2, double a3, double a4, double lo,
        double hi)
      : AnalyticBase({a0, a1, a2, a3, a4}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── StiwWeb2 (Stillinger-Weber pair), matches potfit stiweb_2_value:
//   V = (A·r^{−p} − B·r^{−q})·exp(δ/(r − rc))
// params order follows potfit p[]: {A, B, p, q, delta, rc}
struct StiwWeb2 : AnalyticBase<StiwWeb2, 6> {
  StiwWeb2(double A, double B, double p, double q, double delta, double rc,
           double lo, double hi)
      : AnalyticBase({A, B, p, q, delta, rc}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── StiwWeb3: h(r) = exp(γ/(r−a)),  r < a  [SW 3-body radial function] ───────
// params: {gamma, a}   (γ = γ_SW × σ and a = a_SW × σ, pre-multiplied)
struct StiwWeb3 : AnalyticBase<StiwWeb3, 2> {
  StiwWeb3(double gamma, double a, double lo, double hi)
      : AnalyticBase({gamma, a}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── TersoffPot: V = fc(r)[A exp(−λr) − B exp(−μr)] ──────────────────────────
// Bond-order params β, n, c, d, h are stored but not used in pure pair eval.
// params: {A, B, lambda, mu, beta, n, c, d, h, R, S}
struct TersoffPot : AnalyticBase<TersoffPot, 11> {
  TersoffPot(double A, double B, double lam, double mu, double beta, double n,
             double c, double d, double h, double R, double S, double lo,
             double hi)
      : AnalyticBase({A, B, lam, mu, beta, n, c, d, h, R, S}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── TersoffMix: mixing correction V = χ exp(−ω r) ────────────────────────────
// params: {chi, omega}
struct TersoffMix : AnalyticBase<TersoffMix, 2> {
  TersoffMix(double chi, double omega, double lo, double hi)
      : AnalyticBase({chi, omega}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── TersoffModPot: modified Tersoff pair (16 params) ─────────────────────────
// Extended pair: fc(r)[A exp(−λr) − B exp(−μr)] × polynomial correction.
// Extra params c1..c5 (indices 11-15) provide a polynomial correction.
// params: {A, B, lambda, mu, beta, n, c, d, h, R, S, c1, c2, c3, c4, c5}
struct TersoffModPot : AnalyticBase<TersoffModPot, 16> {
  TersoffModPot(std::array<double, 16> p, double lo, double hi)
      : AnalyticBase(p, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Kawamura ionic potential, matches potfit kawamura_value:
//   V = p0·p1/r + p2·(p5+p6)·exp((p3+p4−r)/(p5+p6)) − p7·p8/r^6
// params follow potfit p[] (charges/sums multiply): {p0..p8}
struct Kawamura : AnalyticBase<Kawamura, 9> {
  Kawamura(double p0, double p1, double p2, double p3, double p4, double p5,
           double p6, double p7, double p8, double lo, double hi)
      : AnalyticBase({p0, p1, p2, p3, p4, p5, p6, p7, p8}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── KawamuraMix, matches potfit kawamura_mix_value:
//   V = kawamura(p0..p8) + p2·p9·(exp(−2·p10·(r−p11)) − 2·exp(−p10·(r−p11)))
// params follow potfit p[]: {p0..p11}
struct KawamuraMix : AnalyticBase<KawamuraMix, 12> {
  KawamuraMix(std::array<double, 12> p, double lo, double hi)
      : AnalyticBase(p, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Softshell: V = (A/r)^n (soft-core repulsion) ─────────────────────────────
// Matches potfit softshell_value: the whole ratio A/r is raised to n.
// params: {A, n}
struct Softshell : AnalyticBase<Softshell, 2> {
  Softshell(double A, double n, double lo, double hi)
      : AnalyticBase({A, n}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── ExpPlus: V = A exp(−Br) + C ──────────────────────────────────────────────
// params: {A, B, C}
struct ExpPlus : AnalyticBase<ExpPlus, 3> {
  ExpPlus(double A, double B, double C, double lo, double hi)
      : AnalyticBase({A, B, C}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

// ── Strmm (Streitz-Mintmire), matches potfit strmm_value:
//   s = r − r0;  V = 2A·exp(−B/2·s) − C·(1 + D·s)·exp(−D·s)
// params order follows potfit p[]: {A, B, C, D, r0}
struct Strmm : AnalyticBase<Strmm, 5> {
  Strmm(double A, double B, double C, double D, double r0, double lo, double hi)
      : AnalyticBase({A, B, C, D, r0}, lo, hi) {}
  POTFIT_ANALYTIC_METHODS
};

#undef POTFIT_ANALYTIC_METHODS

} // namespace potfit
