#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>

namespace potfit {

// ── CRTP base ─────────────────────────────────────────────────────────────────
// Derived must implement eval_impl(double r)->double and
// deriv_impl(double r)->double.

template <typename Derived, std::size_t N> struct AnalyticBase {
    std::array<double, N> params;
    double rmin, rmax;

    constexpr double eval(double r)  const { return static_cast<const Derived&>(*this).eval_impl(r);  }
    constexpr double deriv(double r) const { return static_cast<const Derived&>(*this).deriv_impl(r); }
    constexpr std::pair<double,double> span()        const { return {rmin, rmax}; }
    constexpr int    param_count()                   const { return static_cast<int>(N); }
    constexpr void   gather_params(Eigen::VectorXd& dst, int off) const {
        std::ranges::copy(params, dst.data() + off);
    }
    constexpr void   scatter_params(const Eigen::VectorXd& src, int off) {
        std::copy_n(src.data() + off, N, params.begin());
    }

protected:
    AnalyticBase(std::array<double, N> p, double lo, double hi) noexcept
        : params(p), rmin(lo), rmax(hi) {}
};

// ── Internal helpers ──────────────────────────────────────────────────────────
namespace detail {

// Tersoff / SW smooth cosine cutoff and its derivative.
inline double fc(double r, double R, double S) noexcept {
    if (r <= R) return 1.0;
    if (r >= S) return 0.0;
    const double x = std::numbers::pi * (r - R) / (S - R);
    return 0.5 + 0.5 * std::cos(x);
}
inline double dfc(double r, double R, double S) noexcept {
    if (r <= R || r >= S) return 0.0;
    const double x = std::numbers::pi * (r - R) / (S - R);
    return -0.5 * std::numbers::pi / (S - R) * std::sin(x);
}

} // namespace detail

// ═════════════════════════════════════════════════════════════════════════════
// Pair potentials (done in steps 1-2)
// ═════════════════════════════════════════════════════════════════════════════

// ── Lennard-Jones: V(r) = 4ε[(σ/r)^12 − (σ/r)^6] ───────────────────────────
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

// ── Morse: V(r) = D_e[(1−e^{−a(r−r_e)})^2 − 1] ─────────────────────────────
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

// ═════════════════════════════════════════════════════════════════════════════
// Step 20 — new analytic function types
// ═════════════════════════════════════════════════════════════════════════════

// ── Buckingham: V = A exp(−r/ρ) − C/r^6 ─────────────────────────────────────
// params: {A, rho, C}
struct Buckingham : AnalyticBase<Buckingham, 3> {
    Buckingham(double A, double rho, double C, double lo, double hi)
        : AnalyticBase({A, rho, C}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, rho, C] = params;
        return A * std::exp(-r / rho) - C / (r * r * r * r * r * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, rho, C] = params;
        return -(A / rho) * std::exp(-r / rho) + 6.0 * C / std::pow(r, 7);
    }
};

// ── Born: V = A exp(−Br) − C/r^6 − D/r^8 − E/r^10 ──────────────────────────
// params: {A, B, C, D, E}
struct Born : AnalyticBase<Born, 5> {
    Born(double A, double B, double C, double D, double E, double lo, double hi)
        : AnalyticBase({A, B, C, D, E}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E] = params;
        return A * std::exp(-B * r) - C / std::pow(r, 6)
                                    - D / std::pow(r, 8)
                                    - E / std::pow(r, 10);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E] = params;
        return -A * B * std::exp(-B * r) + 6.0 * C / std::pow(r, 7)
                                         + 8.0 * D / std::pow(r, 9)
                                         + 10.0 * E / std::pow(r, 11);
    }
};

// ── PowerDecay: V = A/r^n ─────────────────────────────────────────────────────
// params: {A, n}
struct PowerDecay : AnalyticBase<PowerDecay, 2> {
    PowerDecay(double A, double n, double lo, double hi)
        : AnalyticBase({A, n}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, n] = params;
        return A / std::pow(r, n);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, n] = params;
        return -A * n / std::pow(r, n + 1.0);
    }
};

// ── ExpDecay: V = A exp(−Br) ─────────────────────────────────────────────────
// params: {A, B}
struct ExpDecay : AnalyticBase<ExpDecay, 2> {
    ExpDecay(double A, double B, double lo, double hi)
        : AnalyticBase({A, B}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B] = params;
        return A * std::exp(-B * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B] = params;
        return -A * B * std::exp(-B * r);
    }
};

// ── MexpDecay: V = A r^n exp(−Br) ────────────────────────────────────────────
// params: {A, n, B}
struct MexpDecay : AnalyticBase<MexpDecay, 3> {
    MexpDecay(double A, double n, double B, double lo, double hi)
        : AnalyticBase({A, n, B}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, n, B] = params;
        return A * std::pow(r, n) * std::exp(-B * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, n, B] = params;
        return A * std::pow(r, n - 1.0) * (n - B * r) * std::exp(-B * r);
    }
};

// ── Harmonic: V = k(r−r0)² ───────────────────────────────────────────────────
// params: {k, r0}
struct Harmonic : AnalyticBase<Harmonic, 2> {
    Harmonic(double k, double r0, double lo, double hi)
        : AnalyticBase({k, r0}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [k, r0] = params;
        return k * (r - r0) * (r - r0);
    }
    constexpr double deriv_impl(double r) const {
        const auto [k, r0] = params;
        return 2.0 * k * (r - r0);
    }
};

// ── Universal (UBER-like): V = E0(1 + A·x)exp(−B·x), x = r−r0 ───────────────
// params: {E0, r0, A, B}
struct Universal : AnalyticBase<Universal, 4> {
    Universal(double E0, double r0, double A, double B, double lo, double hi)
        : AnalyticBase({E0, r0, A, B}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [E0, r0, A, B] = params;
        const double x = r - r0;
        return E0 * (1.0 + A * x) * std::exp(-B * x);
    }
    constexpr double deriv_impl(double r) const {
        const auto [E0, r0, A, B] = params;
        const double x  = r - r0;
        const double ex = std::exp(-B * x);
        return E0 * ex * (A - B * (1.0 + A * x));
    }
};

// ── Eopp (extended oscillatory pair): V = A/r^n + B exp(−Cr) + D/r^m ─────────
// params: {A, n, B, C, D, m}
struct Eopp : AnalyticBase<Eopp, 6> {
    Eopp(double A, double n, double B, double C, double D, double m,
         double lo, double hi)
        : AnalyticBase({A, n, B, C, D, m}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, n, B, C, D, m] = params;
        return A / std::pow(r, n) + B * std::exp(-C * r) + D / std::pow(r, m);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, n, B, C, D, m] = params;
        return -A * n / std::pow(r, n + 1.0)
               - B * C * std::exp(-C * r)
               - D * m / std::pow(r, m + 1.0);
    }
};

// ── EoppExp: V = A exp(−Br) + C r exp(−Dr) + E r² exp(−Fr) ──────────────────
// params: {A, B, C, D, E, F}
struct EoppExp : AnalyticBase<EoppExp, 6> {
    EoppExp(double A, double B, double C, double D, double E, double F,
            double lo, double hi)
        : AnalyticBase({A, B, C, D, E, F}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E, F] = params;
        return A * std::exp(-B * r)
             + C * r * std::exp(-D * r)
             + E * r * r * std::exp(-F * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E, F] = params;
        return -A * B * std::exp(-B * r)
             + C * (1.0 - D * r) * std::exp(-D * r)
             + E * (2.0 * r - F * r * r) * std::exp(-F * r);
    }
};

// ── Meopp: V = A exp(−Br)/r + C exp(−Dr)/r² + E cos(Fr+G)/r³ ────────────────
// params: {A, B, C, D, E, F, G}
struct Meopp : AnalyticBase<Meopp, 7> {
    Meopp(double A, double B, double C, double D, double E, double F, double G,
          double lo, double hi)
        : AnalyticBase({A, B, C, D, E, F, G}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E, F, G] = params;
        return A * std::exp(-B * r) / r
             + C * std::exp(-D * r) / (r * r)
             + E * std::cos(F * r + G) / (r * r * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E, F, G] = params;
        const double r2 = r * r, r3 = r2 * r, r4 = r3 * r;
        return A * std::exp(-B * r) * (-B / r - 1.0 / r2)
             + C * std::exp(-D * r) * (-D / r2 - 2.0 / r3)
             + E * (-F * std::sin(F * r + G) / r3 - 3.0 * std::cos(F * r + G) / r4);
    }
};

// ── GenLJ: V = A[(r0/r)^n − (r0/r)^m] + B ───────────────────────────────────
// params: {A, r0, n, m, B}
struct GenLJ : AnalyticBase<GenLJ, 5> {
    GenLJ(double A, double r0, double n, double m, double B, double lo, double hi)
        : AnalyticBase({A, r0, n, m, B}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, r0, n, m, B] = params;
        return A * (std::pow(r0 / r, n) - std::pow(r0 / r, m)) + B;
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, r0, n, m, B] = params;
        return (A / r) * (-n * std::pow(r0 / r, n) + m * std::pow(r0 / r, m));
    }
};

// ── DoubleMorse: sum of two Morse terms + constant offset ────────────────────
// params: {D1, a1, r1, D2, a2, r2, C}
struct DoubleMorse : AnalyticBase<DoubleMorse, 7> {
    DoubleMorse(double D1, double a1, double r1,
                double D2, double a2, double r2,
                double C, double lo, double hi)
        : AnalyticBase({D1, a1, r1, D2, a2, r2, C}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [D1, a1, r1, D2, a2, r2, C] = params;
        const double e1 = std::exp(-a1 * (r - r1));
        const double e2 = std::exp(-a2 * (r - r2));
        return D1 * ((1.0 - e1) * (1.0 - e1) - 1.0)
             + D2 * ((1.0 - e2) * (1.0 - e2) - 1.0) + C;
    }
    constexpr double deriv_impl(double r) const {
        const auto [D1, a1, r1, D2, a2, r2, C] = params;
        const double e1 = std::exp(-a1 * (r - r1));
        const double e2 = std::exp(-a2 * (r - r2));
        return 2.0 * D1 * a1 * e1 * (1.0 - e1)
             + 2.0 * D2 * a2 * e2 * (1.0 - e2);
    }
};

// ── DoubleExp: V = A exp(−Br) + C exp(−Dr) + E ───────────────────────────────
// params: {A, B, C, D, E}
struct DoubleExp : AnalyticBase<DoubleExp, 5> {
    DoubleExp(double A, double B, double C, double D, double E,
              double lo, double hi)
        : AnalyticBase({A, B, C, D, E}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E] = params;
        return A * std::exp(-B * r) + C * std::exp(-D * r) + E;
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E] = params;
        return -A * B * std::exp(-B * r) - C * D * std::exp(-D * r);
    }
};

// ── Mishin: V = (A + Br) exp(−Cr) + (D + Er) exp(−Fr) ───────────────────────
// params: {A, B, C, D, E, F}
struct Mishin : AnalyticBase<Mishin, 6> {
    Mishin(double A, double B, double C, double D, double E, double F,
           double lo, double hi)
        : AnalyticBase({A, B, C, D, E, F}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E, F] = params;
        return (A + B * r) * std::exp(-C * r) + (D + E * r) * std::exp(-F * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E, F] = params;
        return std::exp(-C * r) * (B - C * (A + B * r))
             + std::exp(-F * r) * (E - F * (D + E * r));
    }
};

// ── SqrtFunc: V = A sqrt(r + B) ──────────────────────────────────────────────
// params: {A, B}   (B shifts the argument; B >= 0 keeps domain valid for r>0)
struct SqrtFunc : AnalyticBase<SqrtFunc, 2> {
    SqrtFunc(double A, double B, double lo, double hi)
        : AnalyticBase({A, B}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B] = params;
        return A * std::sqrt(r + B);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B] = params;
        return A / (2.0 * std::sqrt(r + B));
    }
};

// ── ConstFunc: V = C ──────────────────────────────────────────────────────────
// params: {C}
struct ConstFunc : AnalyticBase<ConstFunc, 1> {
    explicit ConstFunc(double C, double lo, double hi)
        : AnalyticBase({C}, lo, hi) {}

    constexpr double eval_impl(double) const { return params[0]; }
    constexpr double deriv_impl(double) const { return 0.0; }
};

// ── Parabola: V = Ar² + Br + C ───────────────────────────────────────────────
// params: {A, B, C}
struct Parabola : AnalyticBase<Parabola, 3> {
    Parabola(double A, double B, double C, double lo, double hi)
        : AnalyticBase({A, B, C}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C] = params;
        return A * r * r + B * r + C;
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C] = params;
        return 2.0 * A * r + B;
    }
};

// ── Poly5: V = a0 + a1 r + a2 r² + a3 r³ + a4 r⁴ ───────────────────────────
// params: {a0, a1, a2, a3, a4}
struct Poly5 : AnalyticBase<Poly5, 5> {
    Poly5(double a0, double a1, double a2, double a3, double a4,
          double lo, double hi)
        : AnalyticBase({a0, a1, a2, a3, a4}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [a0, a1, a2, a3, a4] = params;
        return a0 + r * (a1 + r * (a2 + r * (a3 + r * a4)));
    }
    constexpr double deriv_impl(double r) const {
        const auto [a0, a1, a2, a3, a4] = params;
        return a1 + r * (2.0 * a2 + r * (3.0 * a3 + r * 4.0 * a4));
    }
};

// ── StiwWeb2: V = A[B(σ/r)^p − (σ/r)^q] exp(σ/(r−aσ)),  r < aσ ─────────────
// params: {A, B, p, q, a, sigma}
struct StiwWeb2 : AnalyticBase<StiwWeb2, 6> {
    StiwWeb2(double A, double B, double p, double q, double a, double sigma,
             double lo, double hi)
        : AnalyticBase({A, B, p, q, a, sigma}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, p, q, a, sigma] = params;
        const double rcut = a * sigma;
        if (r >= rcut) return 0.0;
        const double d    = r - rcut;
        const double e    = std::exp(sigma / d);
        if (e == 0.0) return 0.0;
        const double sr   = sigma / r;
        return A * (B * std::pow(sr, p) - std::pow(sr, q)) * e;
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, p, q, a, sigma] = params;
        const double rcut = a * sigma;
        if (r >= rcut) return 0.0;
        const double d    = r - rcut;
        const double e    = std::exp(sigma / d);
        if (e == 0.0) return 0.0;
        const double sr   = sigma / r;
        const double srp  = std::pow(sr, p), srq = std::pow(sr, q);
        const double phi  = B * srp - srq;
        const double dphi = (-p * B * srp + q * srq) / r;
        return A * (dphi * e + phi * e * (-sigma / (d * d)));
    }
};

// ── StiwWeb3: h(r) = exp(γ/(r−a)),  r < a  [SW 3-body radial function] ───────
// params: {gamma, a}   (γ = γ_SW × σ and a = a_SW × σ, pre-multiplied)
struct StiwWeb3 : AnalyticBase<StiwWeb3, 2> {
    StiwWeb3(double gamma, double a, double lo, double hi)
        : AnalyticBase({gamma, a}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [gamma, a] = params;
        if (r >= a) return 0.0;
        const double h = std::exp(gamma / (r - a));
        return h;
    }
    constexpr double deriv_impl(double r) const {
        const auto [gamma, a] = params;
        if (r >= a) return 0.0;
        const double d  = r - a;
        const double h  = std::exp(gamma / d);
        if (h == 0.0) return 0.0;
        return h * (-gamma / (d * d));
    }
};

// ── TersoffPot: V = fc(r)[A exp(−λr) − B exp(−μr)] ──────────────────────────
// Bond-order params β, n, c, d, h are stored but not used in pure pair eval.
// params: {A, B, lambda, mu, beta, n, c, d, h, R, S}
struct TersoffPot : AnalyticBase<TersoffPot, 11> {
    TersoffPot(double A, double B, double lam, double mu,
               double beta, double n, double c, double d, double h,
               double R, double S, double lo, double hi)
        : AnalyticBase({A, B, lam, mu, beta, n, c, d, h, R, S}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const double A = params[0], B = params[1], lam = params[2], mu = params[3];
        const double R = params[9], S = params[10];
        return detail::fc(r, R, S) * (A * std::exp(-lam * r) - B * std::exp(-mu * r));
    }
    constexpr double deriv_impl(double r) const {
        const double A = params[0], B = params[1], lam = params[2], mu = params[3];
        const double R = params[9], S = params[10];
        const double f    = detail::fc(r, R, S);
        const double df   = detail::dfc(r, R, S);
        const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
        const double dpair = -lam * A * std::exp(-lam * r) + mu * B * std::exp(-mu * r);
        return df * pair + f * dpair;
    }
};

// ── TersoffMix: mixing correction V = χ exp(−ω r) ────────────────────────────
// params: {chi, omega}
struct TersoffMix : AnalyticBase<TersoffMix, 2> {
    TersoffMix(double chi, double omega, double lo, double hi)
        : AnalyticBase({chi, omega}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [chi, omega] = params;
        return chi * std::exp(-omega * r);
    }
    constexpr double deriv_impl(double r) const {
        const auto [chi, omega] = params;
        return -chi * omega * std::exp(-omega * r);
    }
};

// ── TersoffModPot: modified Tersoff pair (16 params) ─────────────────────────
// Extended pair: fc(r)[A exp(−λr) − B exp(−μr)] × polynomial correction.
// Extra params c1..c5 (indices 11-15) provide a polynomial correction.
// params: {A, B, lambda, mu, beta, n, c, d, h, R, S, c1, c2, c3, c4, c5}
struct TersoffModPot : AnalyticBase<TersoffModPot, 16> {
    TersoffModPot(std::array<double,16> p, double lo, double hi)
        : AnalyticBase(p, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const double A   = params[0], B   = params[1];
        const double lam = params[2], mu  = params[3];
        const double R   = params[9], S   = params[10];
        const double c1  = params[11], c2 = params[12];
        const double c3  = params[13], c4 = params[14], c5 = params[15];
        const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
        const double corr = 1.0 + c1 * r + c2 * r*r + c3 * r*r*r + c4 * r*r*r*r + c5 * r*r*r*r*r;
        return detail::fc(r, R, S) * pair * corr;
    }
    constexpr double deriv_impl(double r) const {
        const double A   = params[0], B   = params[1];
        const double lam = params[2], mu  = params[3];
        const double R   = params[9], S   = params[10];
        const double c1  = params[11], c2 = params[12];
        const double c3  = params[13], c4 = params[14], c5 = params[15];
        const double f    = detail::fc(r, R, S);
        const double df   = detail::dfc(r, R, S);
        const double pair  = A * std::exp(-lam * r) - B * std::exp(-mu * r);
        const double dpair = -lam * A * std::exp(-lam * r) + mu * B * std::exp(-mu * r);
        const double r2 = r*r, r3 = r2*r, r4 = r3*r, r5 = r4*r;
        const double corr  = 1.0 + c1*r + c2*r2 + c3*r3 + c4*r4 + c5*r5;
        const double dcorr = c1 + 2.0*c2*r + 3.0*c3*r2 + 4.0*c4*r3 + 5.0*c5*r4;
        return (df * pair + f * dpair) * corr + f * pair * dcorr;
    }
};

// ── Kawamura: ionic potential ─────────────────────────────────────────────────
// V = A/r + B exp((C−r)/D) − E/r^6 − F/r^8 + G exp(−Hr) − I/r^10
// params: {A, B, C, D, E, F, G, H, I}
struct Kawamura : AnalyticBase<Kawamura, 9> {
    Kawamura(double A, double B, double C, double D,
             double E, double F, double G, double H, double I,
             double lo, double hi)
        : AnalyticBase({A, B, C, D, E, F, G, H, I}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E, F, G, H, I] = params;
        return A / r + B * std::exp((C - r) / D)
             - E / std::pow(r, 6) - F / std::pow(r, 8)
             + G * std::exp(-H * r) - I / std::pow(r, 10);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E, F, G, H, I] = params;
        return -A / (r * r) - (B / D) * std::exp((C - r) / D)
             + 6.0 * E / std::pow(r, 7) + 8.0 * F / std::pow(r, 9)
             - G * H * std::exp(-H * r) + 10.0 * I / std::pow(r, 11);
    }
};

// ── KawamuraMix: extended Kawamura (12 params) ────────────────────────────────
// V = Kawamura(9) + J/r + K exp(−Lr)
// params: {A, B, C, D, E, F, G, H, I, J, K, L}
struct KawamuraMix : AnalyticBase<KawamuraMix, 12> {
    KawamuraMix(std::array<double,12> p, double lo, double hi)
        : AnalyticBase(p, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const double A=params[0], B=params[1], C=params[2], D=params[3];
        const double E=params[4], F=params[5], G=params[6], H=params[7], I=params[8];
        const double J=params[9], K=params[10], L=params[11];
        return A/r + B*std::exp((C-r)/D)
             - E/std::pow(r,6) - F/std::pow(r,8)
             + G*std::exp(-H*r) - I/std::pow(r,10)
             + J/r + K*std::exp(-L*r);
    }
    constexpr double deriv_impl(double r) const {
        const double A=params[0], B=params[1], C=params[2], D=params[3];
        const double E=params[4], F=params[5], G=params[6], H=params[7], I=params[8];
        const double J=params[9], K=params[10], L=params[11];
        return -A/(r*r) - (B/D)*std::exp((C-r)/D)
             + 6.0*E/std::pow(r,7) + 8.0*F/std::pow(r,9)
             - G*H*std::exp(-H*r) + 10.0*I/std::pow(r,11)
             - J/(r*r) - K*L*std::exp(-L*r);
    }
};

// ── Softshell: V = A/r^n (soft-core repulsion) ───────────────────────────────
// params: {A, n}
struct Softshell : AnalyticBase<Softshell, 2> {
    Softshell(double A, double n, double lo, double hi)
        : AnalyticBase({A, n}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, n] = params;
        return A / std::pow(r, n);
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, n] = params;
        return -A * n / std::pow(r, n + 1.0);
    }
};

// ── ExpPlus: V = A exp(−Br) + C ──────────────────────────────────────────────
// params: {A, B, C}
struct ExpPlus : AnalyticBase<ExpPlus, 3> {
    ExpPlus(double A, double B, double C, double lo, double hi)
        : AnalyticBase({A, B, C}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C] = params;
        return A * std::exp(-B * r) + C;
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C] = params;
        return -A * B * std::exp(-B * r);
    }
};

// ── Strmm: V = (A + Br + Cr²) exp(−Dr) + E ──────────────────────────────────
// params: {A, B, C, D, E}
struct Strmm : AnalyticBase<Strmm, 5> {
    Strmm(double A, double B, double C, double D, double E, double lo, double hi)
        : AnalyticBase({A, B, C, D, E}, lo, hi) {}

    constexpr double eval_impl(double r) const {
        const auto [A, B, C, D, E] = params;
        return (A + B * r + C * r * r) * std::exp(-D * r) + E;
    }
    constexpr double deriv_impl(double r) const {
        const auto [A, B, C, D, E] = params;
        const double poly  = A + B * r + C * r * r;
        const double dpoly = B + 2.0 * C * r;
        return std::exp(-D * r) * (dpoly - D * poly);
    }
};

} // namespace potfit
