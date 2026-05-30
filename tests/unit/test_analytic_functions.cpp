#include "potfit/potentials/analytic_potential.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace potfit;

// ── FD helper ─────────────────────────────────────────────────────────────────
// Checks that deriv(r) ≈ (eval(r+h) - eval(r-h)) / (2h) within 1e-6 relative
// tolerance, 1e-10 absolute tolerance.  Returns true on pass.

template <typename Pot>
static void fd_check(const Pot& pot, double r, double h = 1e-5) {
    const double fd = (pot.eval(r + h) - pot.eval(r - h)) / (2.0 * h);
    const double an = pot.deriv(r);
    EXPECT_NEAR(an, fd, 1e-6 * std::abs(fd) + 1e-10)
        << "FD=" << fd << "  analytic=" << an << "  at r=" << r;
}

// ── Tests: existing functions ──────────────────────────────────────────────────

TEST(AnalyticFunctions, LennardJones_FD) {
    LennardJones lj(1.0, 1.0, 0.5, 10.0);
    fd_check(lj, 1.2);
    fd_check(lj, 2.5);
}

TEST(AnalyticFunctions, Morse_FD) {
    Morse m(1.5, 2.0, 2.0, 0.5, 10.0);
    fd_check(m, 1.5);
    fd_check(m, 2.5);
}

// ── Tests: new functions ───────────────────────────────────────────────────────

TEST(AnalyticFunctions, Buckingham_FD) {
    Buckingham b(500.0, 0.3, 2.0, 0.5, 10.0);
    fd_check(b, 1.5);
    fd_check(b, 3.0);
}

TEST(AnalyticFunctions, Born_FD) {
    Born b(1000.0, 3.0, 5.0, 2.0, 1.0, 0.5, 10.0);
    fd_check(b, 1.5);
    fd_check(b, 2.5);
}

TEST(AnalyticFunctions, PowerDecay_FD) {
    PowerDecay p(3.0, 4.0, 0.5, 10.0);
    fd_check(p, 2.0);
    fd_check(p, 3.0);
}

TEST(AnalyticFunctions, ExpDecay_FD) {
    ExpDecay e(2.0, 1.5, 0.5, 10.0);
    fd_check(e, 1.0);
    fd_check(e, 2.5);
}

TEST(AnalyticFunctions, MexpDecay_FD) {
    MexpDecay m(1.0, 2.0, 1.5, 0.5, 10.0);
    fd_check(m, 1.0);
    fd_check(m, 2.5);
}

TEST(AnalyticFunctions, Harmonic_FD) {
    Harmonic h(2.0, 2.5, 0.5, 10.0);
    fd_check(h, 1.5);
    fd_check(h, 3.5);
}

TEST(AnalyticFunctions, Universal_FD) {
    Universal u(-3.0, 2.0, 1.5, 2.0, 0.5, 10.0);
    fd_check(u, 1.5);
    fd_check(u, 2.5);
}

TEST(AnalyticFunctions, Eopp_FD) {
    Eopp e(10.0, 6.0, 1.0, 2.0, 2.0, 4.0, 0.5, 10.0);
    fd_check(e, 1.5);
    fd_check(e, 2.5);
}

TEST(AnalyticFunctions, EoppExp_FD) {
    EoppExp e(2.0, 1.0, 1.0, 2.0, 0.5, 3.0, 0.5, 10.0);
    fd_check(e, 1.0);
    fd_check(e, 2.0);
}

TEST(AnalyticFunctions, Meopp_FD) {
    Meopp m(1.0, 1.0, 0.5, 2.0, 1.0, 1.5, 0.3, 0.5, 10.0);
    fd_check(m, 1.5);
    fd_check(m, 2.5);
}

TEST(AnalyticFunctions, GenLJ_FD) {
    GenLJ g(2.0, 2.5, 12.0, 6.0, 0.0, 0.5, 10.0);
    fd_check(g, 2.0);
    fd_check(g, 3.0);
}

TEST(AnalyticFunctions, DoubleMorse_FD) {
    DoubleMorse dm(1.0, 2.0, 2.0, 0.5, 1.5, 3.0, 0.1, 0.5, 10.0);
    fd_check(dm, 1.5);
    fd_check(dm, 3.0);
}

TEST(AnalyticFunctions, DoubleExp_FD) {
    DoubleExp de(2.0, 1.5, -1.0, 0.8, 0.3, 0.5, 10.0);
    fd_check(de, 1.0);
    fd_check(de, 3.0);
}

TEST(AnalyticFunctions, Mishin_FD) {
    Mishin m(1.5, -0.5, 2.0, 0.5, 0.3, 1.0, 0.5, 10.0);
    fd_check(m, 1.0);
    fd_check(m, 2.5);
}

TEST(AnalyticFunctions, SqrtFunc_FD) {
    SqrtFunc s(2.0, 1.0, 0.5, 10.0);
    fd_check(s, 1.5);
    fd_check(s, 3.0);
}

TEST(AnalyticFunctions, ConstFunc_Value) {
    ConstFunc c(3.14, 0.5, 10.0);
    EXPECT_DOUBLE_EQ(c.eval(1.0), 3.14);
    EXPECT_DOUBLE_EQ(c.eval(5.0), 3.14);
    EXPECT_DOUBLE_EQ(c.deriv(1.0), 0.0);
}

TEST(AnalyticFunctions, Parabola_FD) {
    Parabola p(1.0, -2.0, 0.5, 0.5, 10.0);
    fd_check(p, 1.5);
    fd_check(p, 3.0);
}

TEST(AnalyticFunctions, Poly5_FD) {
    Poly5 p(1.0, -0.5, 0.2, -0.1, 0.05, 0.5, 10.0);
    fd_check(p, 1.5);
    fd_check(p, 2.5);
}

TEST(AnalyticFunctions, StiwWeb2_FD) {
    // SW Si 2-body: a=1.8, sigma=2.0951, cutoff at 3.77 Å.  Test at r=2.3 (inside).
    StiwWeb2 sw(7.0496, 0.6022, 4.0, 0.0, 1.8, 2.0951, 0.5, 4.0);
    fd_check(sw, 2.3);
    // Beyond cutoff: value and derivative must be exactly zero.
    EXPECT_DOUBLE_EQ(sw.eval(4.0),  0.0);
    EXPECT_DOUBLE_EQ(sw.deriv(4.0), 0.0);
}

TEST(AnalyticFunctions, StiwWeb3_FD) {
    // h(r) = exp(γ/(r−a)) for r < a; test at r=1.5 with a=3.0.
    StiwWeb3 sw3(2.0951 * 1.2, 3.7712, 0.5, 4.0);  // gamma=γσ, a=rcut
    fd_check(sw3, 2.0);
    fd_check(sw3, 3.0);
    EXPECT_DOUBLE_EQ(sw3.eval(4.0),  0.0);
    EXPECT_DOUBLE_EQ(sw3.deriv(4.0), 0.0);
}

TEST(AnalyticFunctions, TersoffPot_FD) {
    // Si Tersoff 1988 params; test inside cutoff (r=2.4 < R=2.7).
    TersoffPot tp(1830.8, 471.18, 2.4799, 1.7322,
                  1.1e-6, 0.78734, 1.0039e5, 16.217, -0.59825,
                  2.7, 3.0, 0.5, 4.0);
    fd_check(tp, 2.4);
    // Test in the transition zone.
    fd_check(tp, 2.85);
    // Beyond S: zero.
    EXPECT_DOUBLE_EQ(tp.eval(3.1),  0.0);
    EXPECT_DOUBLE_EQ(tp.deriv(3.1), 0.0);
}

TEST(AnalyticFunctions, TersoffMix_FD) {
    TersoffMix tm(1.0, 0.5, 0.5, 10.0);
    fd_check(tm, 1.5);
    fd_check(tm, 3.0);
}

TEST(AnalyticFunctions, TersoffModPot_FD) {
    TersoffModPot tmp(
        {1830.8, 471.18, 2.4799, 1.7322,
         1.1e-6, 0.78734, 1.0039e5, 16.217, -0.59825,
         2.7, 3.0,
         0.01, -0.005, 0.002, -0.001, 0.0005},
        0.5, 4.0);
    fd_check(tmp, 2.4);
    fd_check(tmp, 2.85);
}

TEST(AnalyticFunctions, Kawamura_FD) {
    // Avoid singularity at r=0; test at r >= 1.5.
    Kawamura k(1.0, 0.5, 2.0, 0.3, 0.1, 0.05, 0.3, 2.0, 0.01, 0.5, 10.0);
    fd_check(k, 1.5);
    fd_check(k, 3.0);
}

TEST(AnalyticFunctions, KawamuraMix_FD) {
    KawamuraMix km(
        {1.0, 0.5, 2.0, 0.3, 0.1, 0.05, 0.3, 2.0, 0.01,
         0.5, 0.2, 1.0},
        0.5, 10.0);
    fd_check(km, 1.5);
    fd_check(km, 3.0);
}

TEST(AnalyticFunctions, Softshell_FD) {
    Softshell s(5.0, 8.0, 0.5, 10.0);
    fd_check(s, 1.5);
    fd_check(s, 2.5);
}

TEST(AnalyticFunctions, ExpPlus_FD) {
    ExpPlus e(3.0, 2.0, 1.0, 0.5, 10.0);
    fd_check(e, 1.0);
    fd_check(e, 2.5);
}

TEST(AnalyticFunctions, Strmm_FD) {
    Strmm s(1.5, -0.5, 0.1, 1.5, 0.2, 0.5, 10.0);
    fd_check(s, 1.0);
    fd_check(s, 2.5);
}

// ── Spot-check analytic values ────────────────────────────────────────────────

TEST(AnalyticFunctions, LennardJones_ZeroAtSigma) {
    LennardJones lj(1.0, 1.0, 0.5, 10.0);
    // LJ is zero at r = sigma.
    EXPECT_NEAR(lj.eval(1.0), 0.0, 1e-14);
}

TEST(AnalyticFunctions, Harmonic_ZeroAtEquilibrium) {
    Harmonic h(5.0, 2.3, 0.5, 10.0);
    EXPECT_NEAR(h.eval(2.3),  0.0, 1e-14);
    EXPECT_NEAR(h.deriv(2.3), 0.0, 1e-14);
}

TEST(AnalyticFunctions, ConstFunc_DerivIsZero) {
    ConstFunc c(-7.42, 0.5, 10.0);
    for (double r : {1.0, 2.0, 5.0})
        EXPECT_DOUBLE_EQ(c.deriv(r), 0.0);
}

TEST(AnalyticFunctions, Parabola_VertexDerivIsZero) {
    // V = (r-3)^2 + 1 = r^2 - 6r + 10, vertex at r=3.
    Parabola p(1.0, -6.0, 10.0, 0.5, 10.0);
    EXPECT_NEAR(p.deriv(3.0), 0.0, 1e-14);
}

TEST(AnalyticFunctions, DoubleMorse_MinimumEnergy) {
    // At equilibrium r = re, each Morse term = De*((1-1)^2 - 1) = -De.
    DoubleMorse dm(1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 0.0, 0.5, 10.0);
    // V(2.0) = -D1 - D2 + C = -1 - 1 + 0 = -2.
    EXPECT_NEAR(dm.eval(2.0), -2.0, 1e-14);
}
