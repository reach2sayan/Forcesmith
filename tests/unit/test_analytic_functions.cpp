#include "forcesmith/potentials/analytic_potential.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace forcesmith;

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

// ── Independent value oracle for every form ──────────────────────────────────
// The FD tests above check that deriv agrees with eval; they cannot catch a
// form whose EXPRESSION was transcribed wrongly, because both sides would be
// wrong together. This test pins the value side against the reference formula
// written out independently here — the shape the hand-written kernels had
// before the expressions became symbolic. Any drift in a form's maths, or in
// the order its parameters are read, fails here.

namespace {

// |a - b| <= rtol·|b| — bit-equality is not required (a symbolic expression may
// associate operands differently), agreement to 1e-12 relative is.
void expect_value(double got, double want, const char *what, double r) {
    EXPECT_NEAR(got, want, 1e-12 * std::abs(want) + 1e-300)
        << what << " at r=" << r;
}

// The reference cutoffs, as the branch-free forms replaced them.
double ref_fc(double r, double R, double S) {
    if (r <= R) return 1.0;
    if (r >= S) return 0.0;
    return 0.5 + 0.5 * std::cos(M_PI * (r - R) / (S - R));
}
double ref_sc(double r, double r0, double h) {
    if (r >= r0) return 0.0;
    const double u = (r - r0) / h, u4 = (u * u) * (u * u);
    return u4 / (1.0 + u4);
}

} // namespace

TEST(AnalyticFunctions, EveryFormMatchesItsReferenceFormula) {
    for (double r : {1.4, 2.05, 2.9}) {
        {   LennardJones f(0.7, 1.9, 0.5, 6.0);
            const double sr6 = std::pow(1.9 / r, 6);
            expect_value(f.eval(r), 4.0 * 0.7 * (sr6 * sr6 - sr6), "lj", r); }
        {   Morse f(0.35, 1.4, 2.6, 0.5, 6.0);
            const double e = std::exp(-1.4 * (r - 2.6));
            expect_value(f.eval(r), 0.35 * (1.0 - e) * (1.0 - e) - 0.35, "morse", r); }
        {   Buckingham f(500.0, 0.3, 2.0, 0.5, 6.0);
            const double x = (0.3 * 0.3) / (r * r);
            expect_value(f.eval(r), 500.0 * std::exp(-r / 0.3) - 2.0 * x * x * x,
                         "buckingham", r); }
        {   Born f(0.4, 2.1, 3.2, 1.7, 2.4, 0.5, 6.0);
            const double r2 = r * r, r6 = r2 * r2 * r2, r8 = r6 * r2;
            expect_value(f.eval(r), 0.4 * std::exp((3.2 - r) / 2.1) - 1.7 / r6 + 2.4 / r8,
                         "born", r); }
        {   PowerDecay f(1.3, 2.4, 0.5, 6.0);
            expect_value(f.eval(r), 1.3 / std::pow(r, 2.4), "power_decay", r); }
        {   ExpDecay f(4.2, 1.1, 0.5, 6.0);
            expect_value(f.eval(r), 4.2 * std::exp(-1.1 * r), "exp_decay", r); }
        {   MexpDecay f(0.9, 1.2, 2.0, 0.5, 6.0);
            expect_value(f.eval(r), 0.9 * std::exp(-1.2 * (r - 2.0)), "mexp_decay", r); }
        {   Harmonic f(0.6, 2.2, 0.5, 6.0);
            expect_value(f.eval(r), 0.6 * (r - 2.2) * (r - 2.2), "harmonic", r); }
        {   Universal f(1.4, 1.1, 2.3, 0.2, 0.5, 6.0);
            const double E0 = 1.4, a = 1.1, b = 2.3, c = 0.2;
            expect_value(f.eval(r),
                         E0 * (b / (b - a) * std::pow(r, a) - a / (b - a) * std::pow(r, b)) + c * r,
                         "universal", r); }
        {   Eopp f(15.0, 6.0, 5.0, 3.0, 2.5, 3.0, 0.5, 6.0);
            expect_value(f.eval(r),
                         15.0 / std::pow(r, 6.0) + (5.0 / std::pow(r, 3.0)) * std::cos(2.5 * r + 3.0),
                         "eopp", r); }
        {   EoppExp f(12.0, 1.3, 5.0, 3.0, 2.5, 3.0, 0.5, 6.0);
            expect_value(f.eval(r),
                         12.0 * std::exp(-1.3 * r) + (5.0 / std::pow(r, 3.0)) * std::cos(2.5 * r + 3.0),
                         "eopp_exp", r); }
        {   Meopp f(15.0, 6.0, 5.0, 3.0, 2.5, 3.0, 0.4, 0.5, 6.0);
            expect_value(f.eval(r),
                         15.0 / std::pow(r - 0.4, 6.0) + (5.0 / std::pow(r, 3.0)) * std::cos(2.5 * r + 3.0),
                         "meopp", r); }
        {   GenLJ f(0.8, 12.0, 6.0, 2.4, 0.1, 0.5, 6.0);
            const double A = 0.8, n = 12.0, m = 6.0, r0 = 2.4, B = 0.1, x = r / r0;
            expect_value(f.eval(r),
                         A / (m - n) * (m * std::pow(x, -n) - n * std::pow(x, -m)) + B,
                         "gen_lj", r); }
        {   DoubleMorse f(0.4, 1.3, 2.2, 0.2, 2.1, 2.9, 0.05, 0.5, 6.0);
            const double e1 = std::exp(-1.3 * (r - 2.2)), e2 = std::exp(-2.1 * (r - 2.9));
            expect_value(f.eval(r),
                         0.4 * ((1.0 - e1) * (1.0 - e1) - 1.0) +
                             0.2 * ((1.0 - e2) * (1.0 - e2) - 1.0) + 0.05,
                         "double_morse", r); }
        {   DoubleExp f(0.7, 1.9, 2.3, 1.4, 2.0, 0.5, 6.0);
            const double dr = r - 2.3;
            expect_value(f.eval(r),
                         0.7 * std::exp(-1.9 * dr * dr) + std::exp(-1.4 * (r - 2.0)),
                         "double_exp", r); }
        {   Mishin f(0.9, 1.6, 0.05, 1.0, 2.0, 1.3, 0.5, 6.0);
            const double z = r - 1.0, e = std::exp(-1.3 * z);
            expect_value(f.eval(r), 0.9 * std::pow(z, 2.0) * e * (1.0 + 1.6 * e) + 0.05,
                         "mishin", r); }
        {   SqrtFunc f(0.8, 2.0, 0.5, 6.0);
            expect_value(f.eval(r), 0.8 * std::sqrt(r / 2.0), "sqrt", r); }
        {   ConstFunc f(-7.42, 0.5, 6.0);
            expect_value(f.eval(r), -7.42, "const", r); }
        {   Parabola f(1.1, -6.0, 10.0, 0.5, 6.0);
            expect_value(f.eval(r), 1.1 * r * r - 6.0 * r + 10.0, "parabola", r); }
        {   Poly5 f(0.3, 1.1, -0.4, 0.2, -0.05, 0.5, 6.0);
            const double s = r - 1.0, s2 = s * s;
            expect_value(f.eval(r),
                         0.3 + 0.5 * 1.1 * s2 + (-0.4) * s * s2 + 0.2 * s2 * s2 +
                             (-0.05) * s2 * s2 * s,
                         "poly5", r); }
        {   StiwWeb2 f(7.0, 0.6, 4.0, 0.0, 1.2, 3.5, 0.5, 6.0);
            const double poly = 7.0 * std::pow(r, -4.0) - 0.6 * std::pow(r, -0.0);
            expect_value(f.eval(r), poly * std::exp(1.2 / (r - 3.5)), "stiweb_2", r); }
        {   StiwWeb3 f(1.2, 3.5, 0.5, 6.0);
            expect_value(f.eval(r), std::exp(1.2 / (r - 3.5)), "stiweb_3", r); }
        {   TersoffPot f(1830.8, 471.18, 2.4799, 1.7322, 1.1e-6, 0.78734,
                         100390.0, 16.217, -0.59825, 2.7, 3.0, 0.5, 6.0);
            expect_value(f.eval(r),
                         ref_fc(r, 2.7, 3.0) *
                             (1830.8 * std::exp(-2.4799 * r) - 471.18 * std::exp(-1.7322 * r)),
                         "tersoff", r); }
        {   TersoffMix f(0.9, 1.4, 0.5, 6.0);
            expect_value(f.eval(r), 0.9 * std::exp(-1.4 * r), "tersoff_mix", r); }
        {   TersoffModPot f(std::array<double, 16>{1830.8, 471.18, 2.4799, 1.7322, 1.1e-6,
                                                   0.78734, 100390.0, 16.217, -0.59825,
                                                   2.7, 3.0, 0.01, -0.02, 0.003, -0.0004, 0.00005},
                            0.5, 6.0);
            const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
            const double corr = 1.0 + 0.01 * r - 0.02 * r2 + 0.003 * r3 - 0.0004 * r4 + 0.00005 * r5;
            expect_value(f.eval(r),
                         ref_fc(r, 2.7, 3.0) *
                             (1830.8 * std::exp(-2.4799 * r) - 471.18 * std::exp(-1.7322 * r)) * corr,
                         "tersoff_mod", r); }
        {   Kawamura f(1.2, -0.8, 3.0, 1.1, 0.9, 0.15, 0.12, 2.0, 1.5, 0.5, 6.0);
            const double s = 0.15 + 0.12, t = 1.1 + 0.9;
            expect_value(f.eval(r),
                         1.2 * -0.8 / r + 3.0 * s * std::exp((t - r) / s) -
                             2.0 * 1.5 / std::pow(r, 6.0),
                         "kawamura", r); }
        {   KawamuraMix f(std::array<double, 12>{1.2, -0.8, 3.0, 1.1, 0.9, 0.15, 0.12,
                                                 2.0, 1.5, 0.4, 1.3, 2.2},
                          0.5, 6.0);
            const double s = 0.15 + 0.12, t = 1.1 + 0.9, w = r - 2.2;
            expect_value(f.eval(r),
                         1.2 * -0.8 / r + 3.0 * s * std::exp((t - r) / s) -
                             2.0 * 1.5 / std::pow(r, 6.0) +
                             3.0 * 0.4 * (std::exp(-2.0 * 1.3 * w) - 2.0 * std::exp(-1.3 * w)),
                         "kawamura_mix", r); }
        {   Softshell f(1.7, 3.0, 0.5, 6.0);
            expect_value(f.eval(r), std::pow(1.7 / r, 3.0), "softshell", r); }
        {   ExpPlus f(4.2, 1.1, -0.3, 0.5, 6.0);
            expect_value(f.eval(r), 4.2 * std::exp(-1.1 * r) - 0.3, "exp_plus", r); }
        {   Strmm f(1.5, -0.5, 0.1, 1.5, 0.2, 0.5, 6.0);
            const double s = r - 0.2;
            expect_value(f.eval(r),
                         2.0 * 1.5 * std::exp(-(-0.5) / 2.0 * s) -
                             0.1 * (1.0 + 1.5 * s) * std::exp(-1.5 * s),
                         "strmm", r); }

        // The `_sc` wrapper: base value times the switching factor at rmax.
        {   LennardJonesSC f(0.7, 1.9, 1.0, 0.5, 6.0);
            const double sr6 = std::pow(1.9 / r, 6);
            expect_value(f.eval(r), 4.0 * 0.7 * (sr6 * sr6 - sr6) * ref_sc(r, 6.0, 1.0),
                         "lj_sc", r); }
        {   MorseSC f(0.35, 1.4, 2.6, 1.0, 0.5, 6.0);
            const double e = std::exp(-1.4 * (r - 2.6));
            expect_value(f.eval(r), (0.35 * (1.0 - e) * (1.0 - e) - 0.35) * ref_sc(r, 6.0, 1.0),
                         "morse_sc", r); }
        {   ExpDecaySC f(4.2, 1.1, 1.0, 0.5, 6.0);
            expect_value(f.eval(r), 4.2 * std::exp(-1.1 * r) * ref_sc(r, 6.0, 1.0),
                         "exp_decay_sc", r); }
        {   EoppSC f(15.0, 6.0, 5.0, 3.0, 2.5, 3.0, 1.0, 0.5, 6.0);
            expect_value(f.eval(r),
                         (15.0 / std::pow(r, 6.0) +
                          (5.0 / std::pow(r, 3.0)) * std::cos(2.5 * r + 3.0)) *
                             ref_sc(r, 6.0, 1.0),
                         "eopp_sc", r); }
    }
}

// The cutoff pieces the branch-free expressions replaced: identical inside the
// switching region and exactly flat outside it.
TEST(AnalyticFunctions, TersoffCutoffMatchesThePiecewiseForm) {
    TersoffPot f(1830.8, 471.18, 2.4799, 1.7322, 1.1e-6, 0.78734, 100390.0,
                 16.217, -0.59825, 2.7, 3.0, 0.5, 6.0);
    const auto pair = [](double r) {
        return 1830.8 * std::exp(-2.4799 * r) - 471.18 * std::exp(-1.7322 * r);
    };
    for (double r : {1.0, 2.69, 2.7, 2.8, 2.9, 3.0, 3.01, 4.0}) {
        expect_value(f.eval(r), ref_fc(r, 2.7, 3.0) * pair(r), "tersoff cutoff", r);
    }
    EXPECT_DOUBLE_EQ(f.eval(3.5), 0.0);   // beyond S
    EXPECT_DOUBLE_EQ(f.deriv(3.5), 0.0);
    EXPECT_DOUBLE_EQ(f.eval(2.5), pair(2.5)); // below R: fc == 1 exactly
}
