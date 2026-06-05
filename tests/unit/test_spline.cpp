#include "forcesmith/potentials/spline.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <cmath>
#include <ranges>
#include <vector>

using namespace forcesmith;

// makima reproduces linear functions exactly (all differences are equal, weights degenerate to average = slope).

TEST(SplinePotential, EvalAtKnotsEqualsValues) {
    // arbitrary non-uniform knots
    SplinePotential sp({0.0, 1.5, 3.0, 5.0, 6.0},
                       {2.0, 0.5, 3.0, 1.0, 4.0});
    const std::vector<double> xs = {0.0, 1.5, 3.0, 5.0, 6.0};
    const std::vector<double> ys = {2.0, 0.5, 3.0, 1.0, 4.0};
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_NEAR(sp.eval(xs[i]), ys[i], 1e-10) << "at knot " << i;
}

TEST(SplinePotential, LinearFunctionExact) {
    const double a = 3.0, b = -1.5;
    SplinePotential sp({0.0, 1.0, 2.5, 4.0},
                       {b, a + b, 2.5*a + b, 4.0*a + b});
    EXPECT_NEAR(sp.eval(0.5),  0.5*a + b, 1e-10);
    EXPECT_NEAR(sp.eval(1.75), 1.75*a + b, 1e-10);
    EXPECT_NEAR(sp.eval(3.0),  3.0*a + b, 1e-10);
}

TEST(SplinePotential, LinearDerivExact) {
    const double slope = 2.5;
    SplinePotential sp({0.0, 1.0, 3.0, 5.0},
                       {0.0, slope, 3.0*slope, 5.0*slope});
    EXPECT_NEAR(sp.deriv(0.5),  slope, 1e-10);
    EXPECT_NEAR(sp.deriv(2.0),  slope, 1e-10);
    EXPECT_NEAR(sp.deriv(4.0),  slope, 1e-10);
}

// Derivative is consistent with numerical central difference of eval.
TEST(SplinePotential, DerivConsistentWithEval) {
    SplinePotential sp({0.0, 1.0, 2.0, 3.0, 4.0},
                       {0.0, 1.0, 4.0, 9.0, 16.0});
    const double eps = 1e-5;
    for (double r : {0.5, 1.5, 2.5, 3.5}) {
        const double fd = (sp.eval(r + eps) - sp.eval(r - eps)) / (2.0 * eps);
        EXPECT_NEAR(sp.deriv(r), fd, 1e-5) << "at r=" << r;
    }
}

// Out of range, eval extrapolates linearly with the boundary slope (matching
// forcesmith's splint), not clamping. Boundary slope here is (3-5)/(2-1) = -2.
TEST(SplinePotential, BoundaryExtrapolateBelow) {
    SplinePotential sp({1.0, 2.0, 3.0}, {5.0, 3.0, 1.0});
    EXPECT_NEAR(sp.eval(0.0), 7.0, 1e-15);   // 5 + (-2)(0-1)
    EXPECT_NEAR(sp.deriv(0.0), -2.0, 1e-15);
}

TEST(SplinePotential, BoundaryExtrapolateAbove) {
    SplinePotential sp({1.0, 2.0, 3.0}, {5.0, 3.0, 1.0});
    EXPECT_NEAR(sp.eval(4.0), -1.0, 1e-15);  // 1 + (-2)(4-3)
    EXPECT_NEAR(sp.deriv(4.0), -2.0, 1e-15);
}

TEST(SplinePotential, SpanReturnsKnotExtents) {
    SplinePotential sp({0.5, 2.5, 4.5}, {1.0, 2.0, 3.0});
    auto [lo, hi] = sp.span();
    EXPECT_DOUBLE_EQ(lo, 0.5);
    EXPECT_DOUBLE_EQ(hi, 4.5);
}

TEST(SplinePotential, TwoKnotLinearCase) {
    SplinePotential sp({0.0, 1.0}, {0.0, 1.0});
    EXPECT_NEAR(sp.eval(0.5),  0.5, 1e-15);
    EXPECT_NEAR(sp.deriv(0.5), 1.0, 1e-15);
}

TEST(SplinePotential, EvalMonotonicOnMonotonicData) {
    // Monotone input; spline should not wildly oscillate between knots.
    SplinePotential sp({0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0});
    for (int k : std::views::iota(1, 10)) {
        const double r = k * 0.3;
        EXPECT_GT(sp.eval(r + 0.05), sp.eval(r - 0.05))
            << "non-monotone at r=" << r;
    }
}

TEST(SplinePotential, ConstructFromKnots) {
    SplinePotential pp({1.0, 2.0, 3.0}, {2.0, 0.0, 2.0});
    auto [lo, hi] = pp.span();
    EXPECT_DOUBLE_EQ(lo, 1.0);
    EXPECT_DOUBLE_EQ(hi, 3.0);
    EXPECT_NEAR(pp.eval(2.0), 0.0, 1e-10);
}

// ── Fit-time evaluation cache (prepare_site / eval_at / deriv_at) ────────────

// The cached path must reproduce eval/deriv across interior, both boundaries,
// and exactly-at-knot for a cubic (n >= 4) spline.
TEST(SplinePotentialCache, EvalAtMatchesEvalCubic) {
    SplinePotential sp({0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                       {2.0, 0.5, 3.0, 1.0, 4.0, 2.5});
    for (double r : {-0.3, 0.0, 0.4, 1.0, 2.7, 3.9999, 5.0, 5.6}) {
        const int s = sp.prepare_site(r);
        ASSERT_GE(s, 0);
        EXPECT_NEAR(sp.eval_at(s), sp.eval(r), 1e-11) << "eval at r=" << r;
        EXPECT_NEAR(sp.deriv_at(s), sp.deriv(r), 1e-11) << "deriv at r=" << r;
    }
}

// Same for the n < 4 piecewise-linear fallback.
TEST(SplinePotentialCache, EvalAtMatchesEvalLinearFallback) {
    SplinePotential sp({1.0, 2.0, 3.0}, {5.0, 3.0, 1.0}); // n=3
    for (double r : {0.0, 1.5, 2.5, 4.0}) {
        const int s = sp.prepare_site(r);
        ASSERT_GE(s, 0);
        EXPECT_NEAR(sp.eval_at(s), sp.eval(r), 1e-12) << "eval at r=" << r;
        EXPECT_NEAR(sp.deriv_at(s), sp.deriv(r), 1e-12) << "deriv at r=" << r;
    }
}

// A site is geometry-only: after scatter_params changes the knot VALUES (same
// x-grid), a previously-prepared site still tracks eval/deriv — proving the
// cache survives the per-iteration parameter updates of a fit.
TEST(SplinePotentialCache, SurvivesScatterParams) {
    SplinePotential sp({0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                       {2.0, 0.5, 3.0, 1.0, 4.0, 2.5});
    const std::vector<double> rs = {0.4, 2.7, 3.5};
    std::vector<int> sites;
    for (double r : rs) {
        sites.push_back(sp.prepare_site(r));
    }
    Eigen::VectorXd y(6);
    y << 1.0, -2.0, 0.5, 3.5, -1.0, 2.0; // new values, same knots
    sp.scatter_params(y, 0);
    for (std::size_t k = 0; k < rs.size(); ++k) {
        EXPECT_NEAR(sp.eval_at(sites[k]), sp.eval(rs[k]), 1e-11) << "r=" << rs[k];
        EXPECT_NEAR(sp.deriv_at(sites[k]), sp.deriv(rs[k]), 1e-11) << "r=" << rs[k];
    }
}

// Repeated prepare_site for the same distance returns the same cached index.
TEST(SplinePotentialCache, MemoizesByDistance) {
    SplinePotential sp({0.0, 1.0, 2.0, 3.0, 4.0},
                       {0.0, 1.0, 4.0, 9.0, 16.0});
    EXPECT_EQ(sp.prepare_site(2.3), sp.prepare_site(2.3));
    EXPECT_NE(sp.prepare_site(2.3), sp.prepare_site(1.1));
}

// ── Fused eval_and_deriv / eval_and_deriv_at ─────────────────────────────────
// The fused calls must be BIT-IDENTICAL to the two separate calls — the force
// loops rely on this so a fit's residuals are unchanged. Exact == (not a
// tolerance): same operations in the same order must yield the same bits.

// Cached fused path vs eval_at/deriv_at, for a cubic (n >= 4) spline. Covers
// interior, exactly-at-knot, and both extrapolation boundaries.
TEST(SplinePotentialCache, EvalAndDerivAtMatchesSeparateCubic) {
    SplinePotential sp({0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                       {2.0, 0.5, 3.0, 1.0, 4.0, 2.5});
    for (double r : {-0.3, 0.0, 0.4, 1.0, 2.7, 3.9999, 5.0, 5.6}) {
        const int s = sp.prepare_site(r);
        ASSERT_GE(s, 0);
        const auto [v, d] = sp.eval_and_deriv_at(s);
        EXPECT_EQ(v, sp.eval_at(s)) << "value at r=" << r;
        EXPECT_EQ(d, sp.deriv_at(s)) << "deriv at r=" << r;
    }
}

// Same for the n < 4 piecewise-linear fallback (the Linear EvalSite branch).
TEST(SplinePotentialCache, EvalAndDerivAtMatchesSeparateLinear) {
    SplinePotential sp({1.0, 2.0, 3.0}, {5.0, 3.0, 1.0}); // n=3
    for (double r : {0.0, 1.5, 2.5, 4.0}) {
        const int s = sp.prepare_site(r);
        ASSERT_GE(s, 0);
        const auto [v, d] = sp.eval_and_deriv_at(s);
        EXPECT_EQ(v, sp.eval_at(s)) << "value at r=" << r;
        EXPECT_EQ(d, sp.deriv_at(s)) << "deriv at r=" << r;
    }
}

// Uncached fused path vs eval(r)/deriv(r) — the shared interval search must not
// change the result. Cubic and linear, including both extrapolation boundaries.
TEST(SplinePotential, EvalAndDerivMatchesSeparate) {
    SplinePotential cubic({0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                          {2.0, 0.5, 3.0, 1.0, 4.0, 2.5});
    for (double r : {-0.3, 0.0, 0.4, 1.0, 2.7, 3.9999, 5.0, 5.6}) {
        const auto [v, d] = cubic.eval_and_deriv(r);
        EXPECT_EQ(v, cubic.eval(r)) << "value at r=" << r;
        EXPECT_EQ(d, cubic.deriv(r)) << "deriv at r=" << r;
    }
    SplinePotential linear({1.0, 2.0, 3.0}, {5.0, 3.0, 1.0}); // n=3
    for (double r : {0.0, 1.5, 2.5, 4.0}) {
        const auto [v, d] = linear.eval_and_deriv(r);
        EXPECT_EQ(v, linear.eval(r)) << "value at r=" << r;
        EXPECT_EQ(d, linear.deriv(r)) << "deriv at r=" << r;
    }
}
