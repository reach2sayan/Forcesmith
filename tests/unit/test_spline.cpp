#include "potfit/potentials/spline.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <ranges>

using namespace potfit;

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
// potfit's splint), not clamping. Boundary slope here is (3-5)/(2-1) = -2.
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
