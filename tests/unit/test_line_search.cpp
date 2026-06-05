#include "forcesmith/optimization/line_search.hpp"
#include "forcesmith/optimization/solver.hpp"

#include <boost/math/tools/minima.hpp>
#include <gtest/gtest.h>

#include <limits>

using Eigen::VectorXd;

TEST(LineSearch, BrentQuadratic) {
    // Direct use of boost brent: minimize (α-3)²
    auto g = [](double a) { return (a - 3.0) * (a - 3.0); };
    std::uintmax_t iters = 200;
    auto [alpha, fval] = boost::math::tools::brent_find_minima(
        g, 0.0, 10.0,
        std::numeric_limits<double>::digits / 2,
        iters);
    EXPECT_NEAR(alpha, 3.0, 1e-6);
    EXPECT_NEAR(fval,  0.0, 1e-12);
}

TEST(LineSearch, LinminQuadraticResidual) {
    // F(x) = [x[0] - 5]; objective 0.5*(x[0]-5)²; minimum at x[0]=5.
    VectorXd x   = VectorXd::Zero(1);
    VectorXd dir = VectorXd::Ones(1);

    auto F = [](const VectorXd &v) {
        VectorXd r(1);
        r[0] = v[0] - 5.0;
        return r;
    };

    double alpha = forcesmith::linmin(x, dir, F, {});

    EXPECT_NEAR(alpha, 5.0, 1e-6);
    EXPECT_NEAR(x[0],  5.0, 1e-6);
}

TEST(LineSearch, LinminMovesSteadily) {
    // F(x) = [x[0]-2, x[1]-7]; step from x=[0,0] in steepest-descent direction.
    VectorXd x(2);
    x << 0.0, 0.0;

    auto F = [](const VectorXd &v) {
        VectorXd r(2);
        r[0] = v[0] - 2.0;
        r[1] = v[1] - 7.0;
        return r;
    };

    double before = 0.5 * F(x).squaredNorm();

    // Steepest-descent direction: -∇(0.5*||F||²) = -F (since J=I here)
    VectorXd dir = -F(x);
    dir.normalize();

    forcesmith::linmin(x, dir, F, {});

    EXPECT_LT(0.5 * F(x).squaredNorm(), before);
}

TEST(LineSearch, LinminNoStepIfAtMinimum) {
    // F(x)=0 everywhere: g(α)=0, bracket sees fb >= fa immediately,
    // Brent returns the left endpoint ≈ 0.
    VectorXd x   = VectorXd::Zero(1);
    VectorXd dir = VectorXd::Ones(1);

    auto F = [](const VectorXd &) { return VectorXd::Zero(1); };

    double alpha = forcesmith::linmin(x, dir, F, {});

    EXPECT_NEAR(x[0], alpha, 1e-6);
    EXPECT_NEAR(0.5 * F(x).squaredNorm(), 0.0, 1e-14);
}

TEST(LineSearchSolver, ConvergesCoupledQuadratic) {
    // F(x) = [x0 + x1 - 3, x0 - x1 - 1]; zero residual at (2, 1). The coupled
    // residuals exercise Powell's conjugate-direction update (axis-aligned
    // line searches alone would zig-zag).
    auto F = [](const VectorXd &v) {
        VectorXd r(2);
        r[0] = v[0] + v[1] - 3.0;
        r[1] = v[0] - v[1] - 1.0;
        return r;
    };

    VectorXd x(2);
    x << 0.0, 0.0;

    forcesmith::LineSearchSolver solver{200, 1e-9};
    const VectorXd unbounded =
        VectorXd::Constant(2, std::numeric_limits<double>::infinity());
    solver.minimize(x, F, {}, 2, -unbounded, unbounded);

    EXPECT_NEAR(x[0], 2.0, 1e-5);
    EXPECT_NEAR(x[1], 1.0, 1e-5);
    EXPECT_NEAR(0.5 * F(x).squaredNorm(), 0.0, 1e-10);
}
