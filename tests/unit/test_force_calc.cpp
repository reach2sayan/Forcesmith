#include "forcesmith/force/pair_force.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace forcesmith;

// ── Minimal analytic LJ potential for testing ─────────────────────────────────

struct LJPotential : NoBounds<LJPotential> {
    double eps   = 1.0;
    double sigma = 1.0;
    LJPotential(double e = 1.0, double s = 1.0) : eps(e), sigma(s) {}

    double eval(double r) const {
        const double sr6 = std::pow(sigma / r, 6);
        return 4.0 * eps * (sr6 * sr6 - sr6);
    }

    double deriv(double r) const {
        const double sr6 = std::pow(sigma / r, 6);
        return 4.0 * eps * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;
    }

    std::pair<double, double> span() const { return {0.1, 20.0}; }
    int    param_count() const { return 0; }
    void   gather_params(Eigen::VectorXd&, int) const {}
    void   scatter_params(const Eigen::VectorXd&, int) {}
};

// ── Test helpers ──────────────────────────────────────────────────────────────

static Configuration make_dimer_geometry(double r) {
    Configuration cfg;
    cfg.bc     = PeriodicBC(100.0 * Mat3::Identity());
    cfg.weight = 1.0;

    Atom a0, a1;
    a0.type = 0;  a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0;  a1.pos = {r,   0.0, 0.0};
    cfg.atoms = {a0, a1};
    return cfg;
}

static PairForceCalculator lj_calc(double eps = 1.0, double sigma = 1.0) {
    PairForceCalculator calc;
    calc.pair.reserve(1);
    calc.pair.emplace_back(LJPotential{eps, sigma});
    return calc;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(ForceCalculator, ZeroForceAtEquilibrium) {
    const double sigma = 2.0;
    const double eps   = 1.5;
    const double r_eq  = std::pow(2.0, 1.0 / 6.0) * sigma;

    auto calc = lj_calc(eps, sigma);
    auto cfg  = make_dimer_geometry(r_eq);
    calc.eval_forces(cfg);

    EXPECT_NEAR(cfg.atoms[0].calc_force.norm(), 0.0, 1e-10);
    EXPECT_NEAR(cfg.atoms[1].calc_force.norm(), 0.0, 1e-10);
}

TEST(ForceCalculator, NewtonThirdLaw) {
    const double sigma = 2.0;
    const double r     = 2.7 * sigma;

    auto calc = lj_calc(1.0, sigma);
    auto cfg  = make_dimer_geometry(r);
    calc.eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force + cfg.atoms[1].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-12);
}

TEST(ForceCalculator, AttractiveForceInWell) {
    const double sigma = 2.0;
    const double r     = 2.5 * sigma;

    auto calc = lj_calc(1.0, sigma);
    auto cfg  = make_dimer_geometry(r);
    calc.eval_forces(cfg);

    EXPECT_GT(cfg.atoms[0].calc_force[0], 0.0);
    EXPECT_LT(cfg.atoms[1].calc_force[0], 0.0);
}

TEST(ForceCalculator, EnergyNegativeInWell) {
    const double sigma = 2.0;
    const double r     = 2.5 * sigma;

    auto calc = lj_calc(1.0, sigma);
    auto cfg  = make_dimer_geometry(r);
    calc.eval_forces(cfg);

    EXPECT_TRUE(std::isfinite(cfg.calc_energy));
    EXPECT_LT(cfg.calc_energy, 0.0);
}

TEST(ForceCalculator, EnergyAtEquilibriumIsMinimum) {
    const double sigma = 2.0;
    const double eps   = 1.5;
    const double r_eq  = std::pow(2.0, 1.0 / 6.0) * sigma;

    auto calc = lj_calc(eps, sigma);
    auto cfg  = make_dimer_geometry(r_eq);
    calc.eval_forces(cfg);

    EXPECT_NEAR(cfg.calc_energy, -eps, 1e-12);
}

TEST(ForceCalculator, StressSymmetric) {
    const double sigma = 2.0;
    const double r     = 2.5 * sigma;

    auto calc = lj_calc(1.0, sigma);
    auto cfg  = make_dimer_geometry(r);
    calc.eval_forces(cfg);

    const SymTens& s = cfg.calc_stress;
    EXPECT_NEAR(s(0, 1), s(1, 0), 1e-14);
    EXPECT_NEAR(s(0, 2), s(2, 0), 1e-14);
    EXPECT_NEAR(s(1, 2), s(2, 1), 1e-14);
}

TEST(ForceCalculator, ForcesClearedOnReeval) {
    const double sigma = 2.0;
    const double r     = 2.5 * sigma;

    auto calc = lj_calc(1.0, sigma);
    auto cfg  = make_dimer_geometry(r);
    calc.eval_forces(cfg);
    const Vec3   f0_first = cfg.atoms[0].calc_force;
    const double e_first  = cfg.calc_energy;

    calc.eval_forces(cfg);
    EXPECT_NEAR((cfg.atoms[0].calc_force - f0_first).norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.calc_energy, e_first, 1e-14);
}

TEST(ForceCalculator, ParamCountZeroForNoParam) {
    auto calc = lj_calc(1.0, 2.0);
    EXPECT_EQ(calc.param_count(), 0);
}

TEST(ForceCalculator, MaxCutoffMatchesSpan) {
    auto calc = lj_calc(1.0, 2.0);
    EXPECT_NEAR(calc.max_cutoff(), 20.0, 1e-14);
}
