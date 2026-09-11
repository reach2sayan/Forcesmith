#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/force/angular_force.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <ranges>

using namespace forcesmith;

// ── Minimal analytic potential types ─────────────────────────────────────────

struct LJPair : NoBounds<LJPair>, NoSiteCache<LJPair>, NoRawParamAccess,
                     NoParamJacobian {
    double eps = 1.0, sigma = 1.0;
    double eval(double r)  const {
        const double sr6 = std::pow(sigma/r, 6);
        return 4.0*eps*(sr6*sr6 - sr6);
    }
    double deriv(double r) const {
        const double sr6 = std::pow(sigma/r, 6);
        return 4.0*eps*(-12.0*sr6*sr6 + 6.0*sr6)/r;
    }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// f(r) = exp(-r): smooth radial modulation that goes to 0 at large r
struct ExpMod : NoBounds<ExpMod>, NoSiteCache<ExpMod>, NoRawParamAccess,
                     NoParamJacobian {
    double alpha = 1.0;
    double eval(double r)  const { return std::exp(-alpha * r); }
    double deriv(double r) const { return -alpha * std::exp(-alpha * r); }
    std::pair<double,double> span() const { return {0.0, 20.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// g(cos θ) = (cos θ − cos θ₀)²: penalises deviation from preferred angle
struct CosAngle : NoBounds<CosAngle>, NoSiteCache<CosAngle>, NoRawParamAccess,
                     NoParamJacobian {
    double cos0 = -0.5;  // preferred angle ~120°
    double k    = 1.0;
    double eval(double c)  const { return k * (c - cos0) * (c - cos0); }
    double deriv(double c) const { return 2.0 * k * (c - cos0); }
    std::pair<double,double> span() const { return {-1.0, 1.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

struct ZeroPot : NoBounds<ZeroPot>, NoSiteCache<ZeroPot>, NoRawParamAccess,
                     NoParamJacobian {
    double eval(double)  const { return 0.0; }
    double deriv(double) const { return 0.0; }
    std::pair<double,double> span() const { return {0.0, 100.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Equilateral triangle: atoms at vertices, side length r.
static Configuration make_triangle(double r) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1, a2;
    a0.type = 0; a0.pos = {0.0,           0.0, 0.0};
    a1.type = 0; a1.pos = {r,              0.0, 0.0};
    a2.type = 0; a2.pos = {r / 2.0, r * std::sqrt(3.0) / 2.0, 0.0};
    cfg.atoms = {a0, a1, a2};
    return cfg;
}

static AngularForceCalculator make_ang_calc() {
    AngularForceCalculator calc;
    calc.ntypes = 1;
    calc.pair.emplace_back(LJPair{});
    calc.radial.emplace_back(ExpMod{});
    calc.angular.emplace_back(CosAngle{});
    return calc;
}

// Pair-only: f=0 so three-body vanishes, forces should match PairForceCalculator.
static AngularForceCalculator make_pair_only_calc() {
    AngularForceCalculator calc;
    calc.ntypes = 1;
    calc.pair.emplace_back(LJPair{});
    calc.radial.emplace_back(ZeroPot{});
    calc.angular.emplace_back(CosAngle{});
    return calc;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(AngularForce, NewtonThirdLaw_Triangle) {
    const double r = 2.5;
    auto cfg  = make_triangle(r);
    build_neighbor_list(cfg, r * 1.5);

    auto calc = make_ang_calc();
    calc.eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force
                   + cfg.atoms[1].calc_force
                   + cfg.atoms[2].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-11);
}

TEST(AngularForce, NewtonThirdLaw_Dimer) {
    // No three-body term for a 2-atom system; pair forces must cancel.
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1;
    a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0; a1.pos = {2.5, 0.0, 0.0};
    cfg.atoms = {a0, a1};
    build_neighbor_list(cfg, 5.0);

    auto calc = make_ang_calc();
    calc.eval_forces(cfg);

    EXPECT_NEAR((cfg.atoms[0].calc_force + cfg.atoms[1].calc_force).norm(), 0.0, 1e-12);
}

TEST(AngularForce, ZeroFModZerosThreeBody) {
    // With f=0 the three-body energy vanishes. Forces must equal the pair-only result.
    const double r = 2.5;
    auto cfg_ang  = make_triangle(r);
    auto cfg_pair = make_triangle(r);
    build_neighbor_list(cfg_ang,  r * 1.5);
    build_neighbor_list(cfg_pair, r * 1.5);

    make_pair_only_calc().eval_forces(cfg_ang);

    // Build reference with pair-only PairForceCalculator via zero-f AngularForceCalculator
    make_pair_only_calc().eval_forces(cfg_pair);

    for (const auto& [ang, pair] : std::views::zip(cfg_ang.atoms, cfg_pair.atoms))
        EXPECT_NEAR((ang.calc_force - pair.calc_force).norm(), 0.0, 1e-12);
    EXPECT_NEAR(cfg_ang.calc_energy, cfg_pair.calc_energy, 1e-12);
}

TEST(AngularForce, EnergyIsFinite) {
    const double r = 2.5;
    auto cfg  = make_triangle(r);
    build_neighbor_list(cfg, r * 1.5);

    make_ang_calc().eval_forces(cfg);

    EXPECT_TRUE(std::isfinite(cfg.calc_energy));
}

TEST(AngularForce, ForcesIdempotent) {
    const double r = 2.5;
    auto cfg  = make_triangle(r);
    build_neighbor_list(cfg, r * 1.5);

    auto calc = make_ang_calc();
    calc.eval_forces(cfg);
    const Vec3   f0  = cfg.atoms[0].calc_force;
    const double e0  = cfg.calc_energy;

    calc.eval_forces(cfg);
    EXPECT_NEAR((cfg.atoms[0].calc_force - f0).norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.calc_energy, e0, 1e-14);
}

TEST(AngularForce, ForceConsistentWithEnergyFD) {
    // F_{0,x} ≈ −dE/dx₀ via central finite difference.
    const double r  = 2.5;
    const double dr = 1e-5;
    auto calc = make_ang_calc();

    auto energy_at = [&](double x0) {
        auto c = make_triangle(r);
        c.atoms[0].pos[0] = x0;
        build_neighbor_list(c, r * 3.0);
        calc.eval_forces(c);
        return c.calc_energy;
    };

    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, r * 3.0);
    calc.eval_forces(cfg);

    const double fx_fd   = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];
    EXPECT_NEAR(fx_calc, fx_fd, 1e-5 * std::abs(fx_fd) + 1e-10);
}

TEST(AngularForce, StressSymmetric) {
    const double r = 2.5;
    auto cfg  = make_triangle(r);
    build_neighbor_list(cfg, r * 1.5);

    make_ang_calc().eval_forces(cfg);

    const SymTens& s = cfg.calc_stress;
    EXPECT_NEAR(s(0,1), s(1,0), 1e-13);
    EXPECT_NEAR(s(0,2), s(2,0), 1e-13);
    EXPECT_NEAR(s(1,2), s(2,1), 1e-13);
}
