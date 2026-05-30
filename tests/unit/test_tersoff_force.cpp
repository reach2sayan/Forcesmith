#include "potfit/core/neighbor_list.hpp"
#include "potfit/force/tersoff_force.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace potfit;

// ── Test fixtures ─────────────────────────────────────────────────────────────

// Standard Tersoff (1988) Si parameters.
static TersoffParams si_tersoff() {
    TersoffParams p;
    p.A      = 1830.8;     // eV
    p.B      = 471.18;     // eV
    p.lambda = 2.4799;     // 1/Å
    p.mu     = 1.7322;     // 1/Å
    p.beta   = 1.1e-6;
    p.n      = 0.78734;
    p.c      = 1.0039e5;
    p.d      = 16.217;
    p.h      = -0.59825;
    p.R      = 2.7;        // Å
    p.S      = 3.0;        // Å
    return p;
}

static TersoffForceCalculator make_calc(const TersoffParams& p = si_tersoff()) {
    TersoffForceCalculator calc;
    calc.ntypes = 1;
    calc.params.push_back(p);
    return calc;
}

static Configuration make_dimer(double r) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1;
    a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0; a1.pos = {r,   0.0, 0.0};
    cfg.atoms = {a0, a1};
    return cfg;
}

// Equilateral triangle, side length r.
static Configuration make_triangle(double r) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1, a2;
    a0.type = 0; a0.pos = {0.0,           0.0,                       0.0};
    a1.type = 0; a1.pos = {r,              0.0,                       0.0};
    a2.type = 0; a2.pos = {r * 0.5, r * std::sqrt(3.0) / 2.0, 0.0};
    cfg.atoms = {a0, a1, a2};
    return cfg;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(TersoffForce, DimerEnergyIsNegative) {
    // At the Si equilibrium bond length (~2.35 Å), the dimer energy must be < 0.
    const double r = 2.35;
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, 3.5);

    auto calc = make_calc();
    calc.eval_forces(cfg);

    EXPECT_LT(cfg.calc_energy, 0.0);
    EXPECT_TRUE(std::isfinite(cfg.calc_energy));
}

TEST(TersoffForce, DimerNewtonThirdLaw) {
    const double r = 2.35;
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, 3.5);

    make_calc().eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force + cfg.atoms[1].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-11);
}

TEST(TersoffForce, DimerForceConsistentWithEnergyFD) {
    // F_x on atom 0 ≈ −dE/dx₀ via central finite difference (h = 1e-6 Å).
    const double r  = 2.35;
    const double dr = 1e-6;
    auto calc = make_calc();

    auto energy_at = [&](double x0) {
        auto cfg = make_dimer(r);
        cfg.atoms[0].pos[0] = x0;
        build_neighbor_list(cfg, 3.5);
        calc.eval_forces(cfg);
        return cfg.calc_energy;
    };

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, 3.5);
    calc.eval_forces(cfg);

    const double fx_fd   = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];

    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_fd) + 1e-10);
}

TEST(TersoffForce, DimerEnergyMatchesAnalytic) {
    // For a dimer ζ = 0, b = 1 (no angular neighbors).
    // E = fc(r)[A exp(−λ r) − B exp(−μ r)]
    const double r = 2.35;
    const TersoffParams p = si_tersoff();

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, 3.5);
    make_calc(p).eval_forces(cfg);

    // fc = 1 since r < R = 2.7
    const double E_analytic = p.A * std::exp(-p.lambda * r)
                            - p.B * std::exp(-p.mu     * r);
    EXPECT_NEAR(cfg.calc_energy, E_analytic, 1e-10 * std::abs(E_analytic));
}

TEST(TersoffForce, TriangleNewtonThirdLaw) {
    // 3-body forces must also satisfy ΣF_i = 0.
    const double r = 2.4;
    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, 3.5);

    make_calc().eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force
                   + cfg.atoms[1].calc_force
                   + cfg.atoms[2].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-10);
}

TEST(TersoffForce, TriangleForceConsistentWithEnergyFD) {
    // F_x on atom 0 ≈ −dE/dx₀ for a 3-atom system (exercises 3-body term).
    const double r  = 2.4;
    const double dr = 1e-6;
    auto calc = make_calc();

    auto energy_at = [&](double x0) {
        auto cfg = make_triangle(r);
        cfg.atoms[0].pos[0] = x0;
        build_neighbor_list(cfg, 3.5);
        calc.eval_forces(cfg);
        return cfg.calc_energy;
    };

    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, 3.5);
    calc.eval_forces(cfg);

    const double fx_fd   = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];

    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_fd) + 1e-10);
}

TEST(TersoffForce, TriangleForcesIdempotent) {
    const double r = 2.4;
    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, 3.5);

    auto calc = make_calc();
    calc.eval_forces(cfg);
    const Vec3   f0 = cfg.atoms[0].calc_force;
    const double e0 = cfg.calc_energy;

    calc.eval_forces(cfg);
    EXPECT_NEAR((cfg.atoms[0].calc_force - f0).norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.calc_energy, e0, 1e-14);
}

TEST(TersoffForce, BeyondCutoffZeroContribution) {
    // Atom pair beyond cutoff S = 3.0 should give zero energy and forces.
    const double r = 3.5; // > S = 3.0
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, 5.0);

    make_calc().eval_forces(cfg);

    EXPECT_NEAR(cfg.calc_energy, 0.0, 1e-14);
    EXPECT_NEAR(cfg.atoms[0].calc_force.norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.atoms[1].calc_force.norm(), 0.0, 1e-14);
}

TEST(TersoffForce, StressSymmetric) {
    const double r = 2.4;
    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, 3.5);

    make_calc().eval_forces(cfg);

    const SymTens& s = cfg.calc_stress;
    EXPECT_NEAR(s(0,1), s(1,0), 1e-12);
    EXPECT_NEAR(s(0,2), s(2,0), 1e-12);
    EXPECT_NEAR(s(1,2), s(2,1), 1e-12);
}
