#include "potfit/core/neighbor_list.hpp"
#include "potfit/force/stiweb_force.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace potfit;

// ── Fixtures ──────────────────────────────────────────────────────────────────

// Standard Stillinger-Weber (1985) Si parameters.
// A and lambda are pre-multiplied by ε = 2.1683 eV.
static SWParams si_sw() {
    SWParams p;
    p.A      = 7.049556277 * 2.1683;  // 15.2820 eV
    p.B      = 0.6022245584;
    p.p      = 4.0;
    p.q      = 0.0;
    p.a      = 1.80;
    p.sigma  = 2.0951;                // Å
    p.lambda = 21.0 * 2.1683;         // 45.534 eV
    p.gamma  = 1.20;
    return p;
}

static StiwebForceCalculator make_calc(const SWParams& p = si_sw()) {
    StiwebForceCalculator calc;
    calc.params.reserve(1);
    calc.params.emplace_back(p);
    return calc;
}

// SW cutoff for Si: a × σ = 1.80 × 2.0951 = 3.771 Å
static constexpr double SI_RCUT = 1.80 * 2.0951;

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
    a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0; a1.pos = {r,   0.0, 0.0};
    a2.type = 0; a2.pos = {r * 0.5, r * std::sqrt(3.0) / 2.0, 0.0};
    cfg.atoms = {a0, a1, a2};
    return cfg;
}

// 3-atom configuration: atom 0 at origin, atoms 1 and 2 at distance r from
// atom 0, separated by the tetrahedral angle (cos θ = −1/3).
// v3 at atom 0 as center is exactly zero.
static Configuration make_tetrahedral_angle(double r) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1, a2;
    const double cos_tet = -1.0 / 3.0;                 // cos(109.47°)
    const double sin_tet = std::sqrt(1.0 - cos_tet * cos_tet); // sin(109.47°)
    a0.type = 0; a0.pos = {0.0,             0.0, 0.0};
    a1.type = 0; a1.pos = {r,               0.0, 0.0};
    a2.type = 0; a2.pos = {r * cos_tet, r * sin_tet, 0.0};
    cfg.atoms = {a0, a1, a2};
    return cfg;
}

// Regular tetrahedron, edge length r.
// All pairwise distances equal r, all angles equal 109.47°.
static Configuration make_tetrahedron(double r) {
    // Use known regular-tetrahedron vertex coordinates scaled to edge length r.
    // The four vertices of a tetrahedron inscribed in a unit cube:
    //   (+1,+1,+1), (+1,-1,-1), (-1,+1,-1), (-1,-1,+1)  (edge = 2√2)
    // Scale so edge = r: multiply by r/(2√2).
    const double s = r / (2.0 * std::sqrt(2.0));
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1, a2, a3;
    a0.type = 0; a0.pos = { s,  s,  s};
    a1.type = 0; a1.pos = { s, -s, -s};
    a2.type = 0; a2.pos = {-s,  s, -s};
    a3.type = 0; a3.pos = {-s, -s,  s};
    cfg.atoms = {a0, a1, a2, a3};
    return cfg;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(StiwebForce, DimerEnergyIsNegative) {
    // At r ≈ 2.35 Å the 2-body term should be attractive (negative energy).
    const double r = 2.35;
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);

    auto calc = make_calc();
    calc.eval_forces(cfg);

    EXPECT_LT(cfg.calc_energy, 0.0);
    EXPECT_TRUE(std::isfinite(cfg.calc_energy));
}

TEST(StiwebForce, DimerNewtonThirdLaw) {
    const double r = 2.35;
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);

    make_calc().eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force + cfg.atoms[1].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-11);
}

TEST(StiwebForce, DimerForceConsistentWithEnergyFD) {
    const double r  = 2.35;
    const double dr = 1e-6;
    auto calc = make_calc();

    auto energy_at = [&](double x0) {
        auto cfg = make_dimer(r);
        cfg.atoms[0].pos[0] = x0;
        build_neighbor_list(cfg, SI_RCUT + 0.01);
        calc.eval_forces(cfg);
        return cfg.calc_energy;
    };

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);
    calc.eval_forces(cfg);

    const double fx_fd   = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];
    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_fd) + 1e-10);
}

TEST(StiwebForce, DimerEnergyMatchesAnalytic) {
    // v2(r) = A [B(σ/r)^4 − 1] exp(σ/(r − aσ))  (q=0, p=4)
    // E_dimer = v2(r)  (full neighbor list: 2 entries × 0.5 = 1)
    const double r = 2.35;
    const SWParams p = si_sw();

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);
    make_calc(p).eval_forces(cfg);

    const double rcut = p.a * p.sigma;
    const double sr   = p.sigma / r;
    const double v2   = p.A * (p.B * std::pow(sr, p.p) - 1.0)
                      * std::exp(p.sigma / (r - rcut));
    EXPECT_NEAR(cfg.calc_energy, v2, 1e-10 * std::abs(v2));
}

TEST(StiwebForce, BeyondCutoffZeroContribution) {
    const double r = SI_RCUT + 0.01;  // just outside cutoff
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, SI_RCUT + 0.5);

    make_calc().eval_forces(cfg);

    EXPECT_NEAR(cfg.calc_energy, 0.0, 1e-14);
    EXPECT_NEAR(cfg.atoms[0].calc_force.norm(), 0.0, 1e-14);
}

TEST(StiwebForce, TriangleNewtonThirdLaw) {
    const double r = 2.4;
    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);

    make_calc().eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force
                   + cfg.atoms[1].calc_force
                   + cfg.atoms[2].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-11);
}

TEST(StiwebForce, TriangleForceConsistentWithEnergyFD) {
    const double r  = 2.4;
    const double dr = 1e-6;
    auto calc = make_calc();

    auto energy_at = [&](double x0) {
        auto cfg = make_triangle(r);
        cfg.atoms[0].pos[0] = x0;
        build_neighbor_list(cfg, SI_RCUT + 0.01);
        calc.eval_forces(cfg);
        return cfg.calc_energy;
    };

    auto cfg = make_triangle(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);
    calc.eval_forces(cfg);

    const double fx_fd   = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];
    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_fd) + 1e-10);
}

TEST(StiwebForce, TetrahedralAngle_IsEnergyMinimum) {
    // At cos θ = −1/3, w(cos θ) = λ(cos θ + 1/3)² = 0 (minimum of v3).
    // Moving atom 2 off the tetrahedral angle must increase the total energy.
    // Atoms 1 and 2 are r_12 = r√(8/3) ≈ 3.92 Å apart (beyond SI_RCUT = 3.77 Å),
    // so the only 3-body triplet is at atom 0 as center.
    const double r = 2.4;
    SWParams p_small = si_sw();
    p_small.lambda = 1.0;  // isolate 3-body effect; sign matters, not magnitude
    auto calc = make_calc(p_small);

    auto cfg_tet = make_tetrahedral_angle(r);
    build_neighbor_list(cfg_tet, SI_RCUT + 0.01);
    calc.eval_forces(cfg_tet);
    const double e_tet = cfg_tet.calc_energy;

    // Perturb atom 2 slightly off the tetrahedral angle.
    auto cfg_off = make_tetrahedral_angle(r);
    cfg_off.atoms[2].pos[1] += 0.05;
    build_neighbor_list(cfg_off, SI_RCUT + 0.01);
    calc.eval_forces(cfg_off);
    const double e_off = cfg_off.calc_energy;

    // v3 is zero at tetrahedral angle and positive elsewhere → energy increases.
    EXPECT_LT(e_tet, e_off);
}

TEST(StiwebForce, TetrahedronNewtonThirdLaw) {
    const double r = 2.35;
    auto cfg = make_tetrahedron(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);

    make_calc().eval_forces(cfg);

    Vec3 net = Vec3::Zero();
    for (const auto& a : cfg.atoms) net += a.calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-10);
}

TEST(StiwebForce, TetrahedronForceConsistentWithEnergyFD) {
    const double r  = 2.35;
    const double dr = 1e-6;
    auto calc = make_calc();

    auto energy_at = [&](double x0) {
        auto cfg = make_tetrahedron(r);
        cfg.atoms[0].pos[0] = x0;
        build_neighbor_list(cfg, SI_RCUT + 0.01);
        calc.eval_forces(cfg);
        return cfg.calc_energy;
    };

    auto cfg = make_tetrahedron(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);
    calc.eval_forces(cfg);

    const double x0     = cfg.atoms[0].pos[0];
    const double fx_fd  = -(energy_at(x0 + dr / 2.0) - energy_at(x0 - dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];
    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_fd) + 1e-10);
}

TEST(StiwebForce, ForcesIdempotent) {
    const double r = 2.35;
    auto cfg = make_tetrahedron(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);

    auto calc = make_calc();
    calc.eval_forces(cfg);
    const Vec3   f0 = cfg.atoms[0].calc_force;
    const double e0 = cfg.calc_energy;

    calc.eval_forces(cfg);
    EXPECT_NEAR((cfg.atoms[0].calc_force - f0).norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.calc_energy, e0, 1e-14);
}

TEST(StiwebForce, StressSymmetric) {
    const double r = 2.35;
    auto cfg = make_tetrahedron(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);

    make_calc().eval_forces(cfg);

    const SymTens& s = cfg.calc_stress;
    EXPECT_NEAR(s(0,1), s(1,0), 1e-12);
    EXPECT_NEAR(s(0,2), s(2,0), 1e-12);
    EXPECT_NEAR(s(1,2), s(2,1), 1e-12);
}
