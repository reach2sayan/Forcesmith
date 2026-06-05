#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/force/stiweb_force.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace forcesmith;

// ── Fixtures ──────────────────────────────────────────────────────────────────

// Standard Stillinger-Weber (1985) Si parameters, in forcesmith's parameterization.
// Textbook form  v2 = Aε[B(σ/r)^p − (σ/r)^q] exp(σ/(r − aσ))
// is converted to forcesmith form  (A'·r^{−p} − B'·r^{−q}) exp(δ/(r − a1)).
static constexpr double SI_EPS       = 2.1683;       // ε (eV)
static constexpr double SI_SIGMA     = 2.0951;       // σ (Å)
static constexpr double SI_A_SW      = 7.049556277;  // A_SW
static constexpr double SI_B_SW      = 0.6022245584; // B_SW
static constexpr double SI_P         = 4.0;
static constexpr double SI_Q         = 0.0;
static constexpr double SI_ACUT      = 1.80;         // cutoff in units of σ
static constexpr double SI_GAMMA_SW  = 1.20;         // γ_SW
static constexpr double SI_LAMBDA_SW = 21.0;         // λ_SW

// 3-body strength λ' = λ_SW·ε (per-triplet; single value for single element).
static const double SI_LAMBDA = SI_LAMBDA_SW * SI_EPS; // 45.534 eV

static SWParams si_sw() {
    SWParams p;
    p.A     = SI_A_SW * SI_EPS * SI_B_SW * std::pow(SI_SIGMA, SI_P);
    p.B     = SI_A_SW * SI_EPS * std::pow(SI_SIGMA, SI_Q);
    p.p     = SI_P;
    p.q     = SI_Q;
    p.delta = SI_SIGMA;
    p.a1    = SI_ACUT * SI_SIGMA;
    p.gamma = SI_GAMMA_SW * SI_SIGMA;
    p.a2    = SI_ACUT * SI_SIGMA;
    return p;
}

static StiwebForceCalculator make_calc(const SWParams& p = si_sw(),
                                       double lambda = SI_LAMBDA) {
    StiwebForceCalculator calc;
    calc.ntypes = 1;
    calc.params.reserve(1);
    calc.params.emplace_back(p);
    calc.lambda = {Param{lambda}}; // single triplet (i=j=k=0)
    return calc;
}

// SW cutoff for Si: a × σ = 1.80 × 2.0951 = 3.771 Å
static constexpr double SI_RCUT = SI_ACUT * SI_SIGMA;

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
    // v2(r) = (A·r^{−p} − B·r^{−q}) exp(δ/(r − a1))
    // E_dimer = v2(r)  (full neighbor list: 2 entries × 0.5 = 1)
    const double r = 2.35;
    const SWParams p = si_sw();

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, SI_RCUT + 0.01);
    make_calc(p).eval_forces(cfg);

    const double v2 = (p.A * std::pow(r, -double(p.p)) -
                       p.B * std::pow(r, -double(p.q))) *
                      std::exp(double(p.delta) / (r - double(p.a1)));
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
    // isolate 3-body effect (λ = 1.0); sign matters, not magnitude
    auto calc = make_calc(si_sw(), 1.0);

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

// Per-triplet λ[i][j][k] indexing for a 2-type system: symmetric in (j,k),
// and central type i selects a distinct block. paircol = 3 → lambda size = 6.
TEST(StiwebForce, LambdaPerTripletIndexing) {
    StiwebForceCalculator calc;
    calc.ntypes = 2;
    calc.params.reserve(2);              // paircol = 3 SWParams entries
    for (int s = 0; s < 3; ++s) calc.params.emplace_back(SWParams{});
    // lambda[i·paircol + slot(j,k)], slot(0,0)=0 slot(0,1)=1 slot(1,1)=2
    calc.lambda = {Param{10.0}, Param{11.0}, Param{12.0},   // i = 0
                   Param{20.0}, Param{21.0}, Param{22.0}};  // i = 1

    // Symmetric in j,k.
    EXPECT_EQ(&calc.lambda_at(0, 0, 1), &calc.lambda_at(0, 1, 0));
    EXPECT_EQ(&calc.lambda_at(1, 0, 1), &calc.lambda_at(1, 1, 0));

    EXPECT_DOUBLE_EQ(double(calc.lambda_at(0, 0, 0)), 10.0);
    EXPECT_DOUBLE_EQ(double(calc.lambda_at(0, 0, 1)), 11.0);
    EXPECT_DOUBLE_EQ(double(calc.lambda_at(0, 1, 1)), 12.0);
    EXPECT_DOUBLE_EQ(double(calc.lambda_at(1, 0, 0)), 20.0);
    EXPECT_DOUBLE_EQ(double(calc.lambda_at(1, 0, 1)), 21.0);
    EXPECT_DOUBLE_EQ(double(calc.lambda_at(1, 1, 1)), 22.0);
}
