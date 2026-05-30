#include "potfit/core/neighbor_list.hpp"
#include "potfit/force/eam_force.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace potfit;

// ── Minimal analytic potential types for EAM testing ─────────────────────────

// Pair repulsion: φ(r) = A/r^12
struct RepulsivePair {
    double A = 1.0;
    double eval(double r)  const { return A / std::pow(r, 12); }
    double deriv(double r) const { return -12.0 * A / std::pow(r, 13); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// Density: g(r) = exp(-β r)  (decay, always non-negative)
struct ExpDensity {
    double beta = 1.0;
    double eval(double r)  const { return std::exp(-beta * r); }
    double deriv(double r) const { return -beta * std::exp(-beta * r); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// Embedding: F(ρ) = -c × √ρ   (attractive, like Finnis-Sinclair style)
struct SqrtEmbedding {
    double c = 1.0;
    double eval(double rho)  const { return -c * std::sqrt(rho); }
    double deriv(double rho) const { return -c / (2.0 * std::sqrt(rho)); }
    std::pair<double,double> span() const { return {1e-10, 1e6}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// ── Test helpers ──────────────────────────────────────────────────────────────

static Configuration make_dimer(double r) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1;
    a0.type = 0;  a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0;  a1.pos = {r,   0.0, 0.0};
    cfg.atoms = {a0, a1};
    return cfg;
}

// Single-type EAM calculator with the analytic potentials above.
static EAMForceCalculator make_eam_calc(double A = 1.0, double beta = 1.0, double c = 1.0) {
    EAMForceCalculator calc;
    calc.ntypes = 1;
    calc.pair_pots.emplace_back(RepulsivePair{A});
    calc.rho_pots.emplace_back(ExpDensity{beta});
    calc.F_pots.emplace_back(SqrtEmbedding{c});
    return calc;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(EAMForce, NewtonThirdLaw) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);  // no pair pot assigned via neighbor list

    auto calc = make_eam_calc();
    calc.eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force + cfg.atoms[1].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-12);
}

TEST(EAMForce, EnergyIsFiniteAndNegative) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_eam_calc();
    calc.eval_forces(cfg);

    EXPECT_TRUE(std::isfinite(cfg.calc_energy));
    // Embedding energy = -2 × c × √(exp(-β r)) dominates for small A
    EXPECT_LT(cfg.calc_energy, 0.0);
}

TEST(EAMForce, DensityAccumulatedCorrectly) {
    const double r    = 2.5;
    const double beta = 1.0;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_eam_calc(0.0, beta, 1.0);  // no pair repulsion
    calc.eval_forces(cfg);

    // Each atom has exactly one neighbor at distance r.
    // ρ_i = exp(-β r)
    const double rho_expected = std::exp(-beta * r);
    EXPECT_NEAR(cfg.atoms[0].rho, rho_expected, 1e-12);
    EXPECT_NEAR(cfg.atoms[1].rho, rho_expected, 1e-12);
}

TEST(EAMForce, EmbeddingEnergyMatchesAnalytic) {
    const double r    = 2.5;
    const double beta = 1.0;
    const double c    = 2.0;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_eam_calc(0.0, beta, c);
    calc.eval_forces(cfg);

    // E_emb = 2 × F(ρ) = 2 × (-c × √(exp(-β r)))
    const double rho      = std::exp(-beta * r);
    const double e_expect = 2.0 * (-c * std::sqrt(rho));
    EXPECT_NEAR(cfg.calc_energy, e_expect, 1e-10);
}

TEST(EAMForce, ForcesIdempotent) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_eam_calc();
    calc.eval_forces(cfg);
    const Vec3   f0_first = cfg.atoms[0].calc_force;
    const double e_first  = cfg.calc_energy;

    calc.eval_forces(cfg);
    EXPECT_NEAR((cfg.atoms[0].calc_force - f0_first).norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.calc_energy, e_first, 1e-14);
}

TEST(EAMForce, ForceConsistentWithEnergyFD) {
    // Verify F_x ≈ -dE/dx by finite difference on atom 0's x-coordinate.
    const double r  = 2.5;
    const double dr = 1e-5;

    auto calc = make_eam_calc();

    auto energy_at = [&](double x) {
        auto c = make_dimer(r);
        c.atoms[0].pos[0] = x;
        build_neighbor_list(c, r * 4.0);
        calc.eval_forces(c);
        return c.calc_energy;
    };

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, r * 4.0);
    calc.eval_forces(cfg);

    const double fx_fd  = -(energy_at(dr/2) - energy_at(-dr/2)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];
    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_calc));
}

TEST(EAMForce, StressSymmetric) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_eam_calc();
    calc.eval_forces(cfg);

    const SymTens& s = cfg.calc_stress;
    EXPECT_NEAR(s(0,1), s(1,0), 1e-14);
    EXPECT_NEAR(s(0,2), s(2,0), 1e-14);
    EXPECT_NEAR(s(1,2), s(2,1), 1e-14);
}
