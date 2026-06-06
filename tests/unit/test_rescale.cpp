#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/core/rescale.hpp"
#include "forcesmith/force/eam_force.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace forcesmith;

// ── Minimal analytic potential types ─────────────────────────────────────────

struct RepulsivePair : NoBounds<RepulsivePair> {
    double A = 1.0;
    explicit RepulsivePair(double a = 1.0) : A(a) {}
    double eval(double r)  const { return A / std::pow(r, 12); }
    double deriv(double r) const { return -12.0 * A / std::pow(r, 13); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

struct ExpDensity : NoBounds<ExpDensity> {
    double beta = 1.0;
    explicit ExpDensity(double b = 1.0) : beta(b) {}
    double eval(double r)  const { return std::exp(-beta * r); }
    double deriv(double r) const { return -beta * std::exp(-beta * r); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    int  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, int) const {}
    void scatter_params(const Eigen::VectorXd&, int) {}
};

// F(ρ) = −c × √ρ  (Finnis-Sinclair style — has non-zero slope everywhere)
struct SqrtEmbedding : NoBounds<SqrtEmbedding> {
    double c = 1.0;
    explicit SqrtEmbedding(double cc = 1.0) : c(cc) {}
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
    a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0; a1.pos = {r,   0.0, 0.0};
    cfg.atoms = {a0, a1};
    // build_neighbor_list must be called by the caller AFTER the Configuration
    // is at its final storage location. Calling it here and then copying into
    // a std::vector via initializer_list would leave dangling neighbor pointers.
    return cfg;
}

static EAMForceCalculator make_eam_calc(double A = 1.0, double beta = 1.0, double c = 1.0) {
    EAMForceCalculator calc;
    calc.ntypes = 1;
    calc.pair.emplace_back(RepulsivePair{A});
    calc.density.emplace_back(ExpDensity{beta});
    calc.embedding.emplace_back(SqrtEmbedding{c});
    return calc;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(EAMRescale, ComputeRhoRefMatchesDimerDensity) {
    const double r    = 2.5;
    const double beta = 1.0;
    auto calc = make_eam_calc(0.0, beta, 1.0);
    std::vector<Configuration> cfgs(1, make_dimer(r));
    build_neighbor_list(cfgs[0], r * 2.0);

    const auto rho_ref = compute_rho_ref(calc, cfgs);

    ASSERT_EQ(static_cast<int>(rho_ref.size()), 1);
    // Each atom has one neighbor at distance r → ρ = exp(−β r)
    EXPECT_NEAR(rho_ref[0], std::exp(-beta * r), 1e-12);
}

TEST(EAMRescale, EmbedShiftZerosDerivAtRef) {
    const double r    = 2.5;
    const double beta = 1.0;
    const double c    = 1.5;
    auto calc = make_eam_calc(1.0, beta, c);
    std::vector<Configuration> cfgs(1, make_dimer(r));
    build_neighbor_list(cfgs[0], r * 2.0);

    const auto rho_ref = compute_rho_ref(calc, cfgs);
    embed_shift(calc, rho_ref);

    // F′(rho_ref) must be zero after embed_shift
    EXPECT_NEAR(calc.embedding[0].deriv(rho_ref[0]), 0.0, 1e-12);
}

TEST(EAMRescale, EmbedShiftPreservesEnergy) {
    const double r = 2.5;
    auto calc = make_eam_calc(1.0, 1.0, 1.5);
    std::vector<Configuration> cfgs(1, make_dimer(r));
    build_neighbor_list(cfgs[0], r * 2.0);

    calc.eval_forces(cfgs[0]);
    const double E_before    = cfgs[0].calc_energy;
    const double rho_ref_val = cfgs[0].atoms[0].rho;

    const std::vector<double> rho_ref_vec = {rho_ref_val};
    embed_shift(calc, rho_ref_vec);

    calc.eval_forces(cfgs[0]);
    const double E_after = cfgs[0].calc_energy;

    // Gauge transformation preserves total energy exactly
    EXPECT_NEAR(E_after, E_before, 1e-10 * std::abs(E_before) + 1e-14);
}

TEST(EAMRescale, RescaleEAMZerosEmbeddingAtRef) {
    const double r = 2.5;
    auto calc = make_eam_calc(1.0, 1.0, 1.5);
    std::vector<Configuration> cfgs(1, make_dimer(r));
    build_neighbor_list(cfgs[0], r * 2.0);

    rescale_eam(calc, cfgs);

    // Re-evaluate to get the post-rescale reference density
    const auto rho_ref = compute_rho_ref(calc, cfgs);

    EXPECT_NEAR(calc.embedding[0].eval(rho_ref[0]), 0.0, 1e-12);
}

TEST(EAMRescale, RescaleEAMZerosEmbeddingDerivAtRef) {
    const double r = 2.5;
    auto calc = make_eam_calc(1.0, 1.0, 1.5);
    std::vector<Configuration> cfgs(1, make_dimer(r));
    build_neighbor_list(cfgs[0], r * 2.0);

    rescale_eam(calc, cfgs);

    const auto rho_ref = compute_rho_ref(calc, cfgs);

    EXPECT_NEAR(calc.embedding[0].deriv(rho_ref[0]), 0.0, 1e-12);
}

TEST(EAMRescale, RescaleEAMPairEnergyCompensated) {
    // Verify that after embed_shift, forces are consistent with the adjusted energy.
    // F_x ≈ −dE/dx by central finite difference.
    const double r  = 2.5;
    const double dr = 1e-5;

    auto calc = make_eam_calc(1.0, 1.0, 1.5);

    // Apply full rescaling
    std::vector<Configuration> cfgs_ref(1, make_dimer(r));
    build_neighbor_list(cfgs_ref[0], r * 2.0);
    rescale_eam(calc, cfgs_ref);

    auto energy_at = [&](double x0) {
        Configuration c = make_dimer(r);
        c.atoms[0].pos[0] = x0;
        build_neighbor_list(c, r * 4.0);
        calc.eval_forces(c);
        return c.calc_energy;
    };

    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, r * 4.0);
    calc.eval_forces(cfg);

    const double fx_fd   = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;
    const double fx_calc = cfg.atoms[0].calc_force[0];
    EXPECT_NEAR(fx_calc, fx_fd, 1e-6 * std::abs(fx_calc) + 1e-10);
}
