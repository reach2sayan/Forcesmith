#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/eam_force.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace forcesmith;

// ── Minimal analytic potential types ─────────────────────────────────────────

struct RepulsivePair : NoBounds<RepulsivePair>, NoSiteCache<RepulsivePair>, NoRawParamAccess,
                     NoParamJacobian {
    double A = 1.0;
    double eval(double r)  const { return A / std::pow(r, 12); }
    double deriv(double r) const { return -12.0 * A / std::pow(r, 13); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    std::size_t  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, std::size_t) const {}
    void scatter_params(const Eigen::VectorXd&, std::size_t) {}
};

struct ExpDensity : NoBounds<ExpDensity>, NoSiteCache<ExpDensity>, NoRawParamAccess,
                     NoParamJacobian {
    double beta = 1.0;
    double eval(double r)  const { return std::exp(-beta * r); }
    double deriv(double r) const { return -beta * std::exp(-beta * r); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    std::size_t  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, std::size_t) const {}
    void scatter_params(const Eigen::VectorXd&, std::size_t) {}
};

struct SqrtEmbedding : NoBounds<SqrtEmbedding>, NoSiteCache<SqrtEmbedding>, NoRawParamAccess,
                     NoParamJacobian {
    double c = 1.0;
    double eval(double rho)  const { return -c * std::sqrt(rho); }
    double deriv(double rho) const { return -c / (2.0 * std::sqrt(rho)); }
    std::pair<double,double> span() const { return {1e-10, 1e6}; }
    std::size_t  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, std::size_t) const {}
    void scatter_params(const Eigen::VectorXd&, std::size_t) {}
};

// Dipole: u(r) = exp(-r)
struct ExpDipole : NoBounds<ExpDipole>, NoSiteCache<ExpDipole>, NoRawParamAccess,
                     NoParamJacobian {
    double alpha = 1.0;
    double eval(double r)  const { return std::exp(-alpha * r); }
    double deriv(double r) const { return -alpha * std::exp(-alpha * r); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    std::size_t  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, std::size_t) const {}
    void scatter_params(const Eigen::VectorXd&, std::size_t) {}
};

// Quadrupole: w(r) = exp(-2r) (faster decay)
struct ExpQuadrupole : NoBounds<ExpQuadrupole>, NoSiteCache<ExpQuadrupole>, NoRawParamAccess,
                     NoParamJacobian {
    double gamma = 2.0;
    double eval(double r)  const { return std::exp(-gamma * r); }
    double deriv(double r) const { return -gamma * std::exp(-gamma * r); }
    std::pair<double,double> span() const { return {0.1, 20.0}; }
    std::size_t  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, std::size_t) const {}
    void scatter_params(const Eigen::VectorXd&, std::size_t) {}
};

struct ZeroPot : NoBounds<ZeroPot>, NoSiteCache<ZeroPot>, NoRawParamAccess,
                     NoParamJacobian {
    double eval(double)  const { return 0.0; }
    double deriv(double) const { return 0.0; }
    std::pair<double,double> span() const { return {0.0, 100.0}; }
    std::size_t  param_count() const { return 0; }
    void gather_params(Eigen::VectorXd&, std::size_t) const {}
    void scatter_params(const Eigen::VectorXd&, std::size_t) {}
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static Configuration make_dimer(double r) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1;
    a0.type = 0;  a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0;  a1.pos = {r,   0.0, 0.0};
    cfg.atoms = {a0, a1};
    return cfg;
}

static ADPForceCalculator make_adp_calc() {
    ADPForceCalculator calc;
    calc.ntypes = 1;
    calc.pair.emplace_back(RepulsivePair{});
    calc.density.emplace_back(ExpDensity{});
    calc.embedding.emplace_back(SqrtEmbedding{});
    calc.dipole.emplace_back(ExpDipole{});
    calc.quadrupole.emplace_back(ExpQuadrupole{});
    return calc;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(ADPForce, NewtonThirdLaw) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_adp_calc();
    calc.eval_forces(cfg);

    const Vec3 net = cfg.atoms[0].calc_force + cfg.atoms[1].calc_force;
    EXPECT_NEAR(net.norm(), 0.0, 1e-11);
}

TEST(ADPForce, ZeroUWReducesToEAM) {
    // With u=0, w=0 the ADP forces must equal EAM forces for the same EAM pots.
    const double r = 2.5;
    auto cfg_adp = make_dimer(r);
    auto cfg_eam = make_dimer(r);
    build_neighbor_list(cfg_adp, r * 2.0);
    build_neighbor_list(cfg_eam, r * 2.0);

    ADPForceCalculator adp_calc;
    adp_calc.ntypes = 1;
    adp_calc.pair.emplace_back(RepulsivePair{});
    adp_calc.density.emplace_back(ExpDensity{});
    adp_calc.embedding.emplace_back(SqrtEmbedding{});
    adp_calc.dipole.emplace_back(ZeroPot{});
    adp_calc.quadrupole.emplace_back(ZeroPot{});
    adp_calc.eval_forces(cfg_adp);

    EAMForceCalculator eam_calc;
    eam_calc.ntypes = 1;
    eam_calc.pair.emplace_back(RepulsivePair{});
    eam_calc.density.emplace_back(ExpDensity{});
    eam_calc.embedding.emplace_back(SqrtEmbedding{});
    eam_calc.eval_forces(cfg_eam);

    EXPECT_NEAR(cfg_adp.calc_energy, cfg_eam.calc_energy, 1e-12);
    EXPECT_NEAR((cfg_adp.atoms[0].calc_force - cfg_eam.atoms[0].calc_force).norm(),
                0.0, 1e-12);
}

TEST(ADPForce, DipoleAccumulation) {
    // For a dimer along x, μ_0 = u(r) × (r, 0, 0), μ_1 = u(r) × (-r, 0, 0).
    const double r   = 2.5;
    const double u_r = std::exp(-r);   // ExpDipole with alpha=1
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_adp_calc();
    calc.eval_forces(cfg);

    EXPECT_NEAR(cfg.atoms[0].mu[0],  r * u_r, 1e-12);
    EXPECT_NEAR(cfg.atoms[0].mu[1],  0.0, 1e-14);
    EXPECT_NEAR(cfg.atoms[1].mu[0], -r * u_r, 1e-12);
}

TEST(ADPForce, QuadrupoleAccumulation) {
    // For a dimer along x, λ_i = w(r) × d⊗d where d = (±r, 0, 0).
    // Both atoms get λ(0,0) = r² × w(r), all other components = 0.
    const double r   = 2.5;
    const double w_r = std::exp(-2.0 * r);  // ExpQuadrupole with gamma=2
    auto cfg = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_adp_calc();
    calc.eval_forces(cfg);

    EXPECT_NEAR(cfg.atoms[0].lambda(0,0), r * r * w_r, 1e-12);
    EXPECT_NEAR(cfg.atoms[0].lambda(1,1), 0.0, 1e-14);
    EXPECT_NEAR(cfg.atoms[0].lambda(2,2), 0.0, 1e-14);
    EXPECT_NEAR(cfg.atoms[0].lambda(0,1), 0.0, 1e-14);
    // atom 1 has d = (-r, 0, 0), so d⊗d = same (sign squares to +)
    EXPECT_NEAR(cfg.atoms[1].lambda(0,0), r * r * w_r, 1e-12);
}

TEST(ADPForce, EnergyIsFinite) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_adp_calc();
    calc.eval_forces(cfg);

    EXPECT_TRUE(std::isfinite(cfg.calc_energy));
}

TEST(ADPForce, ForcesIdempotent) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_adp_calc();
    calc.eval_forces(cfg);
    const Vec3   f0_first = cfg.atoms[0].calc_force;
    const double e_first  = cfg.calc_energy;

    calc.eval_forces(cfg);
    EXPECT_NEAR((cfg.atoms[0].calc_force - f0_first).norm(), 0.0, 1e-14);
    EXPECT_NEAR(cfg.calc_energy, e_first, 1e-14);
}

TEST(ADPForce, ForceConsistentWithEnergyFD) {
    // F_x ≈ -dE/dx, verified by central finite differences.
    const double r  = 2.5;
    const double dr = 1e-5;
    auto calc = make_adp_calc();

    auto energy_at = [&](double x0) {
        auto c = make_dimer(r);
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
    EXPECT_NEAR(fx_calc, fx_fd, 1e-5 * std::abs(fx_fd) + 1e-10);
}

TEST(ADPForce, StressSymmetric) {
    const double r = 2.5;
    auto cfg  = make_dimer(r);
    build_neighbor_list(cfg, r * 2.0);

    auto calc = make_adp_calc();
    calc.eval_forces(cfg);

    const SymTens& s = cfg.calc_stress;
    EXPECT_NEAR(s(0,1), s(1,0), 1e-14);
    EXPECT_NEAR(s(0,2), s(2,0), 1e-14);
    EXPECT_NEAR(s(1,2), s(2,1), 1e-14);
}
