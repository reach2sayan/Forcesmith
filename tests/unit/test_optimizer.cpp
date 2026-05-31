#include "potfit/force/pair_force.hpp"
#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"
#include "potfit/potentials/analytic_potential.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace potfit;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build a two-atom config where reference forces and energy are set from LJ(eps, sigma).
static Configuration make_lj_dimer(double eps, double sigma, double r) {
    Configuration cfg;
    cfg.bc     = PeriodicBC(100.0 * Mat3::Identity());
    cfg.weight = 1.0;

    Atom a0, a1;
    a0.type = 0;  a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0;  a1.pos = {r,   0.0, 0.0};

    // LJ force: F_i = (dV/dr / r) * dist → for atom 0, dist = (r,0,0), so F_x = dV/dr.
    const double sr6  = std::pow(sigma / r, 6);
    const double dvdr = 4.0 * eps * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;

    a0.force = Vec3(dvdr, 0.0, 0.0);
    a1.force = Vec3(-dvdr, 0.0, 0.0);

    cfg.energy = 4.0 * eps * (sr6 * sr6 - sr6);
    cfg.atoms  = {a0, a1};
    return cfg;
}

// Evaluate the functor residual squared norm at the current param values in model.
static double eval_residual(std::vector<Configuration>& configs,
                             ForceCalculator&            model,
                             double                      energy_weight) {
    PotfitFunctor functor(configs, model, energy_weight);
    Eigen::VectorXd x(functor.inputs());
    std::visit([&](const auto& m){ m.gather_params(x, std::size_t{0}); }, model);

    Eigen::VectorXd fvec(functor.values());
    functor(x, fvec);
    return fvec.squaredNorm();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// Residual should be zero when the potential exactly matches the reference data.
TEST(Optimizer, ZeroResidualAtGroundTruth) {
    const double eps = 1.5, sigma = 2.0, r = 2.5 * sigma;

    auto cfg = make_lj_dimer(eps, sigma, r);
    std::vector<Configuration> configs = {cfg};

    std::vector<Potential> pots;
    pots.emplace_back(LennardJones(eps, sigma, sigma * 0.5, sigma * 5.0));
    ForceCalculator model = make_pair_force_calculator(std::move(pots));

    const double res = eval_residual(configs, model, 1.0);
    EXPECT_LT(res, 1e-20)
        << "Residual at ground truth should be ~0, got " << res;
}

// Optimizer should reduce the residual to near zero starting from a perturbed potential.
TEST(Optimizer, ConvergesFromPerturbedEpsilon) {
    const double eps_true = 1.0, sigma = 2.0, r = 2.5 * sigma;

    auto cfg = make_lj_dimer(eps_true, sigma, r);
    std::vector<Configuration> configs = {cfg};

    // Start from eps = 0.5 (significantly perturbed).
    std::vector<Potential> pots;
    pots.emplace_back(LennardJones(0.5, sigma, sigma * 0.5, sigma * 5.0));
    ForceCalculator model = make_pair_force_calculator(std::move(pots));

    // With energy_weight = 1.0 we have 2 force + 1 energy equation for 2 unknowns (ε, σ).
    OptimizerOptions opts;
    opts.max_iter      = 2000;
    opts.xtol          = 1e-10;
    opts.ftol          = 1e-10;
    opts.energy_weight = 1.0;

    const double res_before = eval_residual(configs, model, opts.energy_weight);
    EXPECT_GT(res_before, 1e-4) << "residual should be large before optimization";

    int status = run_optimizer(configs, model, opts);

    // LM success codes: 1=RelErr, 2=FuncEps, 3=XtolReached, 4=GradEps.
    EXPECT_GE(status, 1) << "LM returned failure code " << status;

    const double res_after = eval_residual(configs, model, opts.energy_weight);
    EXPECT_LT(res_after, 1e-6)
        << "Residual after optimization = " << res_after;
}

// Objective should not increase after a single LM step (basic sanity check).
TEST(Optimizer, ResidualDecreasesOrStays) {
    const double eps = 1.0, sigma = 2.0, r = 2.2 * sigma;

    auto cfg = make_lj_dimer(eps, sigma, r);
    std::vector<Configuration> configs = {cfg};

    std::vector<Potential> pots;
    pots.emplace_back(LennardJones(0.7, sigma * 1.05, sigma * 0.5, sigma * 6.0));
    ForceCalculator model = make_pair_force_calculator(std::move(pots));

    const double res_before = eval_residual(configs, model, 1.0);

    OptimizerOptions opts;
    opts.max_iter      = 5;  // just a few iterations
    opts.energy_weight = 1.0;
    run_optimizer(configs, model, opts);

    const double res_after = eval_residual(configs, model, 1.0);
    EXPECT_LE(res_after, res_before * 1.01)  // allow 1% tolerance for numerical noise
        << "Residual increased after optimization: " << res_before << " -> " << res_after;
}
