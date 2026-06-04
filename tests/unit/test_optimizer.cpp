#include "potfit/force/pair_force.hpp"
#include "potfit/optimization/ipopt_solver.hpp"
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

    a0.ref.force = Vec3(dvdr, 0.0, 0.0);
    a1.ref.force = Vec3(-dvdr, 0.0, 0.0);

    cfg.ref.energy = 4.0 * eps * (sr6 * sr6 - sr6);
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

// Gather the model's current free-parameter values (post-optimization).
static Eigen::VectorXd current_params(ForceCalculator& model) {
    const std::size_t n =
        std::visit([](const auto& m) { return m.param_count(); }, model);
    Eigen::VectorXd x(static_cast<Eigen::Index>(n));
    std::visit([&](const auto& m){ m.gather_params(x, std::size_t{0}); }, model);
    return x;
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
    opts.energy_weight = 1.0;

    const double res_before = eval_residual(configs, model, opts.energy_weight);
    EXPECT_GT(res_before, 1e-4) << "residual should be large before optimization";

    int status = run_optimizer(configs, model, opts,
                               Solver{EigenLMSolver{2000, 1e-10, 1e-10}});

    // LM success codes: 1=RelErr, 2=FuncEps, 3=XtolReached, 4=GradEps.
    EXPECT_GE(status, 1) << "LM returned failure code " << status;

    const double res_after = eval_residual(configs, model, opts.energy_weight);
    EXPECT_LT(res_after, 1e-6)
        << "Residual after optimization = " << res_after;
}

// Ipopt (L-BFGS) should drive the residual to near zero from the same perturbed
// start as the LM test — proving it drops into the Solver façade as a replacement.
TEST(Optimizer, IpoptConvergesFromPerturbedEpsilon) {
    const double eps_true = 1.0, sigma = 2.0, r = 2.5 * sigma;

    auto cfg = make_lj_dimer(eps_true, sigma, r);
    std::vector<Configuration> configs = {cfg};

    // Start from eps = 0.5 (significantly perturbed).
    std::vector<Potential> pots;
    pots.emplace_back(LennardJones(0.5, sigma, sigma * 0.5, sigma * 5.0));
    ForceCalculator model = make_pair_force_calculator(std::move(pots));

    OptimizerOptions opts;
    opts.energy_weight = 1.0;

    const double res_before = eval_residual(configs, model, opts.energy_weight);
    EXPECT_GT(res_before, 1e-4) << "residual should be large before optimization";

    int status = run_optimizer(configs, model, opts,
                               Solver{IpoptSolver{2000, 1e-10}});

    // 1 = SUCCESS, 2 = STOP_AT_ACCEPTABLE_POINT (see IpoptSolver::minimize).
    EXPECT_GE(status, 1) << "Ipopt returned failure code " << status;

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
    opts.energy_weight = 1.0;
    run_optimizer(configs, model, opts,
                  Solver{EigenLMSolver{5}}); // just a few iterations

    const double res_after = eval_residual(configs, model, 1.0);
    EXPECT_LE(res_after, res_before * 1.01)  // allow 1% tolerance for numerical noise
        << "Residual increased after optimization: " << res_before << " -> " << res_after;
}

// ── Bounds ──────────────────────────────────────────────────────────────────
// gather_bounds yields ±∞ for a potential built without explicit bounds, so the
// legacy (bare-value) path stays unconstrained.
TEST(Optimizer, UnboundedParamsGatherInfiniteBounds) {
    Potential lj{LennardJones(1.0, 2.0, 1.0, 10.0)};
    std::vector<Potential> pots;
    pots.push_back(std::move(lj));
    ForceCalculator model = make_pair_force_calculator(std::move(pots));

    const std::size_t n =
        std::visit([](const auto& m) { return m.param_count(); }, model);
    Eigen::VectorXd lo(static_cast<Eigen::Index>(n)),
        hi(static_cast<Eigen::Index>(n));
    std::visit([&](const auto& m){ m.gather_bounds(lo, hi, std::size_t{0}); },
               model);

    for (Eigen::Index i = 0; i < lo.size(); ++i) {
        EXPECT_FALSE(std::isfinite(lo[i])) << "lower[" << i << "] should be -inf";
        EXPECT_FALSE(std::isfinite(hi[i])) << "upper[" << i << "] should be +inf";
    }
}

// Build the perturbed LJ dimer fit but constrain epsilon to [0.6, 0.8], strictly
// below its true value (1.0). Sigma is held fixed so epsilon is the lone free
// parameter — otherwise sigma compensates (the model depends on eps*sigma^6) and
// the bound never binds. With sigma pinned the unconstrained optimum is exactly
// eps=1.0, so a bound-honoring solver must sit at the upper bound 0.8.
static ForceCalculator make_eps_bounded_model(double sigma) {
    Potential lj{LennardJones(0.7, sigma, sigma * 0.5, sigma * 5.0)};
    lj.set_bounds(0, 0.6, 0.8); // index 0 = epsilon
    lj.set_fixed(1, true);      // index 1 = sigma (held at its true value)
    std::vector<Potential> pots;
    pots.push_back(std::move(lj));
    return make_pair_force_calculator(std::move(pots));
}

TEST(Optimizer, IpoptRespectsParameterBounds) {
    const double sigma = 2.0, r = 2.5 * sigma;
    auto cfg = make_lj_dimer(1.0, sigma, r); // true epsilon = 1.0
    std::vector<Configuration> configs = {cfg};
    ForceCalculator model = make_eps_bounded_model(sigma);

    OptimizerOptions opts;
    opts.energy_weight = 1.0;
    int status = run_optimizer(configs, model, opts,
                               Solver{IpoptSolver{2000, 1e-10}});
    EXPECT_GE(status, 1) << "Ipopt returned failure code " << status;

    const Eigen::VectorXd p = current_params(model);
    EXPECT_GE(p[0], 0.6 - 1e-6) << "epsilon below lower bound: " << p[0];
    EXPECT_LE(p[0], 0.8 + 1e-6) << "epsilon above upper bound: " << p[0];
    EXPECT_NEAR(p[0], 0.8, 5e-3) << "constrained optimum should sit at the bound";
}

TEST(Optimizer, DifferentialEvolutionRespectsParameterBounds) {
    const double sigma = 2.0, r = 2.5 * sigma;
    auto cfg = make_lj_dimer(1.0, sigma, r);
    std::vector<Configuration> configs = {cfg};
    ForceCalculator model = make_eps_bounded_model(sigma);

    OptimizerOptions opts;
    opts.energy_weight = 1.0;
    run_optimizer(configs, model, opts, Solver{BoostDESolver{}});

    const Eigen::VectorXd p = current_params(model);
    // DE samples strictly within [lower, upper], so the box must hold exactly.
    EXPECT_GE(p[0], 0.6) << "epsilon below lower bound: " << p[0];
    EXPECT_LE(p[0], 0.8) << "epsilon above upper bound: " << p[0];
    EXPECT_GT(p[0], 0.7) << "epsilon should be driven toward the upper bound";
}
