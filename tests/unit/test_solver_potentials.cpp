// Optimizer coverage: every force model × {Levenberg-Marquardt, Ipopt}.
//
// Each case builds a ground-truth model, labels a handful of configurations
// from it (ref forces + energy), then perturbs a sensitive parameter and runs
// the solver for 10 iterations. The contract is intentionally weak and robust:
// 10 steps need not converge, but a descent solver started from a non-stationary
// point MUST reduce the weighted residual. (Pair × {LM, Ipopt} already lives in
// test_optimizer.cpp; SymmetryFunction × LM in test_ml_fit.cpp.)

#include "potfit/force/force_calculator.hpp"
#include "potfit/optimization/ipopt_solver.hpp"
#include "potfit/optimization/optimizer.hpp"
#include "potfit/optimization/potfit_functor.hpp"
#include "potfit/potentials/analytic_potential.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <variant>
#include <vector>

using namespace potfit;

namespace {

// ── Geometry helpers ──────────────────────────────────────────────────────────

Configuration make_dimer(double r) {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1;
  a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
  a1.type = 0; a1.pos = {r,   0.0, 0.0};
  cfg.atoms = {a0, a1};
  return cfg;
}

// Equilateral triangle, side length r — exercises three-body terms.
Configuration make_triangle(double r) {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1, a2;
  a0.type = 0; a0.pos = {0.0,     0.0,                      0.0};
  a1.type = 0; a1.pos = {r,       0.0,                      0.0};
  a2.type = 0; a2.pos = {r * 0.5, r * std::sqrt(3.0) / 2.0, 0.0};
  cfg.atoms = {a0, a1, a2};
  return cfg;
}

// Small asymmetric 3D cluster (all force components nonzero).
Configuration make_cluster() {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1, a2, a3;
  a0.type = 0; a0.pos = {0.0,  0.0,  0.0};
  a1.type = 0; a1.pos = {2.1,  0.3, -0.2};
  a2.type = 0; a2.pos = {0.4,  2.0,  0.5};
  a3.type = 0; a3.pos = {-1.3, 0.7,  1.9};
  cfg.atoms = {a0, a1, a2, a3};
  return cfg;
}

// ── Residual evaluation (mirrors test_optimizer.cpp) ──────────────────────────

double eval_residual(std::vector<Configuration> &configs, ForceCalculator &model,
                     double energy_weight) {
  PotfitFunctor functor(configs, model, energy_weight);
  Eigen::VectorXd x(functor.inputs());
  std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); }, model);

  Eigen::VectorXd fvec(functor.values());
  functor(x, fvec);
  return fvec.squaredNorm();
}

// Label every config's reference forces/energy from the ground-truth model, then
// run `solver` for a few iterations on `start` and assert the residual dropped.
void expect_reduces(ForceCalculator truth, ForceCalculator start,
                    std::vector<Configuration> configs, double energy_weight,
                    Solver solver, const char *tag) {
  for (auto &cfg : configs) {
    std::visit([&](auto &m) { m.eval_forces(cfg); }, truth);
    for (auto &a : cfg.atoms) {
      a.ref.force = a.calc_force;
    }
    cfg.ref.energy = cfg.calc_energy;
  }

  OptimizerOptions opts;
  opts.energy_weight = energy_weight;

  const double before = eval_residual(configs, start, energy_weight);
  run_optimizer(std::span<Configuration>(configs), start, opts, solver);
  const double after = eval_residual(configs, start, energy_weight);

  EXPECT_GT(before, 0.0) << tag << ": perturbed start should have residual > 0";
  EXPECT_LT(after, before)
      << tag << ": 10 steps should reduce residual (before=" << before
      << " after=" << after << ")";
}

Solver lm() { return Solver{EigenLMSolver{10}}; }       // 10 LM iterations
Solver lmne() { return Solver{NormalEquationsLMSolver{10}}; } // normal-eqns LM
Solver ipopt() { return Solver{IpoptSolver{10}}; }      // 10 Ipopt iterations

// ── Model builders (eps / amplitudes drive the perturbation) ──────────────────

ForceCalculator make_eam(double eps) {
  EAMForceCalculator c;
  c.ntypes = 1;
  c.pair.emplace_back(LennardJones(eps, 2.3, 0.5, 5.5));
  c.density.emplace_back(ExpDecay(1.0, 0.8, 0.5, 5.5));
  c.embedding.emplace_back(SqrtFunc(1.0, 1.0, 1e-8, 1e8)); // F(ρ) domain ≫ cutoff
  return ForceCalculator{std::move(c)};
}

ForceCalculator make_adp(double eps) {
  ADPForceCalculator c;
  c.ntypes = 1;
  c.pair.emplace_back(LennardJones(eps, 2.3, 0.5, 5.5));
  c.density.emplace_back(ExpDecay(1.0, 0.8, 0.5, 5.5));
  c.embedding.emplace_back(SqrtFunc(1.0, 1.0, 1e-8, 1e8));
  c.dipole.emplace_back(ExpDecay(0.5, 1.0, 0.5, 5.5));
  c.quadrupole.emplace_back(ExpDecay(0.3, 1.2, 0.5, 5.5));
  return ForceCalculator{std::move(c)};
}

ForceCalculator make_angular(double eps) {
  AngularForceCalculator c;
  c.ntypes = 1;
  c.pair.emplace_back(LennardJones(eps, 2.3, 0.5, 5.5));
  c.radial.emplace_back(ExpDecay(1.0, 0.8, 0.5, 5.5));
  c.angular.emplace_back(Parabola(1.0, 0.0, 0.0, -1.05, 1.05)); // g(cosθ) = cos²θ
  return ForceCalculator{std::move(c)};
}

ForceCalculator make_tersoff(double A, double B, double lam, double mu) {
  TersoffParams p;
  p.A = A; p.B = B; p.lambda = lam; p.mu = mu;
  p.beta = 1.0; p.n = 1.0; p.c = 1.0; p.d = 1.0; p.h = 0.0;
  p.R = 2.5; p.S = 3.0;
  // Free only the four pair amplitudes/decays; pin the bond-order shape so the
  // sub-problem stays well-scaled for a 10-step finite-difference solve.
  for (Param *f : {&p.beta, &p.n, &p.c, &p.d, &p.h, &p.R, &p.S}) {
    f->fixed = true;
  }
  TersoffForceCalculator c;
  c.params.reserve(1);
  c.params.emplace_back(p);
  return ForceCalculator{std::move(c)};
}

ForceCalculator make_stiweb(double A, double B) {
  SWParams p;
  p.A = A; p.B = B; p.p = 4.0; p.q = 0.0;
  p.delta = 1.0; p.a1 = 3.0; p.gamma = 1.0; p.a2 = 3.0;
  for (Param *f : {&p.p, &p.q, &p.delta, &p.a1, &p.gamma, &p.a2}) {
    f->fixed = true;
  }
  StiwebForceCalculator c;
  c.ntypes = 1;
  c.params.reserve(1);
  c.params.emplace_back(p);
  c.lambda = {Param{1.0, true}}; // single triplet, held fixed
  return ForceCalculator{std::move(c)};
}

ForceCalculator make_symfunc(const std::vector<double> &coeffs) {
  SymmetryFunctionModel m;
  m.ntypes = 1;
  m.rcut = 5.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
  LinearHead h;
  for (double cv : coeffs) {
    h.coeffs.push_back(Param{cv, false});
  }
  h.bias = Param{0.0, true};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{std::move(h)});
  return ForceCalculator{std::move(m)};
}

ForceCalculator make_soap(double scale) {
  SoapModel m;
  m.ntypes = 1;
  m.n_max = 2;
  m.l_max = 2;
  m.rcut = 5.0;
  m.sigma = 0.5;
  m.init_radial_basis();
  LinearHead h;
  const std::size_t S = m.descriptor_size();
  for (std::size_t i = 0; i < S; ++i) {
    const double sign = (i % 2 == 0) ? 1.0 : -1.0;
    h.coeffs.push_back(Param{scale * sign * (0.1 + 0.03 * double(i)), false});
  }
  h.bias = Param{0.0, true};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{std::move(h)});
  return ForceCalculator{std::move(m)};
}

} // namespace

// ── EAM ───────────────────────────────────────────────────────────────────────

TEST(SolverPotentials, EAM_LM) {
  expect_reduces(make_eam(1.0), make_eam(0.6), {make_dimer(2.5)}, 1.0, lm(),
                 "EAM/LM");
}
TEST(SolverPotentials, EAM_Ipopt) {
  expect_reduces(make_eam(1.0), make_eam(0.6), {make_dimer(2.5)}, 1.0, ipopt(),
                 "EAM/Ipopt");
}
TEST(SolverPotentials, EAM_LMNE) {
  expect_reduces(make_eam(1.0), make_eam(0.6), {make_dimer(2.5)}, 1.0, lmne(),
                 "EAM/LMNE");
}

// ── ADP ───────────────────────────────────────────────────────────────────────

TEST(SolverPotentials, ADP_LM) {
  expect_reduces(make_adp(1.0), make_adp(0.6), {make_triangle(2.5)}, 1.0, lm(),
                 "ADP/LM");
}
TEST(SolverPotentials, ADP_Ipopt) {
  expect_reduces(make_adp(1.0), make_adp(0.6), {make_triangle(2.5)}, 1.0,
                 ipopt(), "ADP/Ipopt");
}

// ── Angular (pair + three-body) ───────────────────────────────────────────────

TEST(SolverPotentials, Angular_LM) {
  expect_reduces(make_angular(1.0), make_angular(0.6), {make_triangle(2.5)}, 1.0,
                 lm(), "Angular/LM");
}
TEST(SolverPotentials, Angular_Ipopt) {
  expect_reduces(make_angular(1.0), make_angular(0.6), {make_triangle(2.5)}, 1.0,
                 ipopt(), "Angular/Ipopt");
}

// ── Tersoff ───────────────────────────────────────────────────────────────────

TEST(SolverPotentials, Tersoff_LM) {
  expect_reduces(make_tersoff(2.0, 1.0, 1.5, 1.0),
                 make_tersoff(1.5, 0.8, 1.4, 0.9), {make_triangle(2.4)}, 1.0,
                 lm(), "Tersoff/LM");
}
TEST(SolverPotentials, Tersoff_Ipopt) {
  expect_reduces(make_tersoff(2.0, 1.0, 1.5, 1.0),
                 make_tersoff(1.5, 0.8, 1.4, 0.9), {make_triangle(2.4)}, 1.0,
                 ipopt(), "Tersoff/Ipopt");
}

// ── Stillinger-Weber ──────────────────────────────────────────────────────────

TEST(SolverPotentials, Stiweb_LM) {
  expect_reduces(make_stiweb(2.0, 1.0), make_stiweb(1.5, 0.7),
                 {make_triangle(2.4)}, 1.0, lm(), "Stiweb/LM");
}
TEST(SolverPotentials, Stiweb_Ipopt) {
  expect_reduces(make_stiweb(2.0, 1.0), make_stiweb(1.5, 0.7),
                 {make_triangle(2.4)}, 1.0, ipopt(), "Stiweb/Ipopt");
}

// ── ML: symmetry-function descriptor + linear head (Ipopt; LM in test_ml_fit) ──

TEST(SolverPotentials, SymmetryFunction_Ipopt) {
  expect_reduces(make_symfunc({0.6, -0.35, 0.2}),
                 make_symfunc({0.3, -0.1, 0.05}), {make_cluster()}, 1.0,
                 ipopt(), "SymFunc/Ipopt");
}
TEST(SolverPotentials, SymmetryFunction_LM) {
  expect_reduces(make_symfunc({0.6, -0.35, 0.2}),
                 make_symfunc({0.3, -0.1, 0.05}), {make_cluster()}, 1.0, lm(),
                 "SymFunc/LM");
}
TEST(SolverPotentials, SymmetryFunction_LMNE) {
  expect_reduces(make_symfunc({0.6, -0.35, 0.2}),
                 make_symfunc({0.3, -0.1, 0.05}), {make_cluster()}, 1.0, lmne(),
                 "SymFunc/LMNE");
}

// ── ML: SOAP descriptor + linear head ─────────────────────────────────────────

TEST(SolverPotentials, Soap_LM) {
  expect_reduces(make_soap(1.0), make_soap(0.5), {make_cluster()}, 1.0, lm(),
                 "SOAP/LM");
}
TEST(SolverPotentials, Soap_Ipopt) {
  expect_reduces(make_soap(1.0), make_soap(0.5), {make_cluster()}, 1.0, ipopt(),
                 "SOAP/Ipopt");
}
TEST(SolverPotentials, Soap_LMNE) {
  expect_reduces(make_soap(1.0), make_soap(0.5), {make_cluster()}, 1.0, lmne(),
                 "SOAP/LMNE");
}
