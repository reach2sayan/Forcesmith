// The exact parameter Jacobian of every analytic form, cross-checked against
// central differences.
//
// ddx differentiates the form's expression symbolically, so these partials are
// exact rather than approximate — but "exact" is only useful if the parameter
// each column belongs to is the one the optimizer thinks it is. The symbols of
// an Equation are sorted alphabetically, and a second-derivative tensor is
// indexed by that same order, so a mis-mapped slot would produce a Jacobian
// that is perfectly self-consistent and wrong. Finite differences over the
// FORM's own eval/deriv are the independent check on that mapping.

#include "forcesmith/potentials/analytic_potential.hpp"
#include "forcesmith/potentials/spline.hpp"

#include <boost/mp11/algorithm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace forcesmith;

namespace {

// Build one form at its scaffolding defaults over [rmin, rmax].
template <class T> T make_form(double rmin, double rmax) {
  std::array<double, T::num_params> vals{};
  for (std::size_t i = 0; i < T::num_params; ++i) {
    vals[i] = T::defaults[i].value;
  }
  return T(vals, rmin, rmax);
}

// Central difference of `f` in parameter i, restoring the parameter exactly.
template <class T, class F>
double fd_in_param(T &t, std::size_t i, F &&f) {
  const double v = T::defaults[i].value;
  const double h = 1e-6 * std::max(1.0, std::abs(v));
  t.set_param(i, v + h);
  const double up = f(t);
  t.set_param(i, v - h);
  const double dn = f(t);
  t.set_param(i, v);
  return (up - dn) / (2.0 * h);
}

template <class T> void check_form(const char *name, double r) {
  constexpr std::size_t N = T::num_params;
  T t = make_form<T>(0.5, 6.0);

  // A form that is not finite at this radius (a cutoff's pole guard, a
  // parameter default that puts the probe outside the form's domain) says
  // nothing about the mapping; skip it rather than assert on a NaN.
  if (!std::isfinite(t.eval(r)) || !std::isfinite(t.deriv(r))) {
    return;
  }

  std::vector<double> g(N), dg(N);
  t.param_grad(r, g);        // every parameter is free at construction
  t.dderiv_dparam(r, dg);

  for (std::size_t i = 0; i < N; ++i) {
    const double fd_v = fd_in_param(t, i, [&](const T &f) { return f.eval(r); });
    const double fd_d =
        fd_in_param(t, i, [&](const T &f) { return f.deriv(r); });
    if (std::isfinite(fd_v)) {
      EXPECT_NEAR(g[i], fd_v, 1e-5 * std::abs(fd_v) + 1e-8)
          << name << " d/d" << std::string(T::param_names[i]) << " at r=" << r;
    }
    if (std::isfinite(fd_d)) {
      EXPECT_NEAR(dg[i], fd_d, 1e-5 * std::abs(fd_d) + 1e-8)
          << name << " d2/dr d" << std::string(T::param_names[i])
          << " at r=" << r;
    }
  }

  // d²φ/dr², the term the EAM embedding chain needs.
  const double h = 1e-5;
  const double fd2 = (t.deriv(r + h) - t.deriv(r - h)) / (2.0 * h);
  if (std::isfinite(fd2)) {
    EXPECT_NEAR(t.deriv2(r), fd2, 1e-4 * std::abs(fd2) + 1e-7)
        << name << " d2/dr2 at r=" << r;
  }
}

} // namespace

TEST(AnalyticParamJacobian, MatchesFiniteDifferenceForEveryForm) {
  boost::mp11::mp_for_each<
      boost::mp11::mp_transform<boost::mp11::mp_identity, AnalyticForms>>(
      [](auto tag) {
        using T = typename decltype(tag)::type;
        const char *name = T::names[0].data();
        check_form<T>(name, 1.7);
        check_form<T>(name, 2.9);
      });
}

TEST(AnalyticParamJacobian, SkipsFixedParameters) {
  // The column layout must follow gather_params: fixed parameters take no
  // column, so a fixed first parameter shifts every later one down by one.
  Morse m(1.5, 2.0, 2.0, 0.5, 10.0);
  std::vector<double> all(3);
  m.param_grad(2.2, all);

  m.set_fixed(0, true);
  ASSERT_EQ(m.param_count(), 2u);
  std::vector<double> free(2);
  m.param_grad(2.2, free);
  EXPECT_DOUBLE_EQ(free[0], all[1]);
  EXPECT_DOUBLE_EQ(free[1], all[2]);
}

// ── the model-level chain rule ───────────────────────────────────────────────
// Exact per-form partials are only half of it: the residuals are forces,
// energies and stresses assembled by a force kernel, and each family's chain
// from ∂φ/∂θ into those rows is written by hand. Finite differences over the
// REAL residual assembly are the check on that chain — same rows, same cutoff
// gate, same weights the optimizer sees.

#include "forcesmith/force/pair_force.hpp"
#include "forcesmith/optimization/fd_jacobian.hpp"
#include "forcesmith/optimization/residual_layout.hpp"

namespace {

Configuration make_cluster(std::initializer_list<Vec3> pos, double box) {
  Configuration cfg;
  cfg.bc = PeriodicBC(box * Mat3::Identity());
  for (const Vec3 &p : pos) {
    Atom a;
    a.type = 0;
    a.pos = p;
    a.ref.force = Vec3::Zero();
    cfg.atoms.push_back(a);
  }
  cfg.ref.energy = 0.0;
  cfg.ref.stress = SymTens::Zero();
  return cfg;
}

template <class Calc>
void expect_jacobian_matches_fd(Calc calc, std::vector<Configuration> configs,
                                double energy_weight, double stress_weight,
                                double tol) {
  ASSERT_TRUE(calc.has_analytic_jacobian());
  const int inputs = static_cast<int>(calc.param_count());
  ASSERT_GT(inputs, 0);
  int values = 0;
  for (const auto &cfg : configs) {
    values += static_cast<int>(
        opt::config_residual_count(cfg, stress_weight));
  }

  Eigen::MatrixXd exact = Eigen::MatrixXd::Zero(values, inputs);
  int row = 0;
  for (Configuration &cfg : configs) {
    calc.write_param_jacobian(cfg, row, energy_weight, stress_weight, exact);
    row += static_cast<int>(opt::config_residual_count(cfg, stress_weight));
  }

  Eigen::VectorXd x(inputs);
  calc.gather_params(x, std::size_t{0});
  const auto residual = [&](const Eigen::VectorXd &xx, Eigen::VectorXd &out) {
    out.resize(values);
    calc.scatter_params(xx, std::size_t{0});
    int r = 0;
    for (Configuration &cfg : configs) {
      calc.eval_forces(cfg);
      r = opt::write_config_residuals(cfg, out, r, energy_weight,
                                      stress_weight);
    }
  };
  Eigen::MatrixXd fd(values, inputs);
  opt::fd_jacobian(residual, x, fd);
  calc.scatter_params(x, std::size_t{0});

  double worst = 0.0;
  int wr = 0, wc = 0;
  for (int i = 0; i < values; ++i) {
    for (int j = 0; j < inputs; ++j) {
      const double d = std::abs(exact(i, j) - fd(i, j)) /
                       (1.0 + std::abs(fd(i, j)));
      if (d > worst) {
        worst = d;
        wr = i;
        wc = j;
      }
    }
  }
  EXPECT_LT(worst, tol) << "worst at row " << wr << " col " << wc
                        << ": exact=" << exact(wr, wc) << " fd=" << fd(wr, wc);
}

PairForceCalculator morse_pair() {
  PairForceCalculator calc;
  calc.ntypes = 1;
  calc.pair.reserve(1);
  calc.pair.emplace_back(RadialPotential(Morse(0.35, 1.4, 2.6, 0.5, 6.0)));
  return calc;
}

} // namespace

TEST(AnalyticParamJacobian, PairForcesMatchFiniteDifference) {
  expect_jacobian_matches_fd(morse_pair(),
                             {make_cluster({{0.0, 0.0, 0.0},
                                            {2.55, 0.0, 0.0},
                                            {0.3, 2.7, 0.1},
                                            {2.2, 2.4, 2.9}},
                                           14.0)},
                             /*energy_weight=*/0.7, /*stress_weight=*/0.0,
                             1e-6);
}

TEST(AnalyticParamJacobian, PairStressRowsMatchFiniteDifference) {
  expect_jacobian_matches_fd(morse_pair(),
                             {make_cluster({{0.0, 0.0, 0.0},
                                            {2.55, 0.1, 0.0},
                                            {0.3, 2.7, 0.1}},
                                           11.0),
                              make_cluster({{0.0, 0.0, 0.0},
                                            {2.9, 0.0, 0.4},
                                            {1.1, 2.5, 0.2},
                                            {2.6, 2.2, 2.7}},
                                           12.0)},
                             /*energy_weight=*/1.0, /*stress_weight=*/0.4,
                             1e-6);
}

TEST(AnalyticParamJacobian, SplineTableFallsBackToFiniteDifference) {
  // A tabulated table has no exact ∂φ/∂θ, so the model must say so rather than
  // hand back a wrong Jacobian.
  PairForceCalculator calc;
  calc.ntypes = 1;
  calc.pair.reserve(1);
  calc.pair.emplace_back(RadialPotential(
      SplinePotential(std::vector<double>{1.0, 2.0, 3.0, 4.0},
                      std::vector<double>{0.5, -0.2, -0.1, 0.0})));
  EXPECT_FALSE(calc.has_analytic_jacobian());
}

#include "forcesmith/force/eam_force.hpp"

namespace {

// One-element EAM whose three tables are all analytic: an exponential-decay
// pair term, an exponential density, and a square-root-like embedding.
EAMForceCalculator analytic_eam() {
  EAMForceCalculator calc;
  calc.ntypes = 1;
  calc.pair.reserve(1);
  calc.pair.emplace_back(RadialPotential(ExpDecay(2.4, 0.9, 0.5, 6.0)));
  calc.density.reserve(1);
  calc.density.emplace_back(RadialPotential(ExpDecay(1.1, 1.3, 0.5, 6.0)));
  calc.embedding.reserve(1);
  calc.embedding.emplace_back(RadialPotential(SqrtFunc(-2.0, 0.0, 0.05, 40.0)));
  return calc;
}

} // namespace

TEST(AnalyticParamJacobian, EamForcesMatchFiniteDifference) {
  expect_jacobian_matches_fd(analytic_eam(),
                             {make_cluster({{0.0, 0.0, 0.0},
                                            {2.55, 0.0, 0.0},
                                            {0.3, 2.7, 0.1},
                                            {2.2, 2.4, 2.9}},
                                           14.0)},
                             /*energy_weight=*/0.7, /*stress_weight=*/0.0,
                             1e-5);
}

TEST(AnalyticParamJacobian, EamStressRowsMatchFiniteDifference) {
  expect_jacobian_matches_fd(analytic_eam(),
                             {make_cluster({{0.0, 0.0, 0.0},
                                            {2.55, 0.1, 0.0},
                                            {0.3, 2.7, 0.1}},
                                           11.0),
                              make_cluster({{0.0, 0.0, 0.0},
                                            {2.9, 0.0, 0.4},
                                            {1.1, 2.5, 0.2},
                                            {2.6, 2.2, 2.7}},
                                           12.0)},
                             /*energy_weight=*/1.0, /*stress_weight=*/0.4,
                             1e-5);
}

// Two element types: the single-type cases above cannot catch a column-mapping
// error, because with one type every table has exactly one entry and every
// wrong index still lands on it. With two types the pair table has three
// entries in pair_ordinal order and the density/embedding tables two each.

namespace {

Configuration make_binary(std::initializer_list<std::pair<int, Vec3>> spec,
                          double box) {
  Configuration cfg;
  cfg.bc = PeriodicBC(box * Mat3::Identity());
  for (const auto &[t, p] : spec) {
    Atom a;
    a.type = static_cast<std::size_t>(t);
    a.pos = p;
    a.ref.force = Vec3::Zero();
    cfg.atoms.push_back(a);
  }
  cfg.ref.energy = 0.0;
  cfg.ref.stress = SymTens::Zero();
  return cfg;
}

} // namespace

TEST(AnalyticParamJacobian, BinaryPairMatchesFiniteDifference) {
  PairForceCalculator calc;
  calc.ntypes = 2;
  calc.pair.reserve(2); // 0-0, 0-1, 1-1 in pair_ordinal order
  calc.pair.emplace_back(RadialPotential(Morse(0.35, 1.4, 2.6, 0.5, 6.0)));
  calc.pair.emplace_back(RadialPotential(Morse(0.22, 1.7, 2.4, 0.5, 6.0)));
  calc.pair.emplace_back(RadialPotential(ExpDecay(2.1, 1.05, 0.5, 6.0)));

  expect_jacobian_matches_fd(std::move(calc),
                             {make_binary({{0, {0.0, 0.0, 0.0}},
                                           {1, {2.55, 0.0, 0.0}},
                                           {0, {0.3, 2.7, 0.1}},
                                           {1, {2.2, 2.4, 2.9}}},
                                          14.0)},
                             /*energy_weight=*/0.7, /*stress_weight=*/0.3,
                             1e-6);
}

TEST(AnalyticParamJacobian, BinaryEamMatchesFiniteDifference) {
  EAMForceCalculator calc;
  calc.ntypes = 2;
  calc.pair.reserve(2);
  calc.pair.emplace_back(RadialPotential(ExpDecay(2.4, 0.9, 0.5, 6.0)));
  calc.pair.emplace_back(RadialPotential(ExpDecay(2.0, 1.1, 0.5, 6.0)));
  calc.pair.emplace_back(RadialPotential(ExpDecay(1.7, 1.0, 0.5, 6.0)));
  calc.density.reserve(2);
  calc.density.emplace_back(RadialPotential(ExpDecay(1.1, 1.3, 0.5, 6.0)));
  calc.density.emplace_back(RadialPotential(ExpDecay(0.9, 1.15, 0.5, 6.0)));
  calc.embedding.reserve(2);
  calc.embedding.emplace_back(RadialPotential(SqrtFunc(-2.0, 0.0, 0.05, 40.0)));
  calc.embedding.emplace_back(RadialPotential(SqrtFunc(-1.6, 0.0, 0.05, 40.0)));

  expect_jacobian_matches_fd(std::move(calc),
                             {make_binary({{0, {0.0, 0.0, 0.0}},
                                           {1, {2.55, 0.0, 0.0}},
                                           {0, {0.3, 2.7, 0.1}},
                                           {1, {2.2, 2.4, 2.9}},
                                           {1, {1.2, 0.9, 2.6}}},
                                          13.0)},
                             /*energy_weight=*/0.8, /*stress_weight=*/0.25,
                             1e-5);
}

#include "forcesmith/force/angular_force.hpp"

TEST(AnalyticParamJacobian, AngularTripletsMatchFiniteDifference) {
  // The triplet term writes into three atoms' force rows and both bonds'
  // virial, and the angular table is a function of cos θ rather than of r —
  // three ways the column or row mapping could be wrong.
  AngularForceCalculator calc;
  calc.ntypes = 1;
  calc.pair.reserve(1);
  calc.pair.emplace_back(RadialPotential(Morse(0.3, 1.5, 2.6, 0.5, 6.0)));
  calc.radial.reserve(1);
  calc.radial.emplace_back(RadialPotential(ExpDecay(1.4, 1.1, 0.5, 6.0)));
  calc.angular.reserve(1);
  calc.angular.emplace_back(RadialPotential(Parabola(0.4, 0.2, -0.3, -1.2, 1.2)));

  expect_jacobian_matches_fd(std::move(calc),
                             {make_cluster({{0.0, 0.0, 0.0},
                                            {2.55, 0.0, 0.0},
                                            {0.3, 2.7, 0.1},
                                            {2.2, 2.4, 2.9}},
                                           14.0)},
                             /*energy_weight=*/0.7, /*stress_weight=*/0.3,
                             1e-5);
}

#include "forcesmith/force/adp_force.hpp"

TEST(AnalyticParamJacobian, AdpMomentsMatchFiniteDifference) {
  // ADP is the deepest chain: a dipole or quadrupole parameter moves the
  // moment at both ends of every bond, on top of the EAM density chain.
  ADPForceCalculator calc;
  calc.ntypes = 1;
  calc.pair.reserve(1);
  calc.pair.emplace_back(RadialPotential(ExpDecay(2.4, 0.9, 0.5, 6.0)));
  calc.density.reserve(1);
  calc.density.emplace_back(RadialPotential(ExpDecay(1.1, 1.3, 0.5, 6.0)));
  calc.embedding.reserve(1);
  calc.embedding.emplace_back(RadialPotential(SqrtFunc(-2.0, 0.0, 0.05, 40.0)));
  calc.dipole.reserve(1);
  calc.dipole.emplace_back(RadialPotential(ExpDecay(0.4, 1.2, 0.5, 6.0)));
  calc.quadrupole.reserve(1);
  calc.quadrupole.emplace_back(RadialPotential(ExpDecay(0.25, 1.05, 0.5, 6.0)));

  expect_jacobian_matches_fd(std::move(calc),
                             {make_cluster({{0.0, 0.0, 0.0},
                                            {2.55, 0.0, 0.0},
                                            {0.3, 2.7, 0.1},
                                            {2.2, 2.4, 2.9}},
                                           14.0)},
                             /*energy_weight=*/0.7, /*stress_weight=*/0.3,
                             1e-5);
}
