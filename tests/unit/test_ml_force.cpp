#include "potfit/core/neighbor_list.hpp"
#include "potfit/io/force_model_reader.hpp"
#include "potfit/potentials/symmetry_functions.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <variant>

using namespace potfit;

// ── Fixtures ─────────────────────────────────────────────────────────────────

// Single-element model with `S` radial G2 functions and given head coeffs.
static SymmetryFunctionModel make_model(const std::vector<SymmetryFunctionModel::G2> &g2,
                                        const std::vector<double> &coeffs,
                                        double bias = 0.0, double rcut = 6.0) {
  SymmetryFunctionModel m;
  m.ntypes = 1;
  m.rcut = rcut;
  m.radial = g2;
  LinearHead h;
  for (double c : coeffs) {
    h.coeffs.push_back(Param{c, false});
  }
  h.bias = Param{bias, true};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{std::move(h)});
  return m;
}

static Configuration make_dimer(double r) {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1;
  a0.type = 0;
  a0.pos = {0.0, 0.0, 0.0};
  a1.type = 0;
  a1.pos = {r, 0.0, 0.0};
  cfg.atoms = {a0, a1};
  return cfg;
}

// Small asymmetric 3D cluster so forces have all components nonzero.
static Configuration make_cluster() {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1, a2, a3;
  a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
  a1.type = 0; a1.pos = {2.1, 0.3, -0.2};
  a2.type = 0; a2.pos = {0.4, 2.0, 0.5};
  a3.type = 0; a3.pos = {-1.3, 0.7, 1.9};
  cfg.atoms = {a0, a1, a2, a3};
  return cfg;
}

// ── Descriptor correctness ───────────────────────────────────────────────────

TEST(MLForce, DimerDescriptorMatchesHand) {
  const double r = 2.5, eta = 0.8, rs = 1.0, rcut = 6.0;
  auto cfg = make_dimer(r);
  build_neighbor_list(cfg, rcut);

  auto m = make_model({{eta, rs}}, {1.0});
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);

  const double fc = 0.5 * (1.0 + std::cos(std::numbers::pi * r / rcut));
  const double expect = std::exp(-eta * (r - rs) * (r - rs)) * fc;
  ASSERT_EQ(d.values.size(), 1);
  EXPECT_NEAR(d.values[0], expect, 1e-12);
  EXPECT_TRUE(d.has_grad);
}

TEST(MLForce, DescriptorGradTranslationInvariant) {
  // dD/dr_self must equal −Σ_neighbors dD/dr_neighbor.
  auto cfg = make_cluster();
  build_neighbor_list(cfg, 6.0);
  auto m = make_model({{0.5, 0.0}, {1.5, 2.0}}, {1.0, 1.0});

  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);
  DescriptorGrad sum = DescriptorGrad::Zero(d.values.size(), 3);
  for (const auto &gn : d.grad_neigh) {
    sum += gn;
  }
  EXPECT_NEAR((d.grad_self + sum).norm(), 0.0, 1e-12);
}

// ── Force correctness ─────────────────────────────────────────────────────────

TEST(MLForce, AnalyticForceMatchesEnergyFD) {
  auto m = make_model({{0.5, 0.0}, {1.2, 1.5}}, {0.7, -0.4}, 0.1);

  auto energy_at = [&](int atom, int comp, double x) {
    auto cfg = make_cluster();
    cfg.atoms[atom].pos[comp] = x;
    m.eval_forces(cfg);
    return cfg.calc_energy;
  };

  auto cfg = make_cluster();
  m.eval_forces(cfg);

  const double dr = 1e-6;
  for (int atom = 0; atom < 4; ++atom) {
    for (int c = 0; c < 3; ++c) {
      const double x0 = make_cluster().atoms[atom].pos[c];
      const double fd =
          -(energy_at(atom, c, x0 + dr / 2) - energy_at(atom, c, x0 - dr / 2)) /
          dr;
      EXPECT_NEAR(cfg.atoms[atom].calc_force[c], fd,
                  1e-5 * std::abs(fd) + 1e-7)
          << "atom " << atom << " comp " << c;
    }
  }
}

TEST(MLForce, NewtonThirdLaw) {
  auto m = make_model({{0.5, 0.0}, {1.2, 1.5}}, {0.7, -0.4});
  auto cfg = make_cluster();
  m.eval_forces(cfg);

  Vec3 net = Vec3::Zero();
  for (const auto &a : cfg.atoms) {
    net += a.calc_force;
  }
  EXPECT_NEAR(net.norm(), 0.0, 1e-10);
}

TEST(MLForce, FdFallbackMatchesAnalytic) {
  // The built-in FD force path must reproduce the analytic one.
  auto cfg_a = make_cluster();
  auto m = make_model({{0.5, 0.0}, {1.2, 1.5}}, {0.7, -0.4}, 0.1);
  m.eval_forces(cfg_a);

  auto cfg_fd = make_cluster();
  m.use_analytic_grads = false;
  m.eval_forces(cfg_fd);

  EXPECT_NEAR(cfg_fd.calc_energy, cfg_a.calc_energy, 1e-12);
  for (std::size_t i = 0; i < cfg_a.atoms.size(); ++i) {
    EXPECT_NEAR((cfg_fd.atoms[i].calc_force - cfg_a.atoms[i].calc_force).norm(),
                0.0, 1e-5);
  }
}

TEST(MLForce, BeyondCutoffZero) {
  const double rcut = 4.0;
  auto cfg = make_dimer(5.0); // > rcut
  auto m = make_model({{0.5, 0.0}}, {1.3}, 0.0, rcut);
  m.eval_forces(cfg);

  EXPECT_NEAR(cfg.calc_energy, 0.0, 1e-14);
  EXPECT_NEAR(cfg.atoms[0].calc_force.norm(), 0.0, 1e-14);
}

// ── Parameter plumbing ────────────────────────────────────────────────────────

TEST(MLForce, ParamRoundTrip) {
  auto m = make_model({{0.5, 0.0}, {1.2, 1.5}}, {0.7, -0.4});
  // 2 coeffs free, bias fixed → 2 params.
  ASSERT_EQ(m.param_count(), 2u);

  Eigen::VectorXd v(m.param_count());
  m.gather_params(v, 0);
  EXPECT_NEAR(v[0], 0.7, 1e-15);
  EXPECT_NEAR(v[1], -0.4, 1e-15);

  v[0] = 2.0;
  v[1] = 3.0;
  m.scatter_params(v, 0);
  Eigen::VectorXd v2(m.param_count());
  m.gather_params(v2, 0);
  EXPECT_NEAR(v2[0], 2.0, 1e-15);
  EXPECT_NEAR(v2[1], 3.0, 1e-15);
}

// ── Reader / round-trip ───────────────────────────────────────────────────────

TEST(MLForce, ReaderBuildsAndEvaluates) {
  const char *json = R"({
    "model": "ml",
    "ntypes": 1,
    "descriptor": {"type": "symmetry_functions", "rcut": 6.0,
                   "g2": [{"eta": 0.5, "rs": 0.0}, {"eta": 1.2, "rs": 1.5}]},
    "heads": [{"type": "linear", "coeffs": [0.7, -0.4], "bias": 0.1}]
  })";

  auto r = io::parse_force_model(json);
  ASSERT_TRUE(r) << "parse failed";
  ASSERT_TRUE(std::holds_alternative<SymmetryFunctionModel>(*r));

  const auto &m = std::get<SymmetryFunctionModel>(*r);
  EXPECT_EQ(m.radial.size(), 2u);
  EXPECT_EQ(m.param_count(), 2u); // bias fixed by default

  auto cfg = make_cluster();
  std::get<SymmetryFunctionModel>(*r).eval_forces(cfg);
  EXPECT_TRUE(std::isfinite(cfg.calc_energy));
}

TEST(MLForce, ReaderRejectsCoeffMismatch) {
  const char *json = R"({
    "model": "ml", "ntypes": 1,
    "descriptor": {"type": "symmetry_functions", "rcut": 6.0,
                   "g2": [{"eta": 0.5, "rs": 0.0}]},
    "heads": [{"type": "linear", "coeffs": [0.7, -0.4]}]
  })";
  EXPECT_FALSE(io::parse_force_model(json));
}

// ── MLP head ──────────────────────────────────────────────────────────────────

TEST(MLPHead, GradMatchesFD) {
  // de/dD (backprop) must match a finite-difference of energy_impl.
  auto h = MLPHead::make({4, 6, 5, 1}, MLPHead::Act::Tanh, 7);
  Eigen::VectorXd D(4);
  D << 0.3, -0.7, 1.1, 0.2;

  const Eigen::VectorXd g = h.grad_impl(D);
  const double dx = 1e-6;
  for (int k = 0; k < 4; ++k) {
    Eigen::VectorXd dp = D, dm = D;
    dp[k] += dx;
    dm[k] -= dx;
    const double fd = (h.energy_impl(dp) - h.energy_impl(dm)) / (2 * dx);
    EXPECT_NEAR(g[k], fd, 1e-6 * std::abs(fd) + 1e-8);
  }
}

TEST(MLPHead, SiLUGradMatchesFD) {
  auto h = MLPHead::make({3, 4, 1}, MLPHead::Act::SiLU, 3);
  Eigen::VectorXd D(3);
  D << 0.5, -1.2, 0.8;
  const Eigen::VectorXd g = h.grad_impl(D);
  const double dx = 1e-6;
  for (int k = 0; k < 3; ++k) {
    Eigen::VectorXd dp = D, dm = D;
    dp[k] += dx;
    dm[k] -= dx;
    const double fd = (h.energy_impl(dp) - h.energy_impl(dm)) / (2 * dx);
    EXPECT_NEAR(g[k], fd, 1e-6 * std::abs(fd) + 1e-8);
  }
}

TEST(MLPHead, ParamRoundTrip) {
  auto h = MLPHead::make({4, 6, 5, 1}, MLPHead::Act::Tanh, 7);
  const std::size_t n = h.param_count();
  ASSERT_EQ(n, 4u * 6 + 6 + 6u * 5 + 5 + 5u * 1 + 1);

  Eigen::VectorXd v(n);
  h.gather_params(v, 0);
  Eigen::VectorXd v2 = v;
  v2.array() += 0.123;
  h.scatter_params(v2, 0);
  Eigen::VectorXd v3(n);
  h.gather_params(v3, 0);
  EXPECT_NEAR((v3 - v2).norm(), 0.0, 1e-14);
}

TEST(MLForce, G2WithMLPHeadForceMatchesFD) {
  // Analytic descriptor gradients (G2) + analytic head gradient (MLP backprop)
  // must reproduce the finite-difference force.
  SymmetryFunctionModel m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{MLPHead::make({3, 5, 1}, MLPHead::Act::Tanh, 2)});

  auto energy_at = [&](int atom, int comp, double x) {
    auto cfg = make_cluster();
    cfg.atoms[atom].pos[comp] = x;
    m.eval_forces(cfg);
    return cfg.calc_energy;
  };
  auto cfg = make_cluster();
  m.eval_forces(cfg);

  const double dr = 1e-6;
  for (int atom = 0; atom < 4; ++atom) {
    for (int c = 0; c < 3; ++c) {
      const double x0 = make_cluster().atoms[atom].pos[c];
      const double fd =
          -(energy_at(atom, c, x0 + dr / 2) - energy_at(atom, c, x0 - dr / 2)) /
          dr;
      EXPECT_NEAR(cfg.atoms[atom].calc_force[c], fd,
                  1e-5 * std::abs(fd) + 1e-7)
          << "atom " << atom << " comp " << c;
    }
  }
}
