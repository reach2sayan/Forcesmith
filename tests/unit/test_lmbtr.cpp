#include "potfit/core/neighbor_list.hpp"
#include "potfit/io/force_model_reader.hpp"
#include "potfit/potentials/lmbtr.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <optional>

using namespace potfit;

namespace {

// Central atom at the origin + a few neighbours (asymmetric, 3D).
Configuration make_env() {
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

LMBTR make_base() {
  LMBTR m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.weight_scale = 3.0;
  m.k2 = LMBTR::Grid{0.0, 6.0, 6, 0.4};
  m.k3 = LMBTR::Grid{-1.0, 1.0, 6, 0.2};
  return m;
}

// Attach a linear head whose coefficient count matches the descriptor size.
LMBTR with_linear_head(LMBTR m) {
  LinearHead h;
  const std::size_t S = m.descriptor_size();
  for (std::size_t i = 0; i < S; ++i) {
    h.coeffs.push_back(Param{0.1 + 0.03 * static_cast<double>(i), false});
  }
  h.bias = Param{0.0, true};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{std::move(h)});
  return m;
}

} // namespace

TEST(Lmbtr, DescriptorSizeFormula) {
  LMBTR m = make_base();
  const LMBTR::Grid g2{0.0, 6.0, 6, 0.4};
  const LMBTR::Grid g3{-1.0, 1.0, 6, 0.2};
  for (std::size_t S : {1u, 2u}) {
    m.ntypes = S;
    const std::size_t P = S * (S + 1) / 2;
    m.k2 = g2; m.k3 = g3;
    EXPECT_EQ(m.descriptor_size(), S * 6 + P * 6);
    m.k2 = g2; m.k3 = std::nullopt;
    EXPECT_EQ(m.descriptor_size(), S * 6);
    m.k2 = std::nullopt; m.k3 = g3;
    EXPECT_EQ(m.descriptor_size(), P * 6);
  }
}

// With k2 disabled, the k3 block moves to base 0. Its values must be unchanged
// — i.e. equal to the k3 tail of the both-terms descriptor — which exercises the
// layout's block-shift when an optional term is absent.
TEST(Lmbtr, K3OnlyMatchesTail) {
  LMBTR both = make_base();
  both.normalize_l2 = false;
  auto cfg = make_env();
  build_neighbor_list(cfg, both.rcut);
  const Eigen::VectorXd v_both = both.get_descriptor(cfg.atoms[0]).values;

  const std::size_t S = both.ntypes;
  const std::size_t n2 = static_cast<std::size_t>(both.k2->n);
  const Eigen::Index k3_base = static_cast<Eigen::Index>(S * n2);
  const Eigen::Index k3_len = v_both.size() - k3_base;

  LMBTR k3only = both;
  k3only.k2 = std::nullopt;
  const Eigen::VectorXd v_k3 = k3only.get_descriptor(cfg.atoms[0]).values;

  ASSERT_EQ(v_k3.size(), k3_len);
  EXPECT_NEAR((v_k3 - v_both.tail(k3_len)).norm(), 0.0, 1e-14);
  EXPECT_GT(v_k3.norm(), 1e-6); // non-trivial
}

TEST(Lmbtr, L2Normalized) {
  LMBTR m = make_base();
  m.normalize_l2 = true;
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v = m.get_descriptor(cfg.atoms[0]).values;
  EXPECT_NEAR(v.norm(), 1.0, 1e-12);
}

TEST(Lmbtr, TranslationInvariant) {
  LMBTR m = make_base();
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  auto shifted = make_env();
  const Vec3 t(3.7, -1.2, 0.9);
  for (auto &a : shifted.atoms) {
    a.pos += t;
  }
  build_neighbor_list(shifted, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(shifted.atoms[0]).values;
  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-10);
}

TEST(Lmbtr, RotationInvariant) {
  LMBTR m = make_base();
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  const Eigen::AngleAxisd R(0.7, Vec3(0.3, -0.8, 0.5).normalized());
  auto rot = make_env();
  for (auto &a : rot.atoms) {
    a.pos = R * a.pos;
  }
  build_neighbor_list(rot, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(rot.atoms[0]).values;
  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-9);
}

TEST(Lmbtr, PermutationInvariant) {
  LMBTR m = make_base();
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  auto perm = make_env();
  std::swap(perm.atoms[1], perm.atoms[3]);
  build_neighbor_list(perm, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(perm.atoms[0]).values;
  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-12);
}

TEST(Lmbtr, SensitiveToGeometry) {
  LMBTR m = make_base();
  m.normalize_l2 = false;
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  auto moved = make_env();
  moved.atoms[1].pos += Vec3(0.4, -0.3, 0.2);
  build_neighbor_list(moved, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(moved.atoms[0]).values;
  EXPECT_GT((v1 - v0).norm(), 1e-3);
}

// Forces (via the framework's finite-difference fallback) must match the
// finite-difference of the total energy.
TEST(Lmbtr, ForceMatchesEnergyFD) {
  LMBTR m = with_linear_head(make_base());

  auto energy_at = [&](int atom, int comp, double x) {
    auto cfg = make_env();
    cfg.atoms[atom].pos[comp] = x;
    m.eval_forces(cfg);
    return cfg.calc_energy;
  };

  auto cfg = make_env();
  m.eval_forces(cfg);

  const double dr = 1e-5;
  for (int atom = 0; atom < 4; ++atom) {
    for (int c = 0; c < 3; ++c) {
      const double x0 = make_env().atoms[atom].pos[c];
      const double fd =
          -(energy_at(atom, c, x0 + dr / 2) - energy_at(atom, c, x0 - dr / 2)) /
          dr;
      EXPECT_NEAR(cfg.atoms[atom].calc_force[c], fd,
                  1e-4 * std::abs(fd) + 1e-5)
          << "atom " << atom << " comp " << c;
    }
  }
}

TEST(Lmbtr, ReaderRoundTrip) {
  const char *json = R"({
    "model": "ml",
    "ntypes": 1,
    "descriptor": {"type": "lmbtr", "rcut": 6.0, "weight_scale": 3.0,
                   "normalize": true,
                   "k2": {"min": 0.0, "max": 6.0, "n": 8, "sigma": 0.3},
                   "k3": {"min": -1.0, "max": 1.0, "n": 8, "sigma": 0.1}},
    "heads": [{"type": "linear",
               "coeffs": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "bias": 0.0}]
  })";

  auto r = io::parse_force_model(json);
  ASSERT_TRUE(r) << "parse failed";
  ASSERT_TRUE(std::holds_alternative<LMBTR>(*r));

  const auto &lm = std::get<LMBTR>(*r);
  EXPECT_EQ(lm.descriptor_size(), 16u); // 8 (k2) + 8 (k3), ntypes=1
  ASSERT_TRUE(lm.k2.has_value());
  ASSERT_TRUE(lm.k3.has_value());
  EXPECT_EQ(lm.k2->n, 8);
  EXPECT_EQ(lm.k3->n, 8);
}
