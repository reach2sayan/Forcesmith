#include "potfit/core/neighbor_list.hpp"
#include "potfit/potentials/soap.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>

using namespace potfit;

namespace {

SoapModel make_soap(int n_max = 3, int l_max = 3, double rcut = 5.0,
                    double sigma = 0.5) {
  SoapModel m;
  m.ntypes = 1;
  m.n_max = n_max;
  m.l_max = l_max;
  m.rcut = rcut;
  m.sigma = sigma;
  m.init_radial_basis();
  return m;
}

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

} // namespace

TEST(Soap, DescriptorSizeMatches) {
  auto m = make_soap(3, 3);
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);
  EXPECT_EQ(static_cast<std::size_t>(d.values.size()), m.descriptor_size());
  EXPECT_EQ(m.descriptor_size(), (3u + 1u) * (3u * 4u / 2u)); // (l_max+1)·(n(n+1)/2)
  EXPECT_TRUE(d.values.allFinite());
  EXPECT_FALSE(d.has_grad);
}

TEST(Soap, RotationInvariant) {
  auto m = make_soap(3, 3);

  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  // Rotate the whole environment about the (fixed) central atom at the origin.
  const Eigen::AngleAxisd R(0.7, Vec3(0.3, -0.8, 0.5).normalized());
  auto rot = make_env();
  for (auto &a : rot.atoms) {
    a.pos = R * a.pos;
  }
  build_neighbor_list(rot, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(rot.atoms[0]).values;

  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-7);
}

TEST(Soap, PermutationInvariant) {
  auto m = make_soap(3, 3);

  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  // Reorder the neighbour atoms; the central-atom descriptor must not change.
  auto perm = make_env();
  std::swap(perm.atoms[1], perm.atoms[3]);
  build_neighbor_list(perm, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(perm.atoms[0]).values;

  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-12);
}

TEST(Soap, NormalizedToUnit) {
  auto m = make_soap(3, 3);
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v = m.get_descriptor(cfg.atoms[0]).values;
  EXPECT_NEAR(v.norm(), 1.0, 1e-10);
}

// ── dscribe test_soap.py parity (basis-independent property tests) ────────────

// test_symmetries: a local descriptor must be invariant under rigid translation
// of the whole system.
TEST(Soap, TranslationInvariant) {
  auto m = make_soap(3, 3);
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

// test_symmetries: the descriptor must respond to a genuine geometry change.
TEST(Soap, SensitiveToGeometry) {
  auto m = make_soap(3, 3);
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  auto moved = make_env();
  moved.atoms[1].pos += Vec3(0.4, -0.3, 0.2);
  build_neighbor_list(moved, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(moved.atoms[0]).values;
  EXPECT_GT((v1 - v0).norm(), 1e-3);
}

// test_no_system_modification: computing the descriptor must not move any atom.
TEST(Soap, NoSystemModification) {
  auto m = make_soap(3, 3);
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  std::vector<Vec3> before;
  for (const auto &a : cfg.atoms) {
    before.push_back(a.pos);
  }
  (void)m.get_descriptor(cfg.atoms[0]);
  for (std::size_t i = 0; i < cfg.atoms.size(); ++i) {
    EXPECT_EQ(cfg.atoms[i].pos, before[i]);
  }
}

// test_rbf_orthonormality: the orthonormalized radial basis g_n = Σ_a β(n,a)φ_a
// must satisfy ∫ g_n g_m r² dr = δ_nm, i.e. β·S·βᵀ = I (S the closed-form
// overlap). Validates init_radial_basis' S^{-1/2}.
TEST(Soap, RbfOrthonormality) {
  const int n = 4;
  const double rcut = 5.0;
  auto m = make_soap(n, 2, rcut);

  Eigen::MatrixXd S(n, n);
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      const double s = a + b;
      S(a, b) = std::pow(rcut, s + 7.0) *
                (1.0 / (s + 5.0) - 2.0 / (s + 6.0) + 1.0 / (s + 7.0));
    }
  }
  const Eigen::MatrixXd G = m.beta * S * m.beta.transpose();
  EXPECT_LT((G - Eigen::MatrixXd::Identity(n, n)).norm(), 1e-6);
}

// test_number_of_features: multi-species size formula
// (l+1)·[ S·n(n+1)/2 + S(S−1)/2·n² ].
TEST(Soap, MultiSpeciesFeatureCount) {
  SoapModel m;
  m.ntypes = 2;
  m.n_max = 3;
  m.l_max = 3;
  m.rcut = 5.0;
  m.sigma = 0.5;
  m.init_radial_basis();

  // S=2, n=3, l=3: 4 · (2·6 + 1·9) = 84.
  EXPECT_EQ(m.descriptor_size(), 84u);

  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1, a2, a3;
  a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
  a1.type = 1; a1.pos = {2.1, 0.3, -0.2};
  a2.type = 0; a2.pos = {0.4, 2.0, 0.5};
  a3.type = 1; a3.pos = {-1.3, 0.7, 1.9};
  cfg.atoms = {a0, a1, a2, a3};
  build_neighbor_list(cfg, m.rcut);
  EXPECT_EQ(static_cast<std::size_t>(m.get_descriptor(cfg.atoms[0]).values.size()),
            m.descriptor_size());
}
