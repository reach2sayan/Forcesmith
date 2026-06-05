#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/potentials/acsf.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

using namespace forcesmith;

namespace {

double fc_cos(double r, double rcut) {
  if (r >= rcut) {
    return 0.0;
  }
  return 0.5 * (1.0 + std::cos(std::numbers::pi * r / rcut));
}

// Central atom at the origin + a few neighbours (asymmetric, 3D). `types` sets
// each neighbour's element; the central atom is always type 0.
Configuration make_env(int t1 = 0, int t2 = 0, int t3 = 0) {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1, a2, a3;
  a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
  a1.type = t1; a1.pos = {2.1, 0.3, -0.2};
  a2.type = t2; a2.pos = {0.4, 2.0, 0.5};
  a3.type = t3; a3.pos = {-1.3, 0.7, 1.9};
  cfg.atoms = {a0, a1, a2, a3};
  return cfg;
}

} // namespace

TEST(Acsf, DescriptorSizeFormula) {
  ACSF m;
  m.rcut = 6.0;
  m.g1 = 1;
  m.radial = {{1.0, 0.0}, {0.5, 1.0}};       // 2 G2
  m.g3 = {{1.0}};                            // 1 G3
  m.g4 = {{1.0, 1.0, 1.0}, {0.5, 2.0, -1.0}}; // 2 G4
  m.g5 = {{1.0, 1.0, 1.0}};                  // 1 G5

  for (std::size_t S : {1u, 2u, 3u}) {
    m.ntypes = S;
    const std::size_t P = S * (S + 1) / 2;
    const std::size_t expect = S * (1 + 2 + 1) + P * (2 + 1);
    EXPECT_EQ(m.descriptor_size(), expect) << "ntypes=" << S;
  }
}

// AcsfLayout is the single source of truth for the descriptor's flat layout.
// Lock its size, the per-component indices, and that the five family blocks tile
// [0, size()) exactly — contiguous, no gaps, no overlap.
TEST(Acsf, LayoutIndices) {
  const std::size_t S = 2, nG1 = 1, nG2 = 2, nG3 = 0, nG4 = 1, nG5 = 1;
  const std::size_t P = S * (S + 1) / 2; // 3
  const AcsfLayout L{S, nG1, nG2, nG3, nG4, nG5};

  // size == hand-computed total: S*(nG1+nG2+nG3) + P*(nG4+nG5).
  EXPECT_EQ(L.size(), static_cast<Eigen::Index>(S * (nG1 + nG2 + nG3) +
                                                P * (nG4 + nG5)));

  // Spot-check a few indices against base + chan*count + t.
  EXPECT_EQ(L.radial(SymmetryFunctionFamily::G1, 0, 0), 0); // first slot
  EXPECT_EQ(L.radial(SymmetryFunctionFamily::G2, 1, 1), 5); // base 2 + 1*2 + 1
  EXPECT_EQ(L.angular(SymmetryFunctionFamily::G4, 2, 0), 8); // base 6 + 2*1 + 0
  EXPECT_EQ(L.angular(SymmetryFunctionFamily::G5, 0, 0), 9); // base 9 + 0 + 0

  // Every (family, channel, t) maps to a distinct index covering [0, size()).
  std::vector<int> hits(static_cast<std::size_t>(L.size()), 0);
  auto stamp = [&](Eigen::Index i) {
    ASSERT_GE(i, 0);
    ASSERT_LT(i, L.size());
    ++hits[static_cast<std::size_t>(i)];
  };
  const std::pair<SymmetryFunctionFamily, std::size_t> radial[] = {
      {SymmetryFunctionFamily::G1, nG1}, {SymmetryFunctionFamily::G2, nG2}, {SymmetryFunctionFamily::G3, nG3}};
  for (auto [f, n] : radial) {
    for (std::size_t s = 0; s < S; ++s) {
      for (std::size_t t = 0; t < n; ++t) {
        stamp(L.radial(f, s, t));
      }
    }
  }
  const std::pair<SymmetryFunctionFamily, std::size_t> angular[] = {{SymmetryFunctionFamily::G4, nG4},
                                                    {SymmetryFunctionFamily::G5, nG5}};
  for (auto [f, n] : angular) {
    for (std::size_t po = 0; po < P; ++po) {
      for (std::size_t t = 0; t < n; ++t) {
        stamp(L.angular(f, po, t));
      }
    }
  }
  for (Eigen::Index i = 0; i < L.size(); ++i) {
    EXPECT_EQ(hits[static_cast<std::size_t>(i)], 1) << "index " << i;
  }
}

// Single-element layout must be identical to the legacy G2-only descriptor.
TEST(Acsf, SingleElementG2Unchanged) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}};
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);
  EXPECT_EQ(static_cast<std::size_t>(d.values.size()), 2u);
  EXPECT_TRUE(d.has_grad);
}

TEST(Acsf, G1HandValue) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.g1 = 1;
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);

  double expect = 0.0;
  for (const auto &nb : cfg.atoms[0].neighbors) {
    expect += fc_cos(nb.dist.norm(), m.rcut);
  }
  ASSERT_EQ(d.values.size(), 1);
  EXPECT_NEAR(d.values[0], expect, 1e-12);
}

TEST(Acsf, G3HandValue) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.g3 = {{1.3}};
  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);

  double expect = 0.0;
  for (const auto &nb : cfg.atoms[0].neighbors) {
    const double r = nb.dist.norm();
    expect += std::cos(1.3 * r) * fc_cos(r, m.rcut);
  }
  ASSERT_EQ(d.values.size(), 1);
  EXPECT_NEAR(d.values[0], expect, 1e-12);
}

// Neighbours of an absent species must leave that species' channel empty.
TEST(Acsf, PerSpeciesChanneling) {
  ACSF m;
  m.ntypes = 2;
  m.rcut = 6.0;
  m.radial = {{0.5, 0.0}}; // 1 G2 → size = S*1 = 2 (channel 0, channel 1)
  auto cfg = make_env(0, 0, 0); // all neighbours type 0
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);
  ASSERT_EQ(d.values.size(), 2);
  EXPECT_GT(std::abs(d.values[0]), 1e-6); // type-0 channel populated
  EXPECT_NEAR(d.values[1], 0.0, 1e-14);   // type-1 channel empty
}

TEST(Acsf, RotationInvariant) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.g1 = 1;
  m.radial = {{0.5, 0.0}};
  m.g3 = {{1.0}};
  m.g4 = {{0.4, 1.0, 1.0}};
  m.g5 = {{0.4, 2.0, -1.0}};

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

TEST(Acsf, PermutationInvariant) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.g4 = {{0.4, 1.0, 1.0}};
  m.g5 = {{0.4, 2.0, -1.0}};

  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const Eigen::VectorXd v0 = m.get_descriptor(cfg.atoms[0]).values;

  auto perm = make_env();
  std::swap(perm.atoms[1], perm.atoms[3]);
  build_neighbor_list(perm, m.rcut);
  const Eigen::VectorXd v1 = m.get_descriptor(perm.atoms[0]).values;
  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-12);
}

TEST(Acsf, TranslationInvariantGrad) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.g1 = 1;
  m.radial = {{0.5, 0.0}};
  m.g3 = {{1.0}};
  m.g4 = {{0.4, 1.0, 1.0}};
  m.g5 = {{0.4, 2.0, -1.0}};

  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);

  // grad_self + Σ_j grad_neigh_j == 0.
  DescriptorGrad sum = d.grad_self;
  for (const auto &g : d.grad_neigh) {
    sum += g;
  }
  EXPECT_LT(sum.norm(), 1e-10);
}

// Analytic dD/dr (radial + angular) vs finite-difference of the descriptor
// w.r.t. each neighbour's bond vector — the oracle for the gradient math.
TEST(Acsf, AnalyticGradMatchesFD) {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.g1 = 1;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}};
  m.g3 = {{1.0}};
  m.g4 = {{0.4, 1.0, 1.0}, {0.6, 2.0, -1.0}};
  m.g5 = {{0.4, 1.0, 1.0}};

  auto cfg = make_env();
  build_neighbor_list(cfg, m.rcut);
  const DescriptorValue d = m.get_descriptor(cfg.atoms[0]);
  ASSERT_TRUE(d.has_grad);
  ASSERT_EQ(d.grad_neigh.size(), cfg.atoms[0].neighbors.size());

  const double h = 1e-6;
  auto &nbrs = cfg.atoms[0].neighbors;
  DescriptorGrad grad_self_fd = DescriptorGrad::Zero(d.values.size(), 3);
  for (std::size_t jj = 0; jj < nbrs.size(); ++jj) {
    for (int k = 0; k < 3; ++k) {
      const double x0 = nbrs[jj].dist[k];
      nbrs[jj].dist[k] = x0 + h;
      const Eigen::VectorXd dp = m.get_descriptor(cfg.atoms[0]).values;
      nbrs[jj].dist[k] = x0 - h;
      const Eigen::VectorXd dm = m.get_descriptor(cfg.atoms[0]).values;
      nbrs[jj].dist[k] = x0;
      const Eigen::VectorXd fd = (dp - dm) / (2.0 * h);
      EXPECT_LT((d.grad_neigh[jj].col(k) - fd).norm(), 1e-5)
          << "neighbor " << jj << " comp " << k;
      grad_self_fd.col(k) -= fd;
    }
  }
  EXPECT_LT((d.grad_self - grad_self_fd).norm(), 1e-5);
}
