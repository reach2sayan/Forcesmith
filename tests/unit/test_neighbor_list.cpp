#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/core/radial_potential.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <ranges>

using namespace forcesmith;

// Helpers ─────────────────────────────────────────────────────────────────────

static Configuration make_box(double side) {
    Configuration cfg;
    cfg.bc = PeriodicBC(side * Mat3::Identity());
    return cfg;
}

static Atom make_atom(int type, Vec3 pos) {
    Atom a;
    a.type = type;
    a.pos  = pos;
    return a;
}

// Trivial potential stub — satisfies the RadialPotential concept.
struct ConstPot : NoBounds<ConstPot>, NoSiteCache<ConstPot>, NoRawParamAccess,
                     NoParamJacobian {
    double eval(double)  const { return 0.0; }
    double deriv(double) const { return 0.0; }
    std::pair<double,double> span() const { return {0.0, 100.0}; }
    int    param_count() const { return 0; }
    void   gather_params(Eigen::VectorXd&, int) const {}
    void   scatter_params(const Eigen::VectorXd&, int) {}
};

// ── Basic dimer ───────────────────────────────────────────────────────────────

TEST(NeighborList, DimerInRange) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0}),
                 make_atom(0, {2.0, 0.0, 0.0})};

    build_neighbor_list(cfg, 3.0);

    ASSERT_EQ(cfg.atoms[0].neighbors.size(), 1u);
    ASSERT_EQ(cfg.atoms[1].neighbors.size(), 1u);

    const auto& nb = cfg.atoms[0].neighbors[0];
    EXPECT_NEAR(nb.dist.norm(),   2.0, 1e-12);
    EXPECT_NEAR(nb.dist.squaredNorm(), 4.0, 1e-12);
    EXPECT_NEAR(1.0 / nb.dist.norm(), 0.5, 1e-12);
    EXPECT_NEAR(nb.dist[0], 2.0, 1e-12);
    EXPECT_NEAR(nb.dist[1], 0.0, 1e-12);
    EXPECT_NEAR(nb.dist[2], 0.0, 1e-12);
}

TEST(NeighborList, DimerOutOfRange) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0}),
                 make_atom(0, {5.0, 0.0, 0.0})};

    build_neighbor_list(cfg, 3.0);

    EXPECT_TRUE(cfg.atoms[0].neighbors.empty());
    EXPECT_TRUE(cfg.atoms[1].neighbors.empty());
}

TEST(NeighborList, SelfNotIncluded) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0})};

    build_neighbor_list(cfg, 5.0);

    EXPECT_TRUE(cfg.atoms[0].neighbors.empty());
}

TEST(NeighborList, RebuildsCleanly) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0}),
                 make_atom(0, {1.0, 0.0, 0.0})};

    build_neighbor_list(cfg, 5.0);
    ASSERT_EQ(cfg.atoms[0].neighbors.size(), 1u);

    build_neighbor_list(cfg, 0.5);  // now out of range
    EXPECT_TRUE(cfg.atoms[0].neighbors.empty());
}

// ── Minimum-image periodic boundary condition ─────────────────────────────────

TEST(NeighborList, MinimumImageAcrossBox) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0}),
                 make_atom(0, {9.5, 0.0, 0.0})};  // image at x = -0.5

    build_neighbor_list(cfg, 1.0);

    ASSERT_EQ(cfg.atoms[0].neighbors.size(), 1u);
    EXPECT_NEAR(cfg.atoms[0].neighbors[0].dist.norm(), 0.5, 1e-10);
    EXPECT_NEAR(cfg.atoms[0].neighbors[0].dist[0], -0.5, 1e-10);
}

// ── Slot/potential assignment ─────────────────────────────────────────────────

TEST(NeighborList, PairSlotSingleType) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0}),
                 make_atom(0, {1.0, 0.0, 0.0})};

    RadialPotentialPair pots;
    pots.reserve(1);
    pots.emplace_back(ConstPot{});  // (0,0)
    build_neighbor_list(cfg, 5.0, pots);

    ASSERT_EQ(cfg.atoms[0].neighbors.size(), 1u);
    EXPECT_EQ(cfg.atoms[0].neighbors[0].pot, &(pots[0, 0]));
}

TEST(NeighborList, PairSlotTwoTypes) {
    auto cfg = make_box(10.0);
    cfg.atoms = {make_atom(0, {0.0, 0.0, 0.0}),
                 make_atom(1, {1.0, 0.0, 0.0}),
                 make_atom(1, {0.0, 1.0, 0.0})};

    RadialPotentialPair pots;
    pots.reserve(2);
    pots.emplace_back(ConstPot{});  // (0,0)
    pots.emplace_back(ConstPot{});  // (0,1)
    pots.emplace_back(ConstPot{});  // (1,1)
    build_neighbor_list(cfg, 5.0, pots);

    for (const auto& nb : cfg.atoms[0].neighbors)
        EXPECT_EQ(nb.pot, &(pots[0, 1])) << "expected (0,1) pair potential";

    bool found_type1_type1 = false;
    for (const auto& nb : cfg.atoms[1].neighbors) {
        if (nb.neighbor->type == 1) {
            EXPECT_EQ(nb.pot, &(pots[1, 1]));
            found_type1_type1 = true;
        }
    }
    EXPECT_TRUE(found_type1_type1);
}

// ── FCC nearest neighbors ─────────────────────────────────────────────────────

static Configuration make_fcc(double a, int m = 2) {
    const double L = a * m;
    auto cfg   = make_box(L);

    const Vec3 basis[4] = {
        {0.0,   0.0,   0.0  },
        {a/2.0, a/2.0, 0.0  },
        {a/2.0, 0.0,   a/2.0},
        {0.0,   a/2.0, a/2.0},
    };

    for (const auto& [nx, ny, nz, b] : std::views::cartesian_product(
             std::views::iota(0, m), std::views::iota(0, m), std::views::iota(0, m),
             std::span<const Vec3>(basis)))
        cfg.atoms.push_back(make_atom(0, {nx * a + b[0], ny * a + b[1], nz * a + b[2]}));
    return cfg;
}

TEST(NeighborList, FccNearestNeighborCount) {
    const double a    = 3.615;
    const double rcut = 0.75 * a;
    auto cfg = make_fcc(a, 2);

    build_neighbor_list(cfg, rcut);

    ASSERT_EQ(cfg.atoms.size(), 32u);
    for (std::size_t i = 0; i < cfg.atoms.size(); ++i) {
        EXPECT_EQ(cfg.atoms[i].neighbors.size(), 12u)
            << "atom " << i << " expected 12 nearest neighbors";
    }
}

TEST(NeighborList, FccNearestNeighborDistance) {
    const double a    = 3.615;
    const double rcut = 0.75 * a;
    auto cfg = make_fcc(a, 2);

    build_neighbor_list(cfg, rcut);

    const double expected_r = a / std::sqrt(2.0);
    for (const auto& nb : cfg.atoms[0].neighbors) {
        const double r = nb.dist.norm();
        EXPECT_NEAR(r, expected_r, 1e-10)
            << "unexpected neighbor distance " << r;
    }
}

TEST(NeighborList, FccSecondShellExcluded) {
    const double a    = 3.615;
    const double rcut = 0.75 * a;
    auto cfg = make_fcc(a, 2);

    build_neighbor_list(cfg, rcut);

    for (const auto& atom : cfg.atoms) {
        for (const auto& nb : atom.neighbors) {
            EXPECT_LT(nb.dist.norm(), rcut);
        }
    }
}
