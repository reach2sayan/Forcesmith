#include "potfit/core/types.hpp"
#include "potfit/core/atom.hpp"

#include <gtest/gtest.h>

using namespace potfit;

TEST(Types, Vec3Zero) {
    Vec3 v = Vec3::Zero();
    EXPECT_DOUBLE_EQ(v.norm(), 0.0);
}

TEST(Types, SymTensZero) {
    SymTens s = SymTens::Zero();
    EXPECT_DOUBLE_EQ(s.sum(), 0.0);
}

TEST(Atom, DefaultConstruction) {
    Atom a;
    EXPECT_EQ(a.type, 0);
    EXPECT_EQ(a.conf, 0);
    EXPECT_DOUBLE_EQ(a.pos.norm(), 0.0);
    EXPECT_DOUBLE_EQ(a.force.norm(), 0.0);
    EXPECT_TRUE(a.neighbors.empty());
}

TEST(Configuration, DefaultConstruction) {
    Configuration cfg;
    EXPECT_TRUE(cfg.atoms.empty());
    EXPECT_DOUBLE_EQ(cfg.energy, 0.0);
    EXPECT_DOUBLE_EQ(cfg.weight, 1.0);
    EXPECT_DOUBLE_EQ(cfg.stress.sum(), 0.0);
}

TEST(NeighborEntry, DefaultConstruction) {
    NeighborEntry n;
    EXPECT_EQ(n.neighbor, nullptr);
    EXPECT_EQ(n.pot, nullptr);
    EXPECT_DOUBLE_EQ(n.dist.norm(), 0.0);
}

TEST(PotentialFormat, EnumValues) {
    EXPECT_EQ(static_cast<int>(PotentialFormat::Analytic),            0);
    EXPECT_EQ(static_cast<int>(PotentialFormat::TabulatedEqDist),     3);
    EXPECT_EQ(static_cast<int>(PotentialFormat::TabulatedNonEqDist),  4);
    EXPECT_EQ(static_cast<int>(PotentialFormat::KIM),                 5);
}
