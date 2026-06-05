#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/species.hpp"
#include "forcesmith/core/types.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

using namespace forcesmith;

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
    EXPECT_DOUBLE_EQ(a.pos.norm(), 0.0);
    EXPECT_DOUBLE_EQ(a.ref.force.norm(), 0.0);
    EXPECT_TRUE(a.neighbors.empty());
}

TEST(Configuration, DefaultConstruction) {
    Configuration cfg;
    EXPECT_TRUE(cfg.atoms.empty());
    EXPECT_DOUBLE_EQ(cfg.ref.energy, 0.0);
    EXPECT_DOUBLE_EQ(cfg.weight, 1.0);
    EXPECT_DOUBLE_EQ(cfg.ref.stress.sum(), 0.0);
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

TEST(Atom, EAMFieldsDefaultToZero) {
    Atom a;
    EXPECT_DOUBLE_EQ(a.rho,    0.0);
    EXPECT_DOUBLE_EQ(a.gradF,  0.0);
    EXPECT_DOUBLE_EQ(a.mu.norm(),     0.0);
    EXPECT_DOUBLE_EQ(a.lambda.norm(), 0.0);
}

TEST(Species, LookupCu) {
    boost::leaf::try_handle_all(
        []() -> boost::leaf::result<void> {
            BOOST_LEAF_AUTO(e, Species::lookup("Cu"));
            EXPECT_EQ(e.Z, 29u);
            EXPECT_EQ(e.symbol, "Cu");
            EXPECT_NEAR(e.mass_amu, 63.546, 0.001);
            return {};
        },
        [](const std::string& msg) { FAIL() << msg; },
        []() { FAIL() << "unknown error"; });
}

TEST(Species, LookupSi) {
    boost::leaf::try_handle_all(
        []() -> boost::leaf::result<void> {
            BOOST_LEAF_AUTO(e, Species::lookup("Si"));
            EXPECT_EQ(e.Z, 14u);
            return {};
        },
        [](const std::string& msg) { FAIL() << msg; },
        []() { FAIL() << "unknown error"; });
}

TEST(Species, LookupUnknownReturnsError) {
    bool got_error = false;
    boost::leaf::try_handle_all(
        [&]() -> boost::leaf::result<void> {
            BOOST_LEAF_AUTO(e, Species::lookup("Xx"));
            (void)e;
            return {};
        },
        [&](const std::string&) { got_error = true; },
        [&]() { got_error = true; });
    EXPECT_TRUE(got_error);
}

TEST(Species, AtomicNumber) {
    boost::leaf::try_handle_all(
        []() -> boost::leaf::result<void> {
            BOOST_LEAF_AUTO(e, Species::lookup("Fe"));
            EXPECT_EQ(e.Z, 26u);
            return {};
        },
        [](const std::string& msg) { FAIL() << msg; },
        []() { FAIL() << "unknown error"; });
}

TEST(Species, AtomicMass) {
    boost::leaf::try_handle_all(
        []() -> boost::leaf::result<void> {
            BOOST_LEAF_AUTO(e, Species::lookup("Au"));
            EXPECT_NEAR(e.mass_amu, 196.967, 0.001);
            return {};
        },
        [](const std::string& msg) { FAIL() << msg; },
        []() { FAIL() << "unknown error"; });
}
