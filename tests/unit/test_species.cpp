#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/species.hpp"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

#include <array>
#include <sstream>
#include <string_view>

namespace leaf = boost::leaf;
using namespace forcesmith;

// ── Species value type ──────────────────────────────────────────────────────────

TEST(Species, RawSlotConvertsToIndex) {
    // A slot-only Species (no element) still indexes tables via the implicit
    // operator std::size_t, exactly like the old bare std::size_t type.
    Species s{std::size_t{3}};
    EXPECT_EQ(static_cast<std::size_t>(s), 3u);
    EXPECT_EQ(s.symbol, std::string_view{});
    EXPECT_EQ(s.Z, 0u);

    std::array<int, 4> table{10, 11, 12, 13};
    EXPECT_EQ(table[s], 13); // implicit conversion in array subscript
}

TEST(Species, CatalogLookup) {
    auto cu = Species::find_by_symbol("Cu");
    ASSERT_TRUE(cu.has_value());
    EXPECT_EQ(cu->Z, 29u);
    EXPECT_EQ(cu->symbol, "Cu");
    EXPECT_NEAR(cu->mass_amu, 63.546, 1e-3);
    EXPECT_EQ(cu->index, 0u); // catalog lookups leave the slot unassigned

    auto by_z = Species::find_by_Z(26);
    ASSERT_TRUE(by_z.has_value());
    EXPECT_EQ(by_z->symbol, "Fe");

    EXPECT_FALSE(Species::find_by_symbol("Xx").has_value());
    EXPECT_FALSE(Species::find_by_Z(999).has_value());
}

// ── SpeciesRegistry ─────────────────────────────────────────────────────────────

TEST(SpeciesRegistry, SlotsAreZSorted) {
    // Input order is Ni, Al — but slots are assigned by Z: Al(13)=0, Ni(28)=1.
    std::array<std::string_view, 2> syms{"Ni", "Al"};
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(reg, build_species_registry(syms));
            EXPECT_EQ(ntypes(reg), 2u);

            BOOST_LEAF_AUTO(al, species_of(reg, "Al"));
            BOOST_LEAF_AUTO(ni, species_of(reg, "Ni"));
            EXPECT_EQ(al.index, 0u);
            EXPECT_EQ(ni.index, 1u);

            EXPECT_EQ(species_at(reg, 0).symbol, "Al");
            EXPECT_EQ(species_at(reg, 1).symbol, "Ni");
            return {};
        },
        [](const std::string& msg) { FAIL() << msg; },
        []() { FAIL() << "unknown error"; });
}

TEST(SpeciesRegistry, DeduplicatesRepeatedSymbols) {
    std::array<std::string_view, 4> syms{"Cu", "Fe", "Cu", "Fe"};
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(reg, build_species_registry(syms));
            EXPECT_EQ(ntypes(reg), 2u);
            BOOST_LEAF_AUTO(fe, species_of(reg, "Fe"));
            BOOST_LEAF_AUTO(cu, species_of(reg, "Cu"));
            EXPECT_EQ(fe.index, 0u); // Fe(26) < Cu(29)
            EXPECT_EQ(cu.index, 1u);
            return {};
        },
        [](const std::string& msg) { FAIL() << msg; },
        []() { FAIL() << "unknown error"; });
}

TEST(SpeciesRegistry, UnknownSymbolIsError) {
    std::array<std::string_view, 2> syms{"Al", "Xx"};
    bool got_error = false;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(reg, build_species_registry(syms));
            (void)reg;
            return {};
        },
        [&](const std::string&) { got_error = true; },
        [&]() { got_error = true; });
    EXPECT_TRUE(got_error);
}

// ── Serialization ───────────────────────────────────────────────────────────────

TEST(Species, AtomRoundTripRestoresIdentityFromZ) {
    Atom a;
    auto cu = Species::find_by_symbol("Cu");
    ASSERT_TRUE(cu.has_value());
    cu->index = 1;
    a.type = *cu;

    std::stringstream ss;
    {
        boost::archive::binary_oarchive oar(ss);
        oar << a;
    }
    Atom b;
    {
        boost::archive::binary_iarchive iar(ss);
        iar >> b;
    }

    EXPECT_EQ(b.type.index, 1u);
    EXPECT_EQ(b.type.Z, 29u);
    EXPECT_EQ(b.type.symbol, "Cu"); // string_view restored into the static catalog
    EXPECT_NEAR(b.type.mass_amu, 63.546, 1e-3);
}

TEST(Species, SyntheticAtomRoundTripsBySlot) {
    Atom a;
    a.type = std::size_t{2}; // no element identity
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oar(ss);
        oar << a;
    }
    Atom b;
    {
        boost::archive::binary_iarchive iar(ss);
        iar >> b;
    }
    EXPECT_EQ(b.type.index, 2u);
    EXPECT_EQ(b.type.Z, 0u);
    EXPECT_EQ(b.type.symbol, std::string_view{});
}
