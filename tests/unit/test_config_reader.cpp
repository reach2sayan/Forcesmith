#include "potfit/io/config_reader.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

namespace leaf = boost::leaf;
using namespace potfit;
using namespace potfit::io;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Run parse_config and surface the result + any ParseError into a plain struct.
struct ParseResult {
    bool                       ok = false;
    std::vector<Configuration> configs;
    ParseError                 error;
};

static ParseResult run(std::string_view input) {
    ParseResult out;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(cfgs, parse_config(input));
            out.ok      = true;
            out.configs = std::move(cfgs);
            return {};
        },
        [&](const ParseError& e) {
            out.ok    = false;
            out.error = e;
        },
        [&]() {
            out.ok            = false;
            out.error.message = "unknown error";
        }
    );
    return out;
}

// ── Happy-path tests ──────────────────────────────────────────────────────────

TEST(ConfigReader, SingleAtomNoForces) {
    auto r = run(
        "#N 1 0\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E -1.5\n"
        "#W 2.0\n"
        "0 1.0 2.0 3.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 1u);
    const auto& cfg = r.configs[0];
    EXPECT_EQ(cfg.atoms.size(), 1u);
    EXPECT_DOUBLE_EQ(cfg.energy, -1.5);
    EXPECT_DOUBLE_EQ(cfg.weight, 2.0);
    EXPECT_EQ(cfg.atoms[0].type, 0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].pos.x(), 1.0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].pos.y(), 2.0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].pos.z(), 3.0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].force.norm(), 0.0);  // no forces
}

TEST(ConfigReader, TwoAtomsWithForces) {
    auto r = run(
        "#N 2 1\n"
        "#X 5.0 0.0 0.0\n"
        "#Y 0.0 5.0 0.0\n"
        "#Z 0.0 0.0 5.0\n"
        "#E -3.14\n"
        "0  0.0 0.0 0.0   0.1  0.2  0.3\n"
        "1  2.5 2.5 2.5  -0.1 -0.2 -0.3\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 1u);
    const auto& cfg = r.configs[0];
    EXPECT_EQ(cfg.atoms.size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.energy, -3.14);
    // atom 0
    EXPECT_EQ(cfg.atoms[0].type, 0);
    EXPECT_NEAR(cfg.atoms[0].force.x(),  0.1, 1e-12);
    EXPECT_NEAR(cfg.atoms[0].force.z(),  0.3, 1e-12);
    // atom 1
    EXPECT_EQ(cfg.atoms[1].type, 1);
    EXPECT_DOUBLE_EQ(cfg.atoms[1].pos.x(), 2.5);
    EXPECT_NEAR(cfg.atoms[1].force.x(), -0.1, 1e-12);
}

TEST(ConfigReader, BoxVectorsStored) {
    auto r = run(
        "#N 1 0\n"
        "#X 3.0 0.1 0.0\n"
        "#Y 0.0 4.0 0.2\n"
        "#Z 0.0 0.0 5.0\n"
        "#E 0.0\n"
        "0 0.0 0.0 0.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    const auto& box = std::get<PeriodicBC>(r.configs[0].bc).box();
    // Column 0 = #X vector
    EXPECT_DOUBLE_EQ(box(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(box(1, 0), 0.1);
    EXPECT_DOUBLE_EQ(box(2, 0), 0.0);
    // Column 1 = #Y vector
    EXPECT_DOUBLE_EQ(box(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(box(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(box(2, 1), 0.2);
    // Column 2 = #Z vector
    EXPECT_DOUBLE_EQ(box(2, 2), 5.0);
}

TEST(ConfigReader, StressTensorStored) {
    auto r = run(
        "#N 1 0\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E 0.0\n"
        "#S 1.1 2.2 3.3 4.4 5.5 6.6\n"
        "0 0.0 0.0 0.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    const auto& s = r.configs[0].stress;
    EXPECT_DOUBLE_EQ(s(0,0), 1.1);  // xx
    EXPECT_DOUBLE_EQ(s(1,1), 2.2);  // yy
    EXPECT_DOUBLE_EQ(s(2,2), 3.3);  // zz
    EXPECT_DOUBLE_EQ(s(0,1), 4.4);  // xy — symmetric
    EXPECT_DOUBLE_EQ(s(1,0), 4.4);
    EXPECT_DOUBLE_EQ(s(1,2), 5.5);  // yz
    EXPECT_DOUBLE_EQ(s(0,2), 6.6);  // zx
}

TEST(ConfigReader, MultipleConfigurations) {
    auto r = run(
        "#N 1 0\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E -1.0\n"
        "0 0.0 0.0 0.0\n"
        "#N 2 0\n"
        "#X 5.0 0.0 0.0\n"
        "#Y 0.0 5.0 0.0\n"
        "#Z 0.0 0.0 5.0\n"
        "#E -2.0\n"
        "0 0.0 0.0 0.0\n"
        "1 2.5 2.5 2.5\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 2u);
    EXPECT_EQ(r.configs[0].atoms.size(), 1u);
    EXPECT_DOUBLE_EQ(r.configs[0].energy, -1.0);
    EXPECT_EQ(r.configs[1].atoms.size(), 2u);
    EXPECT_DOUBLE_EQ(r.configs[1].energy, -2.0);
}

TEST(ConfigReader, SkipsUnknownDirectives) {
    // #C (element names) and #F (legacy force marker) should be silently ignored
    auto r = run(
        "#N 1 0\n"
        "#C Cu\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E -1.0\n"
        "#F\n"
        "0 0.0 0.0 0.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_EQ(r.configs[0].atoms.size(), 1u);
}

TEST(ConfigReader, DefaultWeightIsOne) {
    auto r = run(
        "#N 1 0\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E 0.0\n"
        "0 0.0 0.0 0.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_DOUBLE_EQ(r.configs[0].weight, 1.0);
}

TEST(ConfigReader, EmptyInputReturnsNoConfigs) {
    auto r = run("   \n\n  \n");
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_TRUE(r.configs.empty());
}

// ── Error-path tests ──────────────────────────────────────────────────────────

TEST(ConfigReader, ErrorAtomBeforeN) {
    auto r = run("0 0.0 0.0 0.0\n");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(ConfigReader, ErrorTooManyAtoms) {
    auto r = run(
        "#N 1 0\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E 0.0\n"
        "0 0.0 0.0 0.0\n"
        "1 1.0 1.0 1.0\n"  // extra atom
    );
    EXPECT_FALSE(r.ok);
}

TEST(ConfigReader, ErrorIncompleteConfig) {
    // File ends before all atoms are present
    auto r = run(
        "#N 3 0\n"
        "#X 4.0 0.0 0.0\n"
        "#Y 0.0 4.0 0.0\n"
        "#Z 0.0 0.0 4.0\n"
        "#E 0.0\n"
        "0 0.0 0.0 0.0\n"
        // missing 2 more atoms
    );
    EXPECT_FALSE(r.ok);
}

TEST(ConfigReader, ErrorMalformedNLine) {
    auto r = run("#N notanumber 1\n");
    EXPECT_FALSE(r.ok);
}

TEST(ConfigReader, WindowsLineEndingsHandled) {
    auto r = run("#N 1 0\r\n#X 4.0 0.0 0.0\r\n#Y 0.0 4.0 0.0\r\n#Z 0.0 0.0 4.0\r\n#E 0.0\r\n0 0.0 0.0 0.0\r\n");
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_EQ(r.configs[0].atoms.size(), 1u);
}
