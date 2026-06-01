#include "potfit/io/config_reader.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

namespace leaf = boost::leaf;
using namespace potfit;
using namespace potfit::io;

// ── Helpers ───────────────────────────────────────────────────────────────────

struct ParseResult {
    bool                       ok = false;
    std::vector<Configuration> configs;
    ParseError                 error;
};

static ParseResult run(std::string_view input) {
    ParseResult out;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(parsed, parse_config(input));
            out.ok      = true;
            out.configs = std::move(parsed.configs);
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
    auto r = run(R"([
      {
        "X": [4.0, 0.0, 0.0],
        "Y": [0.0, 4.0, 0.0],
        "Z": [0.0, 0.0, 4.0],
        "E": -1.5,
        "W": 2.0,
        "atoms": [
          {"element": "Cu", "position": [1.0, 2.0, 3.0]}
        ]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 1u);
    const auto& cfg = r.configs[0];
    EXPECT_EQ(cfg.atoms.size(), 1u);
    EXPECT_DOUBLE_EQ(cfg.ref.energy, -1.5);
    EXPECT_DOUBLE_EQ(cfg.weight, 2.0);
    EXPECT_EQ(cfg.atoms[0].type, 0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].pos.x(), 1.0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].pos.y(), 2.0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].pos.z(), 3.0);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].ref.force.norm(), 0.0);
}

TEST(ConfigReader, TwoAtomsWithForces) {
    auto r = run(R"([
      {
        "X": [5.0, 0.0, 0.0],
        "Y": [0.0, 5.0, 0.0],
        "Z": [0.0, 0.0, 5.0],
        "E": -3.14,
        "atoms": [
          {"element": "Cu", "position": [0.0, 0.0, 0.0], "force": [0.1, 0.2, 0.3]},
          {"element": "Fe", "position": [2.5, 2.5, 2.5], "force": [-0.1, -0.2, -0.3]}
        ]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 1u);
    const auto& cfg = r.configs[0];
    EXPECT_EQ(cfg.atoms.size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.ref.energy, -3.14);
    // Slots are Z-sorted: Fe(26) → 0, Cu(29) → 1.
    // atom 0 is Cu → slot 1
    EXPECT_EQ(cfg.atoms[0].type, 1);
    EXPECT_NEAR(cfg.atoms[0].ref.force.x(),  0.1, 1e-12);
    EXPECT_NEAR(cfg.atoms[0].ref.force.z(),  0.3, 1e-12);
    // atom 1 is Fe → slot 0
    EXPECT_EQ(cfg.atoms[1].type, 0);
    EXPECT_DOUBLE_EQ(cfg.atoms[1].pos.x(), 2.5);
    EXPECT_NEAR(cfg.atoms[1].ref.force.x(), -0.1, 1e-12);
}

TEST(ConfigReader, BoxVectorsStored) {
    auto r = run(R"([
      {
        "X": [3.0, 0.1, 0.0],
        "Y": [0.0, 4.0, 0.2],
        "Z": [0.0, 0.0, 5.0],
        "E": 0.0,
        "atoms": [
          {"element": "Cu", "position": [0.0, 0.0, 0.0]}
        ]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    const auto& box = std::get<PeriodicBC>(r.configs[0].bc).box();
    EXPECT_DOUBLE_EQ(box(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(box(1, 0), 0.1);
    EXPECT_DOUBLE_EQ(box(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(box(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(box(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(box(2, 1), 0.2);
    EXPECT_DOUBLE_EQ(box(2, 2), 5.0);
}

TEST(ConfigReader, StressTensorStored) {
    auto r = run(R"([
      {
        "X": [4.0, 0.0, 0.0],
        "Y": [0.0, 4.0, 0.0],
        "Z": [0.0, 0.0, 4.0],
        "E": 0.0,
        "S": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        "atoms": [
          {"element": "Cu", "position": [0.0, 0.0, 0.0]}
        ]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    const auto& s = r.configs[0].ref.stress;
    EXPECT_DOUBLE_EQ(s(0,0), 1.1);
    EXPECT_DOUBLE_EQ(s(1,1), 2.2);
    EXPECT_DOUBLE_EQ(s(2,2), 3.3);
    EXPECT_DOUBLE_EQ(s(0,1), 4.4);
    EXPECT_DOUBLE_EQ(s(1,0), 4.4);
    EXPECT_DOUBLE_EQ(s(1,2), 5.5);
    EXPECT_DOUBLE_EQ(s(0,2), 6.6);
}

TEST(ConfigReader, MultipleConfigurations) {
    auto r = run(R"([
      {
        "X": [4.0, 0.0, 0.0], "Y": [0.0, 4.0, 0.0], "Z": [0.0, 0.0, 4.0],
        "E": -1.0,
        "atoms": [{"element": "Cu", "position": [0.0, 0.0, 0.0]}]
      },
      {
        "X": [5.0, 0.0, 0.0], "Y": [0.0, 5.0, 0.0], "Z": [0.0, 0.0, 5.0],
        "E": -2.0,
        "atoms": [
          {"element": "Cu", "position": [0.0, 0.0, 0.0]},
          {"element": "Fe", "position": [2.5, 2.5, 2.5]}
        ]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 2u);
    EXPECT_EQ(r.configs[0].atoms.size(), 1u);
    EXPECT_DOUBLE_EQ(r.configs[0].ref.energy, -1.0);
    EXPECT_EQ(r.configs[1].atoms.size(), 2u);
    EXPECT_DOUBLE_EQ(r.configs[1].ref.energy, -2.0);
}

TEST(ConfigReader, ElementTypeMapping) {
    // Cu appears in both configs; Fe appears only in cfg[1].
    // Slots are Z-sorted across the whole dataset and consistent everywhere:
    // Fe(26)=0, Cu(29)=1 — independent of which config a symbol first appears in.
    auto r = run(R"([
      {
        "X": [4.0, 0.0, 0.0], "Y": [0.0, 4.0, 0.0], "Z": [0.0, 0.0, 4.0],
        "E": 0.0,
        "atoms": [{"element": "Cu", "position": [0.0, 0.0, 0.0]}]
      },
      {
        "X": [4.0, 0.0, 0.0], "Y": [0.0, 4.0, 0.0], "Z": [0.0, 0.0, 4.0],
        "E": 0.0,
        "atoms": [
          {"element": "Fe", "position": [1.0, 0.0, 0.0]},
          {"element": "Cu", "position": [2.0, 0.0, 0.0]}
        ]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_EQ(r.configs[0].atoms[0].type, 1);   // Cu → 1
    EXPECT_EQ(r.configs[1].atoms[0].type, 0);   // Fe → 0
    EXPECT_EQ(r.configs[1].atoms[1].type, 1);   // Cu → 1 (same as first config)
}

TEST(ConfigReader, DefaultWeightIsOne) {
    auto r = run(R"([
      {
        "X": [4.0, 0.0, 0.0], "Y": [0.0, 4.0, 0.0], "Z": [0.0, 0.0, 4.0],
        "E": 0.0,
        "atoms": [{"element": "Cu", "position": [0.0, 0.0, 0.0]}]
      }
    ])");
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_DOUBLE_EQ(r.configs[0].weight, 1.0);
}

TEST(ConfigReader, EmptyInputReturnsNoConfigs) {
    auto r = run("[]");
    ASSERT_TRUE(r.ok) << r.error.message;
    EXPECT_TRUE(r.configs.empty());
}

TEST(ConfigReader, MinifiedJsonParsed) {
    auto r = run(R"([{"X":[4,0,0],"Y":[0,4,0],"Z":[0,0,4],"E":-1.0,"atoms":[{"element":"Cu","position":[0,0,0]}]}])");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.configs.size(), 1u);
    EXPECT_DOUBLE_EQ(r.configs[0].ref.energy, -1.0);
}

// ── Error-path tests ──────────────────────────────────────────────────────────

TEST(ConfigReader, ErrorTopLevelNotArray) {
    auto r = run(R"({"X": [4,0,0]})");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(ConfigReader, ErrorMissingEnergy) {
    auto r = run(R"([
      {
        "X": [4,0,0], "Y": [0,4,0], "Z": [0,0,4],
        "atoms": [{"element": "Cu", "position": [0,0,0]}]
      }
    ])");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(ConfigReader, ErrorMalformedPosition) {
    auto r = run(R"([
      {
        "X": [4,0,0], "Y": [0,4,0], "Z": [0,0,4], "E": 0.0,
        "atoms": [{"element": "Cu", "position": [0, 0]}]
      }
    ])");
    EXPECT_FALSE(r.ok);
}

TEST(ConfigReader, ErrorInvalidJson) {
    auto r = run("not valid json {{{");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}
