#include "potfit/io/force_model_reader.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace leaf = boost::leaf;
using namespace potfit;
using namespace potfit::io;

// ── Helper ────────────────────────────────────────────────────────────────────

struct FMResult {
    bool       ok = false;
    ForceCalculator model;
    ParseError error;
};

static FMResult run(std::string_view input) {
    FMResult out;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(m, parse_force_model(input));
            out.ok    = true;
            out.model = std::move(m);
            return {};
        },
        [&](const ParseError& e) { out.ok = false; out.error = e; },
        [&]()                    { out.ok = false; out.error.message = "unknown error"; }
    );
    return out;
}

// ── pair model ────────────────────────────────────────────────────────────────

TEST(ForceModelReader, Pair_Analytic_LJ) {
    auto r = run(R"({
      "model": "pair",
      "format": "analytic",
      "potentials": [
        {"type": "pair_lj", "rmin": 2.0, "rmax": 6.0, "epsilon": 1.0, "sigma": 2.5}
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<PairForceCalculator>(r.model));
    const auto& calc = std::get<PairForceCalculator>(r.model);
    EXPECT_EQ(calc.pair.size(), 1u);
    // LJ minimum at r = 2^(1/6) * sigma ≈ 2.806
    const double sig = 2.5;
    const double r_eq = sig * std::pow(2.0, 1.0 / 6.0);
    const auto& phi00 = calc.pair[0, 0];
    EXPECT_NEAR(phi00.eval(r_eq), -1.0, 1e-12);
    EXPECT_NEAR(phi00.deriv(r_eq), 0.0, 1e-10);
}

TEST(ForceModelReader, Pair_Tabulated) {
    auto r = run(R"({
      "model": "pair",
      "format": "tabulated",
      "potentials": [
        {"rmin": 1.0, "rmax": 3.0, "knots": [1.0, 0.0, 1.0]}
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<PairForceCalculator>(r.model));
    const auto& calc = std::get<PairForceCalculator>(r.model);
    EXPECT_EQ(calc.pair.size(), 1u);
    const auto& p00 = calc.pair[0, 0];
    EXPECT_NEAR(p00.eval(2.0), 0.0, 1e-10); // middle knot = 0
}

// ── eam model ─────────────────────────────────────────────────────────────────

TEST(ForceModelReader, EAM_SingleType_Tabulated) {
    auto r = run(R"({
      "model": "eam",
      "ntypes": 1,
      "pair": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 1.5, "rmax": 6.0, "knots": [2.0, 1.0, 0.0, 1.0, 2.0]}
        ]
      },
      "density": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 1.5, "rmax": 6.0, "knots": [1.0, 0.5, 0.0, 0.0, 0.0]}
        ]
      },
      "embedding": {
        "format": "analytic",
        "potentials": [
          {"type": "sqrt", "rmin": 0.0, "rmax": 5.0, "A": -1.0, "B": 0.0}
        ]
      }
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<EAMForceCalculator>(r.model));
    const auto& calc = std::get<EAMForceCalculator>(r.model);
    EXPECT_EQ(calc.ntypes, 1);
    // pair: middle knot at r=3.75 should be near 0
    const auto& phi00 = calc.pair[0, 0];
    EXPECT_NEAR(phi00.eval(3.75), 0.0, 1e-2);
    // density: positive at rmin end
    EXPECT_GT(calc.density[0].eval(1.5), 0.0);
}

TEST(ForceModelReader, EAM_TwoTypes_PairColCount) {
    // ntypes=2 → paircol=3 pair potentials, 2 density, 2 embedding
    auto r = run(R"({
      "model": "eam",
      "ntypes": 2,
      "pair": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 1.5, "rmax": 5.0, "knots": [1.0, 0.0, 1.0]},
          {"rmin": 1.5, "rmax": 5.0, "knots": [0.5, 0.0, 0.5]},
          {"rmin": 1.5, "rmax": 5.0, "knots": [0.8, 0.0, 0.8]}
        ]
      },
      "density": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 1.5, "rmax": 5.0, "knots": [1.0, 0.5, 0.0]},
          {"rmin": 1.5, "rmax": 5.0, "knots": [0.8, 0.4, 0.0]}
        ]
      },
      "embedding": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 0.0, "rmax": 3.0, "knots": [0.0, -1.0, -1.5]},
          {"rmin": 0.0, "rmax": 3.0, "knots": [0.0, -0.8, -1.2]}
        ]
      }
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    const auto& calc = std::get<EAMForceCalculator>(r.model);
    EXPECT_EQ(calc.ntypes, 2);
}

// ── tersoff model ─────────────────────────────────────────────────────────────

TEST(ForceModelReader, Tersoff_Si_SingleType) {
    // Tersoff (1988) Si parameters
    auto r = run(R"({
      "model": "tersoff",
      "ntypes": 1,
      "potentials": [
        {
          "A": 1830.8,  "B": 471.18,
          "lambda": 2.4799, "mu": 1.7322,
          "beta": 1.1e-6, "n": 0.78734,
          "c": 100390.0, "d": 16.218, "h": -0.59825,
          "R": 2.7, "S": 3.0
        }
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<TersoffForceCalculator>(r.model));
    const auto& calc = std::get<TersoffForceCalculator>(r.model);
    EXPECT_EQ(calc.ntypes, 1);
    const auto& tp = calc.params[0, 0];
    EXPECT_DOUBLE_EQ(tp.A,      1830.8);
    EXPECT_DOUBLE_EQ(tp.B,      471.18);
    EXPECT_DOUBLE_EQ(tp.lambda, 2.4799);
    EXPECT_DOUBLE_EQ(tp.R,      2.7);
    EXPECT_DOUBLE_EQ(tp.S,      3.0);
}

// ── stiweb model ──────────────────────────────────────────────────────────────

TEST(ForceModelReader, StiWeb_Si_SingleType) {
    // Stillinger-Weber (1985) Si parameters
    auto r = run(R"({
      "model": "stiweb",
      "ntypes": 1,
      "potentials": [
        {
          "A": 7.0496, "B": 0.6022,
          "p": 4.0, "q": 0.0,
          "a": 1.80, "sigma": 2.0951,
          "lambda": 21.0, "gamma": 1.20
        }
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<StiwebForceCalculator>(r.model));
    const auto& calc = std::get<StiwebForceCalculator>(r.model);
    EXPECT_EQ(calc.ntypes, 1);
    const auto& sp = calc.params[0, 0];
    EXPECT_DOUBLE_EQ(sp.A,      7.0496);
    EXPECT_DOUBLE_EQ(sp.B,      0.6022);
    EXPECT_DOUBLE_EQ(sp.sigma,  2.0951);
    EXPECT_DOUBLE_EQ(sp.lambda, 21.0);
}

// ── angular model ─────────────────────────────────────────────────────────────

TEST(ForceModelReader, Angular_SingleType) {
    auto r = run(R"({
      "model": "angular",
      "ntypes": 1,
      "pair": {
        "format": "analytic",
        "potentials": [{"type": "morse", "rmin": 1.5, "rmax": 5.0, "De": 1.0, "a": 1.5, "re": 2.5}]
      },
      "radial": {
        "format": "analytic",
        "potentials": [{"type": "exp_decay", "rmin": 1.5, "rmax": 5.0, "A": 1.0, "B": 1.0}]
      },
      "angular": {
        "format": "tabulated",
        "potentials": [{"rmin": -1.0, "rmax": 1.0, "knots": [0.0, -0.5, 0.0, -0.5, 0.0]}]
      }
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<AngularForceCalculator>(r.model));
    const auto& calc = std::get<AngularForceCalculator>(r.model);
    EXPECT_EQ(calc.ntypes, 1);
}

// ── error paths ───────────────────────────────────────────────────────────────

TEST(ForceModelReader, Error_MissingModelKey) {
    auto r = run(R"({"format": "analytic", "potentials": []})");
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.message.find("model"), std::string::npos);
}

TEST(ForceModelReader, Error_UnknownModel) {
    auto r = run(R"({"model": "meam", "ntypes": 1})");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(ForceModelReader, Error_EAM_MissingSection) {
    auto r = run(R"({
      "model": "eam",
      "ntypes": 1,
      "pair": {"format": "tabulated", "potentials": [{"rmin": 1.0, "rmax": 3.0, "knots": [1.0, 0.0, 1.0]}]},
      "density": {"format": "tabulated", "potentials": [{"rmin": 1.0, "rmax": 3.0, "knots": [1.0, 0.5, 0.0]}]}
    })");
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.message.find("embedding"), std::string::npos);
}

TEST(ForceModelReader, Error_EAM_WrongPairCount) {
    // ntypes=2 needs 3 pair potentials, providing only 1
    auto r = run(R"({
      "model": "eam",
      "ntypes": 2,
      "pair": {
        "format": "tabulated",
        "potentials": [{"rmin": 1.0, "rmax": 3.0, "knots": [1.0, 0.0, 1.0]}]
      },
      "density": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 1.0, "rmax": 3.0, "knots": [1.0, 0.5, 0.0]},
          {"rmin": 1.0, "rmax": 3.0, "knots": [0.8, 0.4, 0.0]}
        ]
      },
      "embedding": {
        "format": "tabulated",
        "potentials": [
          {"rmin": 0.0, "rmax": 2.0, "knots": [0.0, -1.0, -1.5]},
          {"rmin": 0.0, "rmax": 2.0, "knots": [0.0, -0.8, -1.2]}
        ]
      }
    })");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(ForceModelReader, Error_Tersoff_MissingParam) {
    auto r = run(R"({
      "model": "tersoff",
      "ntypes": 1,
      "potentials": [
        {"A": 1830.8, "B": 471.18}
      ]
    })");
    EXPECT_FALSE(r.ok);
}

TEST(ForceModelReader, Error_InvalidJson) {
    auto r = run("not valid json {{{");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}
