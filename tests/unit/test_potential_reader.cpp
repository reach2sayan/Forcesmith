#include "forcesmith/io/potential_reader.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace leaf = boost::leaf;
using namespace forcesmith;
using namespace forcesmith::io;

// ── Helper ────────────────────────────────────────────────────────────────────

struct PotResult {
    bool ok = false;
    std::vector<Potential> pots;
    ParseError error;
};

static PotResult run(std::string_view input) {
    PotResult out;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(p, parse_potential(input));
            out.ok   = true;
            out.pots = std::move(p);
            return {};
        },
        [&](const ParseError& e) { out.ok = false; out.error = e; },
        [&]()                    { out.ok = false; out.error.message = "unknown error"; }
    );
    return out;
}

// ── Tabulated happy-path ──────────────────────────────────────────────────────

TEST(PotentialReader, Tabulated_TwoFunctions) {
    auto r = run(R"({
      "format": "tabulated",
      "potentials": [
        {"rmin": 1.0, "rmax": 3.0, "knots": [0.5, 0.0, 0.5]},
        {"rmin": 2.0, "rmax": 4.0, "knots": [1.0, 0.0, 1.0]}
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 2u);

    auto [lo0, hi0] = r.pots[0].span();
    EXPECT_DOUBLE_EQ(lo0, 1.0);
    EXPECT_DOUBLE_EQ(hi0, 3.0);

    auto [lo1, hi1] = r.pots[1].span();
    EXPECT_DOUBLE_EQ(lo1, 2.0);
    EXPECT_DOUBLE_EQ(hi1, 4.0);
}

TEST(PotentialReader, Tabulated_KnotValuesRoundTrip) {
    auto r = run(R"({
      "format": "tabulated",
      "potentials": [
        {"rmin": 0.0, "rmax": 4.0, "knots": [2.0, 1.0, 0.0, 1.0, 2.0]}
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 1u);
    const auto& p = r.pots[0];
    EXPECT_NEAR(p.eval(0.0), 2.0, 1e-10);
    EXPECT_NEAR(p.eval(1.0), 1.0, 1e-10);
    EXPECT_NEAR(p.eval(2.0), 0.0, 1e-10);
    EXPECT_NEAR(p.eval(3.0), 1.0, 1e-10);
    EXPECT_NEAR(p.eval(4.0), 2.0, 1e-10);
}

// ── Analytic happy-path ───────────────────────────────────────────────────────

TEST(PotentialReader, Analytic_PairLJ) {
    // V(r) = 4ε[(σ/r)^12 − (σ/r)^6]; minimum at r = 2^(1/6)σ equals −ε
    auto r = run(R"({
      "format": "analytic",
      "potentials": [
        {
          "type": "pair_lj",
          "rmin": 2.0, "rmax": 6.0,
          "epsilon": 1.0,
          "sigma": 2.5
        }
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 1u);

    auto [lo, hi] = r.pots[0].span();
    EXPECT_DOUBLE_EQ(lo, 2.0);
    EXPECT_DOUBLE_EQ(hi, 6.0);

    const double eps = 1.0, sig = 2.5;
    const double r_eq = sig * std::pow(2.0, 1.0 / 6.0);
    EXPECT_NEAR(r.pots[0].eval(r_eq), -eps, 1e-12);
    EXPECT_NEAR(r.pots[0].deriv(r_eq), 0.0, 1e-10);
}

TEST(PotentialReader, Analytic_Morse) {
    // V(r) = De*(1 − exp(−a*(r−re)))^2 − De; minimum at r=re equals −De
    auto r = run(R"({
      "format": "analytic",
      "potentials": [
        {
          "type": "morse",
          "rmin": 1.5, "rmax": 5.0,
          "De": 2.0,
          "a": 1.5,
          "re": 2.5
        }
      ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 1u);

    const double De = 2.0, re = 2.5;
    EXPECT_NEAR(r.pots[0].eval(re), -De, 1e-12);
    EXPECT_NEAR(r.pots[0].deriv(re), 0.0, 1e-12);
}

// ── Error paths ───────────────────────────────────────────────────────────────

TEST(PotentialReader, Error_MissingFormatKey) {
    auto r = run(R"({"potentials": []})");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(PotentialReader, Error_UnknownFormat) {
    auto r = run(R"({"format": "xyz", "potentials": []})");
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.message.find("unsupported"), std::string::npos);
}

TEST(PotentialReader, Error_UnknownAnalyticFunction) {
    auto r = run(R"({
      "format": "analytic",
      "potentials": [
        {"type": "born_mayer", "rmin": 1.0, "rmax": 6.0}
      ]
    })");
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.message.find("unknown analytic function"), std::string::npos);
}

TEST(PotentialReader, Error_KnotNotANumber) {
    auto r = run(R"({
      "format": "tabulated",
      "potentials": [
        {"rmin": 0.0, "rmax": 4.0, "knots": [1.0, "bad", 3.0]}
      ]
    })");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}
