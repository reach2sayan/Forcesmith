#include "potfit/io/potential_reader.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace leaf = boost::leaf;
using namespace potfit;
using namespace potfit::io;

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

// ── Format 3 happy-path ───────────────────────────────────────────────────────

TEST(PotentialReader, Format3_TwoFunctions) {
    auto r = run(
        "#F 3 2\n"
        "#T PAIR\n"
        "#C Cu Cu\n"
        "#E\n"
        // distance block: rmin rmax nknots
        "1.0 3.0 3\n"
        "2.0 4.0 3\n"
        // values for function 0
        "0.5\n" "0.0\n" "0.5\n"
        // values for function 1
        "1.0\n" "0.0\n" "1.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 2u);

    auto [lo0, hi0] = r.pots[0].span();
    EXPECT_DOUBLE_EQ(lo0, 1.0);
    EXPECT_DOUBLE_EQ(hi0, 3.0);

    auto [lo1, hi1] = r.pots[1].span();
    EXPECT_DOUBLE_EQ(lo1, 2.0);
    EXPECT_DOUBLE_EQ(hi1, 4.0);
}

TEST(PotentialReader, Format3_KnotValuesRoundTrip) {
    auto r = run(
        "#F 3 1\n"
        "#E\n"
        "0.0 4.0 5\n"   // knots at 0, 1, 2, 3, 4
        "2.0\n" "1.0\n" "0.0\n" "1.0\n" "2.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 1u);
    const auto& p = r.pots[0];
    EXPECT_NEAR(p.eval(0.0), 2.0, 1e-10);
    EXPECT_NEAR(p.eval(1.0), 1.0, 1e-10);
    EXPECT_NEAR(p.eval(2.0), 0.0, 1e-10);
    EXPECT_NEAR(p.eval(3.0), 1.0, 1e-10);
    EXPECT_NEAR(p.eval(4.0), 2.0, 1e-10);
}

// ── Format 0 happy-path ───────────────────────────────────────────────────────

TEST(PotentialReader, Format0_PairLJ) {
    // pair_lj: V(r) = 4eps[(sig/r)^12 - (sig/r)^6]
    // minimum at r = 2^(1/6)*sig = -eps
    auto r = run(
        "#F 0 1\n"
        "#E\n"
        "type pair_lj\n"
        "2.0 6.0\n"
        "param epsilon 1.0 0.5 2.0\n"
        "param sigma   2.5 2.0 3.0\n"
    );
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

TEST(PotentialReader, Format0_Morse) {
    // V(r) = De*(1 - exp(-a*(r-re)))^2 - De
    // minimum at r=re is -De
    auto r = run(
        "#F 0 1\n"
        "#E\n"
        "type morse\n"
        "1.5 5.0\n"
        "param De 2.0 0.5 4.0\n"
        "param a  1.5 0.5 3.0\n"
        "param re 2.5 2.0 3.0\n"
    );
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_EQ(r.pots.size(), 1u);

    const double De = 2.0, re = 2.5;
    EXPECT_NEAR(r.pots[0].eval(re), -De, 1e-12);
    EXPECT_NEAR(r.pots[0].deriv(re), 0.0, 1e-12);
}

// ── Error paths ───────────────────────────────────────────────────────────────

TEST(PotentialReader, Error_MissingFHeader) {
    auto r = run("#E\n");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}

TEST(PotentialReader, Error_UnknownFormat) {
    auto r = run("#F 5 1\n#E\n");
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.message.find("unsupported"), std::string::npos);
}

TEST(PotentialReader, Error_UnknownAnalyticFunction) {
    auto r = run(
        "#F 0 1\n#E\n"
        "type born_mayer\n"
        "1.0 6.0\n"
    );
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.message.find("unknown analytic function"), std::string::npos);
}

TEST(PotentialReader, Error_KnotCountMismatch) {
    auto r = run(
        "#F 3 1\n#E\n"
        "0.0 4.0 5\n"   // declares 5 knots
        "1.0\n" "2.0\n" "3.0\n"  // but only 3 values then EOF
    );
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
}
