#include "potfit/io/force_model_reader.hpp"
#include "potfit/io/write_model.hpp"
#include "potfit/potentials/symmetry_functions.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace leaf = boost::leaf;
using namespace potfit;
using namespace potfit::io;

// BOOST_LEAF_CHECK expands to a GNU statement-expression ({ ... }); silence the
// pedantic complaint about that Boost idiom for this translation unit.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

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
    // potfit Stillinger-Weber parameterization: per-pair 2-/3-body params plus
    // a per-triplet lambda array (ntypes·paircol = 1 entry for one type).
    auto r = run(R"({
      "model": "stiweb",
      "ntypes": 1,
      "potentials": [
        {
          "A": 7.0496, "B": 0.6022,
          "p": 4.0, "q": 0.0,
          "delta": 2.0951, "a1": 3.7712,
          "gamma": 2.5141, "a2": 3.7712
        }
      ],
      "lambda": [45.534]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<StiwebForceCalculator>(r.model));
    const auto& calc = std::get<StiwebForceCalculator>(r.model);
    EXPECT_EQ(calc.ntypes, 1);
    const auto& sp = calc.params[0, 0];
    EXPECT_DOUBLE_EQ(sp.A,     7.0496);
    EXPECT_DOUBLE_EQ(sp.B,     0.6022);
    EXPECT_DOUBLE_EQ(sp.delta, 2.0951);
    EXPECT_DOUBLE_EQ(sp.a1,    3.7712);
    EXPECT_DOUBLE_EQ(sp.gamma, 2.5141);
    EXPECT_DOUBLE_EQ(sp.a2,    3.7712);
    ASSERT_EQ(calc.lambda.size(), 1u);
    EXPECT_DOUBLE_EQ(double(calc.lambda_at(0, 0, 0)), 45.534);
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

// A bare potential section (no "model" wrapper), as written by io::write_native,
// is accepted as a pair model with ntypes inferred from the potential count.
TEST(ForceModelReader, BarePairSection_TreatedAsPairModel) {
    auto r = run(R"({
      "format": "tabulated",
      "potentials": [ {"rmin": 1.5, "rmax": 6.0, "knots": [2.0, 1.0, 0.0, 1.0, 2.0]} ]
    })");
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<PairForceCalculator>(r.model));
    EXPECT_EQ(std::get<PairForceCalculator>(r.model).ntypes, 1u);
}

// A bare section whose potential count is not a valid paircol (n*(n+1)/2) errors.
TEST(ForceModelReader, BarePairSection_InvalidCount) {
    auto r = run(R"({
      "format": "tabulated",
      "potentials": [
        {"rmin": 1.5, "rmax": 6.0, "knots": [2.0, 1.0, 0.0]},
        {"rmin": 1.5, "rmax": 6.0, "knots": [1.0, 0.5, 0.0]}
      ]
    })");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.message.empty());
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

// ── global parameters (shared smooth-cutoff h) ──────────────────────────────

// A 1-type EAM where the smooth-cutoff h is one shared global used by both the
// morse_sc pair and the exp_decay_sc density. Guards the parameter bookkeeping
// (the single highest-risk part of the global-parameter machinery).
static constexpr const char *kEamGlobalH = R"({
  "model": "eam", "ntypes": 1,
  "globals": { "h": {"value": 1.0, "min": 0.5, "max": 2.0} },
  "pair":      {"format":"analytic","potentials":[
    {"type":"morse_sc","rmin":0.0,"rmax":6.0,"De":0.4,"a":1.2,"re":2.86,"h":{"global":"h"}}]},
  "density":   {"format":"analytic","potentials":[
    {"type":"exp_decay_sc","rmin":0.0,"rmax":6.0,"A":1.0,"B":1.0,"h":{"global":"h"}}]},
  "embedding": {"format":"analytic","potentials":[
    {"type":"sqrt","rmin":0.0,"rmax":5.0,"A":-1.0,"B":1.0}]}
})";

TEST(ForceModelReader, EAM_GlobalH_ParsedAndCounted) {
    auto r = run(kEamGlobalH);
    ASSERT_TRUE(r.ok) << r.error.message;
    ASSERT_TRUE(std::holds_alternative<EAMForceCalculator>(r.model));
    const auto& calc = std::get<EAMForceCalculator>(r.model);

    ASSERT_EQ(calc.globals.size(), 1u);
    EXPECT_EQ(calc.globals[0].links.size(), 2u); // morse_sc.h + exp_decay_sc.h

    // free params: morse_sc 3 (h fixed) + exp_decay_sc 2 (h fixed) + sqrt 2
    //              + 1 shared global = 8
    EXPECT_EQ(calc.param_count(), 8u);
}

TEST(ForceModelReader, EAM_GlobalH_GatherScatterRoundTrip) {
    auto r = run(kEamGlobalH);
    ASSERT_TRUE(r.ok) << r.error.message;
    auto& calc = std::get<EAMForceCalculator>(r.model);

    Eigen::VectorXd x(static_cast<int>(calc.param_count()));
    calc.gather_params(x, 0);
    const Eigen::VectorXd x0 = x;
    calc.scatter_params(x, 0);
    Eigen::VectorXd x2(static_cast<int>(calc.param_count()));
    calc.gather_params(x2, 0);
    EXPECT_TRUE(x0.isApprox(x2)) << "gather→scatter→gather must be identity";
}

TEST(ForceModelReader, EAM_GlobalH_BroadcastReachesAllLinks) {
    auto r = run(kEamGlobalH);
    ASSERT_TRUE(r.ok) << r.error.message;
    auto& calc = std::get<EAMForceCalculator>(r.model);

    const auto& phi = calc.pair[0, 0];   // morse_sc, linked to global h
    const auto& rho = calc.density[0];   // exp_decay_sc, linked to global h
    const double rt = 5.5;               // near cutoff 6.0 → apot_cutoff(h) active
    const double phi0 = phi.eval(rt);
    const double rho0 = rho.eval(rt);

    // The global h is the last gathered slot; perturb it and scatter back.
    Eigen::VectorXd x(static_cast<int>(calc.param_count()));
    calc.gather_params(x, 0);
    x(x.size() - 1) = 1.7;               // new shared h
    calc.scatter_params(x, 0);

    // Both linked potentials must see the new h (their cutoff factor changed).
    EXPECT_NE(phi.eval(rt), phi0);
    EXPECT_NE(rho.eval(rt), rho0);
}

// ── write_model round-trip / robustness ────────────────────────────────────────

namespace {
// Unique temp model path (RAII cleanup).
struct TmpModel {
    std::filesystem::path path;
    explicit TmpModel(const char* tag) {
        auto ns = static_cast<long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path = std::filesystem::temp_directory_path() /
               ("potfit_fm_" + std::string(tag) + "_" + std::to_string(ns) + ".json");
    }
    ~TmpModel() { std::error_code ec; std::filesystem::remove(path, ec); }
};

// write_model(model, path, "native") → result.
static leaf::result<void> write_native_model(const ForceCalculator& m,
                                             const std::filesystem::path& p) {
    return write_model(m, p, "native");
}
} // namespace

// A valid analytic embedding (sqrt with a nonzero scale) writes finite knots and
// parses back as a well-formed EAM model — no `null` in the output.
TEST(WriteModel, EAM_AnalyticEmbedding_RoundTrips) {
    auto parsed = run(R"({
      "model": "eam", "ntypes": 1,
      "pair":      { "format": "tabulated", "potentials": [ {"rmin":1.5,"rmax":6.0,"knots":[2.0,1.0,0.0,1.0,2.0]} ] },
      "density":   { "format": "tabulated", "potentials": [ {"rmin":1.5,"rmax":6.0,"knots":[1.0,0.5,0.0,0.0,0.0]} ] },
      "embedding": { "format": "analytic", "potentials": [ {"type":"sqrt","rmin":0.0,"rmax":5.0,"A":-1.0,"B":1.0} ] }
    })");
    ASSERT_TRUE(parsed.ok) << parsed.error.message;

    TmpModel tmp("emb_ok");
    bool wrote = false, reok = false;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_CHECK(write_native_model(parsed.model, tmp.path));
            wrote = true;
            std::ifstream f(tmp.path);
            const std::string s{std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>{}};
            EXPECT_EQ(s.find("null"), std::string::npos) << "writer emitted null knots";
            BOOST_LEAF_AUTO(m, parse_force_model(s));
            reok = std::holds_alternative<EAMForceCalculator>(m);
            return {};
        },
        [&](const ParseError& e) { ADD_FAILURE() << "ParseError: " << e.message; },
        [&]()                    { ADD_FAILURE() << "unknown error"; });
    EXPECT_TRUE(wrote);
    EXPECT_TRUE(reok);
}

// A degenerate analytic embedding (sqrt scale B=0 → A*sqrt(r/0) = inf) must make
// the writer FAIL with a clear error rather than silently emit `null` knots that
// can never be parsed back.
TEST(WriteModel, NonFiniteKnots_FailsLoudly) {
    auto parsed = run(R"({
      "model": "eam", "ntypes": 1,
      "pair":      { "format": "tabulated", "potentials": [ {"rmin":1.5,"rmax":6.0,"knots":[2.0,1.0,0.0,1.0,2.0]} ] },
      "density":   { "format": "tabulated", "potentials": [ {"rmin":1.5,"rmax":6.0,"knots":[1.0,0.5,0.0,0.0,0.0]} ] },
      "embedding": { "format": "analytic", "potentials": [ {"type":"sqrt","rmin":0.5,"rmax":5.0,"A":-1.0,"B":0.0} ] }
    })");
    ASSERT_TRUE(parsed.ok) << parsed.error.message;

    TmpModel tmp("emb_bad");
    bool failed = false;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_CHECK(write_native_model(parsed.model, tmp.path));
            return {};
        },
        [&](const ParseError& e) {
            failed = true;
            EXPECT_NE(e.message.find("non-finite"), std::string::npos) << e.message;
        },
        [&]() { ADD_FAILURE() << "expected a ParseError, got unknown error"; });
    EXPECT_TRUE(failed) << "write_model should reject non-finite tabulation";
}

// An ML model's per-feature standardization (computed by prepare()) must survive a
// native write → read so a reloaded potential predicts identically.
TEST(WriteModel, ML_Standardization_RoundTrips) {
    SymmetryFunctionModel sf;
    sf.ntypes = 1;
    sf.rcut = 6.0;
    sf.standardize_features = true; // off by default; this test exercises it
    sf.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
    sf.heads.reserve(1);
    sf.heads.emplace_back(EnergyHead{MLPHead::make({3, 5, 1}, MLPHead::Act::Tanh, 2)});

    std::vector<Configuration> cfgs;
    {
        Configuration c;
        c.bc = PeriodicBC(100.0 * Mat3::Identity());
        Atom a0, a1, a2, a3;
        a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
        a1.type = 0; a1.pos = {2.1, 0.3, -0.2};
        a2.type = 0; a2.pos = {0.4, 2.0, 0.5};
        a3.type = 0; a3.pos = {-1.3, 0.7, 1.9};
        c.atoms = {a0, a1, a2, a3};
        cfgs.push_back(c);
    }
    sf.prepare(std::span<Configuration>(cfgs.data(), cfgs.size()));
    ASSERT_TRUE(sf.has_standardization());
    const Eigen::VectorXd mean0 = sf.mean_[0];
    const Eigen::VectorXd iv0 = sf.inv_std_[0];

    ForceCalculator fc{std::move(sf)};
    TmpModel tmp("ml_std");
    bool ok = false;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_CHECK(write_native_model(fc, tmp.path));
            std::ifstream f(tmp.path);
            const std::string s{std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>{}};
            EXPECT_NE(s.find("standardization"), std::string::npos);
            BOOST_LEAF_AUTO(m, parse_force_model(s));
            if (!std::holds_alternative<SymmetryFunctionModel>(m)) {
                ADD_FAILURE() << "expected SF model after reload";
                return {};
            }
            const auto& rsf = std::get<SymmetryFunctionModel>(m);
            if (rsf.inv_std_.size() == 0) {
                ADD_FAILURE() << "standardization not restored";
                return {};
            }
            EXPECT_TRUE(rsf.mean_[0].isApprox(mean0));
            EXPECT_TRUE(rsf.inv_std_[0].isApprox(iv0));
            ok = true;
            return {};
        },
        [&](const ParseError& e) { ADD_FAILURE() << "ParseError: " << e.message; },
        [&]()                    { ADD_FAILURE() << "unknown error"; });
    EXPECT_TRUE(ok);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
