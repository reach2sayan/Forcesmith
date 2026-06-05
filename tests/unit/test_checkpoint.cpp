#include "forcesmith/core/checkpoint.hpp"
#include "forcesmith/io/config_reader.hpp"       // forcesmith::io::ParseError
#include "forcesmith/io/force_model_reader.hpp"  // forcesmith::io::parse_force_model
#include "forcesmith/potentials/analytic_potential.hpp"
#include "forcesmith/potentials/spline.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <ranges>
#include <variant>

namespace leaf = boost::leaf;
using namespace forcesmith;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Configuration make_test_config() {
    Configuration cfg;
    cfg.bc     = PeriodicBC(5.0 * Mat3::Identity());
    cfg.ref.energy = -1.23;
    cfg.weight = 2.0;
    cfg.ref.stress(0, 0) = 0.1;
    Atom a;
    a.type  = 0;
    a.pos   = Vec3(1.0, 2.0, 3.0);
    a.ref.force = Vec3(0.1, -0.2, 0.3);
    cfg.atoms.push_back(a);
    return cfg;
}

// A single-type pair model owning the given potentials.
static ForceCalculator pair_model(std::vector<Potential> pots) {
    return make_pair_force_calculator(std::move(pots));
}

// RAII temp directory.
struct TmpDir {
    std::filesystem::path path;
    TmpDir() {
        auto ns = static_cast<long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path = std::filesystem::temp_directory_path() /
               ("forcesmith_ckpt_" + std::to_string(ns));
        std::filesystem::create_directories(path);
    }
    ~TmpDir() { std::filesystem::remove_all(path); }
    std::filesystem::path prefix(const std::string& name) const { return path / name; }
};

// Run save + load and return true on success, filling configs_out/model_out.
static bool round_trip(const std::filesystem::path& pfx,
                       const std::vector<Configuration>& configs_in,
                       const ForceCalculator& model_in,
                       std::vector<Configuration>& configs_out,
                       ForceCalculator& model_out)
{
    bool ok = true;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            if (auto r = CheckpointWriter(pfx).configs(configs_in).model(model_in).write(); !r)
                return r;
            if (auto r = CheckpointReader(pfx).read(configs_out, model_out); !r)
                return r;
            return {};
        },
        [&](const CheckpointError& e)    { ADD_FAILURE() << "CheckpointError: " << e.message; ok = false; },
        [&](const forcesmith::io::ParseError& e) { ADD_FAILURE() << "ParseError: " << e.message; ok = false; },
        [&]()                               { ADD_FAILURE() << "unknown error";              ok = false; }
    );
    return ok;
}

// First pair potential φ_00 of a round-tripped pair model.
static const Potential& pair00(const ForceCalculator& m) {
    return std::get<PairForceCalculator>(m).pair[0, 0];
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(Checkpoint, SplinePotentialRoundTrip) {
    TmpDir tmp;
    std::vector<Configuration> configs_in = { make_test_config() };
    SplinePotential spline({1.0, 2.0, 3.0, 4.0}, {2.0, 0.5, 0.5, 2.0});
    ForceCalculator model_in = pair_model({Potential(spline)});

    std::vector<Configuration> cfgs2;
    ForceCalculator           model2;
    ASSERT_TRUE(round_trip(tmp.prefix("sp"), configs_in, model_in, cfgs2, model2));

    // Configuration fields preserved.
    ASSERT_EQ(cfgs2.size(), 1u);
    EXPECT_NEAR(cfgs2[0].ref.energy, -1.23, 1e-12);
    EXPECT_NEAR(cfgs2[0].weight, 2.0,   1e-12);
    ASSERT_EQ(cfgs2[0].atoms.size(), 1u);
    EXPECT_NEAR(cfgs2[0].atoms[0].pos.x(), 1.0,  1e-12);
    EXPECT_NEAR(cfgs2[0].atoms[0].ref.force.z(), 0.3, 1e-12);

    // Potential values reproduced within grid sampling tolerance.
    ASSERT_TRUE(std::holds_alternative<PairForceCalculator>(model2));
    for (double r : {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0})
        EXPECT_NEAR(pair00(model2).eval(r), spline.eval(r), 1e-3) << "at r=" << r;
}

TEST(Checkpoint, AnalyticLJRoundTrip) {
    TmpDir tmp;
    std::vector<Configuration> configs_in = { make_test_config() };
    const double eps = 1.5, sigma = 2.0;
    LennardJones lj(eps, sigma, sigma * 0.8, sigma * 3.0);
    ForceCalculator model_in = pair_model({Potential(lj)});

    std::vector<Configuration> cfgs2;
    ForceCalculator           model2;
    ASSERT_TRUE(round_trip(tmp.prefix("lj"), configs_in, model_in, cfgs2, model2));

    ASSERT_TRUE(std::holds_alternative<PairForceCalculator>(model2));
    auto [lo, hi] = pair00(model2).span();
    EXPECT_NEAR(lo, sigma * 0.8, 1e-10);
    EXPECT_NEAR(hi, sigma * 3.0, 1e-10);
    for (double r = lo + 0.1; r < hi - 0.1; r += 0.3)
        EXPECT_NEAR(pair00(model2).eval(r), lj.eval(r), 1e-3) << "at r=" << r;
}

// EAM is a multi-table model (pair + density + embedding); the legacy checkpoint
// format could not represent it. Confirm a full round-trip preserves every table.
TEST(Checkpoint, EAMModelRoundTrip) {
    TmpDir tmp;
    std::vector<Configuration> configs_in = { make_test_config() };

    auto parsed = io::parse_force_model(R"({
      "model": "eam",
      "ntypes": 1,
      "pair":      { "format": "tabulated",
        "potentials": [ {"rmin": 1.5, "rmax": 6.0, "knots": [2.0, 1.0, 0.0, 1.0, 2.0]} ] },
      "density":   { "format": "tabulated",
        "potentials": [ {"rmin": 1.5, "rmax": 6.0, "knots": [1.0, 0.5, 0.0, 0.0, 0.0]} ] },
      "embedding": { "format": "tabulated",
        "potentials": [ {"rmin": 0.0, "rmax": 5.0, "knots": [0.0, -0.7, -1.0, -1.2, -1.4]} ] }
    })");
    ASSERT_TRUE(parsed) << "failed to parse EAM model";
    ForceCalculator model_in = parsed.value();

    std::vector<Configuration> cfgs2;
    ForceCalculator           model2;
    ASSERT_TRUE(round_trip(tmp.prefix("eam"), configs_in, model_in, cfgs2, model2));

    ASSERT_TRUE(std::holds_alternative<EAMForceCalculator>(model2));
    const auto& in  = std::get<EAMForceCalculator>(model_in);
    const auto& out = std::get<EAMForceCalculator>(model2);
    EXPECT_EQ(out.ntypes, in.ntypes);

    // Every table reproduced within tabulation tolerance.
    const Potential& out_phi = out.pair[0, 0];
    const Potential& in_phi  = in.pair[0, 0];
    for (double r : {1.5, 2.5, 3.75, 5.0, 6.0}) {
        EXPECT_NEAR(out_phi.eval(r), in_phi.eval(r), 1e-3) << "pair r=" << r;
        EXPECT_NEAR(out.density[0].eval(r), in.density[0].eval(r), 1e-3) << "density r=" << r;
    }
    for (double rho : {0.1, 0.5, 1.0, 2.0})
        EXPECT_NEAR(out.embedding[0].eval(rho), in.embedding[0].eval(rho), 1e-3) << "embed rho=" << rho;
}

TEST(Checkpoint, BoxMatrixPreserved) {
    TmpDir tmp;
    Mat3 box;
    box << 3.0, 0.1, 0.0,
           0.0, 4.0, 0.2,
           0.0, 0.0, 5.0;
    Configuration cfg;
    cfg.bc = PeriodicBC(box);
    cfg.ref.energy = 0.0;
    Atom a; a.type = 0; a.pos = Vec3::Zero();
    cfg.atoms.push_back(a);

    ForceCalculator model_in = pair_model({Potential(SplinePotential({1.0, 2.0}, {0.0, 1.0}))});

    std::vector<Configuration> cfgs2;
    ForceCalculator           model2;
    ASSERT_TRUE(round_trip(tmp.prefix("box"), {cfg}, model_in, cfgs2, model2));

    ASSERT_EQ(cfgs2.size(), 1u);
    const auto& bc2 = std::get<PeriodicBC>(cfgs2[0].bc);
    EXPECT_NEAR(bc2.box()(0, 0), 3.0, 1e-12);  // row 0, col 0
    EXPECT_NEAR(bc2.box()(0, 1), 0.1, 1e-12);  // row 0, col 1 (filled by <<)
    EXPECT_NEAR(bc2.box()(1, 2), 0.2, 1e-12);  // row 1, col 2
    EXPECT_NEAR(bc2.box()(2, 2), 5.0, 1e-12);  // row 2, col 2
}

TEST(Checkpoint, MultipleConfigurations) {
    TmpDir tmp;
    std::vector<Configuration> configs_in;
    for (int i : std::views::iota(0, 3)) {
        auto cfg = make_test_config();
        cfg.ref.energy = static_cast<double>(i) * -0.5;
        cfg.weight = static_cast<double>(i + 1);
        configs_in.push_back(cfg);
    }
    ForceCalculator model_in = pair_model({Potential(SplinePotential({0.5, 1.5, 2.5}, {1.0, 0.0, 1.0}))});

    std::vector<Configuration> cfgs2;
    ForceCalculator           model2;
    ASSERT_TRUE(round_trip(tmp.prefix("multi"), configs_in, model_in, cfgs2, model2));

    ASSERT_EQ(cfgs2.size(), 3u);
    for (auto [i, cfg] : std::views::enumerate(cfgs2)) {
        EXPECT_NEAR(cfg.ref.energy, static_cast<double>(i) * -0.5, 1e-12);
        EXPECT_NEAR(cfg.weight, static_cast<double>(i + 1),    1e-12);
    }
}
