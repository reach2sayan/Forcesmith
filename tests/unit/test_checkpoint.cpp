#include "potfit/core/checkpoint.hpp"
#include "potfit/io/config_reader.hpp"  // potfit::io::ParseError
#include "potfit/potentials/analytic_potential.hpp"
#include "potfit/potentials/spline.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <ranges>

namespace leaf = boost::leaf;
using namespace potfit;

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

// RAII temp directory.
struct TmpDir {
    std::filesystem::path path;
    TmpDir() {
        auto ns = static_cast<long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path = std::filesystem::temp_directory_path() /
               ("potfit_ckpt_" + std::to_string(ns));
        std::filesystem::create_directories(path);
    }
    ~TmpDir() { std::filesystem::remove_all(path); }
    std::filesystem::path prefix(const std::string& name) const { return path / name; }
};

// Run save + load and return true on success, filling out2/pots2.
static bool round_trip(const std::filesystem::path& pfx,
                       const std::vector<Configuration>& configs_in,
                       const std::vector<Potential>& pots_in,
                       std::vector<Configuration>& configs_out,
                       std::vector<Potential>& pots_out)
{
    bool ok = true;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            if (auto r = CheckpointWriter(pfx).configs(configs_in).potentials(pots_in).write(); !r)
                return r;
            if (auto r = CheckpointReader(pfx).read(configs_out, pots_out); !r)
                return r;
            return {};
        },
        [&](const CheckpointError& e)    { ADD_FAILURE() << "CheckpointError: " << e.message; ok = false; },
        [&](const potfit::io::ParseError& e) { ADD_FAILURE() << "ParseError: " << e.message; ok = false; },
        [&]()                               { ADD_FAILURE() << "unknown error";              ok = false; }
    );
    return ok;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(Checkpoint, SplinePotentialRoundTrip) {
    TmpDir tmp;
    std::vector<Configuration> configs_in = { make_test_config() };
    std::vector<Potential> pots_in;
    pots_in.emplace_back(SplinePotential({1.0, 2.0, 3.0, 4.0},
                                         {2.0, 0.5, 0.5, 2.0}));

    std::vector<Configuration> cfgs2;
    std::vector<Potential>     pots2;
    ASSERT_TRUE(round_trip(tmp.prefix("sp"), configs_in, pots_in, cfgs2, pots2));

    // Configuration fields preserved.
    ASSERT_EQ(cfgs2.size(), 1u);
    EXPECT_NEAR(cfgs2[0].ref.energy, -1.23, 1e-12);
    EXPECT_NEAR(cfgs2[0].weight, 2.0,   1e-12);
    ASSERT_EQ(cfgs2[0].atoms.size(), 1u);
    EXPECT_NEAR(cfgs2[0].atoms[0].pos.x(), 1.0,  1e-12);
    EXPECT_NEAR(cfgs2[0].atoms[0].ref.force.z(), 0.3, 1e-12);

    // Potential values reproduced within grid sampling tolerance.
    ASSERT_EQ(pots2.size(), 1u);
    for (double r : {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0})
        EXPECT_NEAR(pots2[0].eval(r), pots_in[0].eval(r), 1e-3) << "at r=" << r;
}

TEST(Checkpoint, AnalyticLJRoundTrip) {
    TmpDir tmp;
    std::vector<Configuration> configs_in = { make_test_config() };
    std::vector<Potential> pots_in;
    const double eps = 1.5, sigma = 2.0;
    pots_in.emplace_back(LennardJones(eps, sigma, sigma * 0.8, sigma * 3.0));

    std::vector<Configuration> cfgs2;
    std::vector<Potential>     pots2;
    ASSERT_TRUE(round_trip(tmp.prefix("lj"), configs_in, pots_in, cfgs2, pots2));

    ASSERT_EQ(pots2.size(), 1u);
    auto [lo, hi] = pots2[0].span();
    EXPECT_NEAR(lo, sigma * 0.8, 1e-10);
    EXPECT_NEAR(hi, sigma * 3.0, 1e-10);
    for (double r = lo + 0.1; r < hi - 0.1; r += 0.3)
        EXPECT_NEAR(pots2[0].eval(r), pots_in[0].eval(r), 1e-3) << "at r=" << r;
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

    std::vector<Potential> pots_in;
    pots_in.emplace_back(SplinePotential({1.0, 2.0}, {0.0, 1.0}));

    std::vector<Configuration> cfgs2;
    std::vector<Potential>     pots2;
    ASSERT_TRUE(round_trip(tmp.prefix("box"), {cfg}, pots_in, cfgs2, pots2));

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
    std::vector<Potential> pots_in;
    pots_in.emplace_back(SplinePotential({0.5, 1.5, 2.5}, {1.0, 0.0, 1.0}));

    std::vector<Configuration> cfgs2;
    std::vector<Potential>     pots2;
    ASSERT_TRUE(round_trip(tmp.prefix("multi"), configs_in, pots_in, cfgs2, pots2));

    ASSERT_EQ(cfgs2.size(), 3u);
    for (auto [i, cfg] : std::views::enumerate(cfgs2)) {
        EXPECT_NEAR(cfg.ref.energy, static_cast<double>(i) * -0.5, 1e-12);
        EXPECT_NEAR(cfg.weight, static_cast<double>(i + 1),    1e-12);
    }
}
