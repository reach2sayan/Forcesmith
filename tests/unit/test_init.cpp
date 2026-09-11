// Round-trip tests for the `forcesmith init` scaffolder: drive the real init
// code path to write a startpot, then load it back through the force-model
// factory and assert the model parses with the expected per-region
// cardinalities and head sizes. This proves every scaffolded file is a valid,
// loadable startpot — the contract the scaffolder exists to guarantee.

#include "forcesmith/cli/init.hpp"
#include "forcesmith/io/force_model_reader.hpp"
#include "forcesmith/io/potential_reader.hpp"
#include "forcesmith/potentials/acsf.hpp"
#include "forcesmith/potentials/analytic_param_defs.hpp"
#include "forcesmith/potentials/lmbtr.hpp"
#include "forcesmith/potentials/soap.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace leaf = boost::leaf;
using namespace forcesmith;

namespace {

// Run the scaffolder exactly as main() would: argv[0] is "init", then flags.
int run_init(std::vector<std::string> flags) {
  std::vector<std::string> argv_s{"init"};
  argv_s.insert(argv_s.end(), flags.begin(), flags.end());
  std::vector<char *> argv;
  for (auto &s : argv_s) {
    argv.push_back(s.data());
  }
  return forcesmith::cli::init::run(static_cast<int>(argv.size()), argv.data());
}

std::filesystem::path tmp_out(const std::string &tag) {
  return std::filesystem::temp_directory_path() /
         ("forcesmith_init_" + tag + ".json");
}

// Scaffold to a temp file and load it back; returns the parsed model or fails.
ForceCalculator scaffold_and_load(const std::string &tag,
                                  std::vector<std::string> flags) {
  const auto path = tmp_out(tag);
  flags.push_back("--out");
  flags.push_back(path.string());
  EXPECT_EQ(run_init(flags), 0) << "init failed for " << tag;

  std::ifstream f(path);
  EXPECT_TRUE(f.good()) << "no output file for " << tag;
  const std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

  ForceCalculator model;
  bool ok = false;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        BOOST_LEAF_AUTO(m, io::parse_force_model(text));
        model = std::move(m);
        ok = true;
        return {};
      },
      [&](const io::ParseError &e) {
        ADD_FAILURE() << tag << ": parse error: " << e.message;
      },
      [&]() { ADD_FAILURE() << tag << ": unknown parse error"; });
  EXPECT_TRUE(ok) << "could not reload " << tag;
  std::filesystem::remove(path);
  return model;
}

std::size_t paircol(std::size_t nt) { return nt * (nt + 1) / 2; }

} // namespace

// ── analytic classical: default per-region functions ────────────────────────

TEST(Init, PairLoadsForBothNtypes) {
  for (int nt : {1, 2}) {
    auto m =
        scaffold_and_load("pair" + std::to_string(nt),
                          {"--model", "pair", "--ntypes", std::to_string(nt)});
    ASSERT_TRUE(((m).target<PairForceCalculator>() != nullptr));
    EXPECT_EQ((*(m).target<PairForceCalculator>()).pair.size(),
              paircol(static_cast<std::size_t>(nt)));
  }
}

TEST(Init, EamHasPairDensityEmbedding) {
  for (int nt : {1, 2}) {
    auto m =
        scaffold_and_load("eam" + std::to_string(nt),
                          {"--model", "eam", "--ntypes", std::to_string(nt)});
    ASSERT_TRUE(((m).target<EAMForceCalculator>() != nullptr));
    const auto &e = (*(m).target<EAMForceCalculator>());
    EXPECT_EQ(e.pair.size(), paircol(static_cast<std::size_t>(nt)));
    EXPECT_EQ(e.density.size(), static_cast<std::size_t>(nt));
    EXPECT_EQ(e.embedding.size(), static_cast<std::size_t>(nt));
  }
}

TEST(Init, AdpAndAngularLoad) {
  ASSERT_TRUE(((scaffold_and_load("adp2", {"--model", "adp", "--ntypes", "2"}))
                   .target<ADPForceCalculator>() != nullptr));
  ASSERT_TRUE(
      ((scaffold_and_load("ang2", {"--model", "angular", "--ntypes", "2"}))
           .target<AngularForceCalculator>() != nullptr));
}

// ── analytic: explicit makeapot-style -f list ───────────────────────────────

TEST(Init, EamWithExplicitFunctions) {
  // ntypes=1: paircol(1)=1 → 1 pair + 1 density + 1 embedding = 3 functions.
  auto m = scaffold_and_load(
      "eam_f", {"--model", "eam", "--functions", "morse,exp_decay,sqrt"});
  ASSERT_TRUE(((m).target<EAMForceCalculator>() != nullptr));
}

TEST(Init, FunctionsCountMismatchFailsGracefully) {
  // eam ntypes=1 needs 3 functions; give 2. Expect a non-zero return and the
  // mismatch named on stderr — no crash, and no process exit: the scaffolder
  // reports through boost::leaf and run() turns that into the exit code.
  std::ostringstream captured;
  std::streambuf *const saved = std::cerr.rdbuf(captured.rdbuf());
  const int rc = run_init({"--model", "eam", "--functions", "lj,lj", "--out",
                           tmp_out("bad").string()});
  std::cerr.rdbuf(saved);

  EXPECT_EQ(rc, 1);
  EXPECT_NE(captured.str().find("needs 3"), std::string::npos)
      << captured.str();
  EXPECT_FALSE(std::filesystem::exists(tmp_out("bad")));
}

TEST(Init, UnknownFunctionIsReported) {
  std::ostringstream captured;
  std::streambuf *const saved = std::cerr.rdbuf(captured.rdbuf());
  const int rc = run_init({"--model", "pair", "--functions", "no_such_form",
                           "--out", tmp_out("unknown_fn").string()});
  std::cerr.rdbuf(saved);

  EXPECT_EQ(rc, 1);
  EXPECT_NE(captured.str().find("no_such_form"), std::string::npos)
      << captured.str();
}

TEST(Init, MalformedFunctionSpecIsReported) {
  // The spec is a grammar now: a trailing separator is a parse error naming
  // --functions, not a silently skipped empty token.
  std::ostringstream captured;
  std::streambuf *const saved = std::cerr.rdbuf(captured.rdbuf());
  const int rc = run_init({"--model", "pair", "--functions", "lj,", "--out",
                           tmp_out("bad_spec").string()});
  std::cerr.rdbuf(saved);

  EXPECT_EQ(rc, 1);
  EXPECT_NE(captured.str().find("--functions"), std::string::npos)
      << captured.str();
}

TEST(Init, UnknownModelIsReported) {
  std::ostringstream captured;
  std::streambuf *const saved = std::cerr.rdbuf(captured.rdbuf());
  const int rc = run_init(
      {"--model", "nosuchmodel", "--out", tmp_out("bad_model").string()});
  std::cerr.rdbuf(saved);

  EXPECT_EQ(rc, 1);
  EXPECT_NE(captured.str().find("nosuchmodel"), std::string::npos)
      << captured.str();
}

TEST(Init, RepeatCountExpandsFunctions) {
  // "2*lj,exp_decay,sqrt" is eam ntypes=... — use pair with ntypes=2 (paircol
  // 3) so a single region takes all three: the multiplier must expand.
  auto m = scaffold_and_load("pair_repeat", {"--model", "pair", "--ntypes", "2",
                                             "--functions", "2*lj,morse"});
  ASSERT_TRUE(m.target<PairForceCalculator>() != nullptr);
  EXPECT_EQ(m.target<PairForceCalculator>()->pair.size(), 3u);
}

// ── bond-order: built in memory, written via write_model ────────────────────

TEST(Init, TersoffAndStiwebLoad) {
  for (int nt : {1, 2}) {
    ASSERT_TRUE(((scaffold_and_load(
                      "ters" + std::to_string(nt),
                      {"--model", "tersoff", "--ntypes", std::to_string(nt)}))
                     .target<TersoffForceCalculator>() != nullptr));
    ASSERT_TRUE(((scaffold_and_load(
                      "sw" + std::to_string(nt),
                      {"--model", "stiweb", "--ntypes", std::to_string(nt)}))
                     .target<StiwebForceCalculator>() != nullptr));
  }
}

// ── ML: descriptor + one zeroed linear head per type ────────────────────────

TEST(Init, SoapHeadsSizedToDescriptor) {
  for (int nt : {1, 2}) {
    auto m =
        scaffold_and_load("soap" + std::to_string(nt),
                          {"--model", "soap", "--ntypes", std::to_string(nt),
                           "--n-max", "4", "--l-max", "3"});
    ASSERT_TRUE(((m).target<SoapModel>() != nullptr));
    const auto &s = (*(m).target<SoapModel>());
    EXPECT_EQ(s.heads.size(), static_cast<std::size_t>(nt));
  }
}

TEST(Init, AcsfRequiresChannelsAndLoads) {
  auto m =
      scaffold_and_load("acsf", {"--model", "acsf", "--g2-eta", "0.5", "1.2"});
  ASSERT_TRUE(((m).target<ACSF>() != nullptr));
  EXPECT_EQ((*(m).target<ACSF>()).radial.size(), 2u);
}

TEST(Init, LmbtrLoads) {
  ASSERT_TRUE(
      ((scaffold_and_load("lmbtr", {"--model", "lmbtr"})).target<LMBTR>() !=
       nullptr));
}

// ── the "same by definition" guarantee ──────────────────────────────────────
// Every macro-table parameter name must resolve, in order, to the reader
// registry's slot for that function — i.e. the shared macro really is the one
// source both consume.

TEST(Init, MacroNamesMatchRegistryOrder) {
  for (std::string_view fn : analytic_default_functions()) {
    auto defs = analytic_defaults(fn);
    ASSERT_FALSE(defs.empty()) << fn;
    for (std::size_t i = 0; i < defs.size(); ++i) {
      auto idx = io::analytic_param_index(fn, defs[i].name);
      ASSERT_TRUE(idx.has_value())
          << fn << " param '" << defs[i].name << "' not in registry";
      EXPECT_EQ(*idx, i) << fn << " param '" << defs[i].name
                         << "' registry order mismatch";
    }
  }
}
