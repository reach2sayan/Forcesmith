#include "forcesmith/api/forcesmith.hpp"
#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/io/config_reader.hpp" // io::ParseError
#include "forcesmith/potentials/analytic_potential.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <variant>
#include <vector>

namespace leaf = boost::leaf;
using namespace forcesmith;

// BOOST_LEAF_CHECK expands to a GNU statement-expression ({ ... }); silence the
// pedantic complaint about that Boost idiom for this translation unit.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

// ── Helpers ──────────────────────────────────────────────────────────────────

// Run a body returning leaf::result<void>; returns the error message ("" on ok).
template <class F> static std::string run(F &&body) {
  std::string msg;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        BOOST_LEAF_CHECK(body());
        return {};
      },
      [&](const io::ParseError &e) { msg = e.message; },
      [&]() { msg = "unknown error"; });
  return msg;
}

static Mat3 cubic(double a) {
  Mat3 m = Mat3::Zero();
  m(0, 0) = a;
  m(1, 1) = a;
  m(2, 2) = a;
  return m;
}

// A Morse pair Potential with fixed reference parameters.
static Potential morse_cu() {
  return Potential(Morse(0.5, 1.5, 2.5, 0.1, 6.0)); // De, a, re, rmin, rmax
}

// ── Programmatic build → evaluate, and parity with the loader-style path ──────

TEST(Forcesmith, ProgrammaticEvaluate) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(2.5, 0, 0)));
    BOOST_LEAF_CHECK(s.set_ref_energy(c, -1.0));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_AUTO(r, s.evaluate(c));
    EXPECT_EQ(r.forces.size(), 2u);
    EXPECT_TRUE(std::isfinite(r.energy));
    // Newton's third law: the two-atom forces are equal and opposite.
    EXPECT_NEAR((r.forces[0] + r.forces[1]).norm(), 0.0, 1e-9);
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// Building the same fit two ways — programmatic set_pair_potential vs seeding a
// parsed force model — must give identical energy/forces.
TEST(Forcesmith, ProgrammaticMatchesSeeded) {
  auto build_cfg = [](Forcesmith &s) -> leaf::result<std::size_t> {
    const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(2.5, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 2.7, 0)));
    return c;
  };

  double e_prog = 0.0;
  double e_seed = 0.0;
  Vec3 f0_prog = Vec3::Zero();
  Vec3 f0_seed = Vec3::Zero();

  std::string err = run([&]() -> leaf::result<void> {
    Forcesmith prog;
    BOOST_LEAF_AUTO(cp, build_cfg(prog));
    BOOST_LEAF_CHECK(prog.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_AUTO(rp, prog.evaluate(cp));
    e_prog = rp.energy;
    f0_prog = rp.forces[0];

    Forcesmith seed;
    BOOST_LEAF_AUTO(cs, build_cfg(seed));
    BOOST_LEAF_AUTO(pot, Potential::from_text(
                             R"({"type":"morse","rmin":0.1,"rmax":6.0,)"
                             R"("De":0.5,"a":1.5,"re":2.5})"));
    BOOST_LEAF_CHECK(seed.set_pair_potential("Cu", "Cu", std::move(pot)));
    BOOST_LEAF_AUTO(rs, seed.evaluate(cs));
    e_seed = rs.energy;
    f0_seed = rs.forces[0];
    return {};
  });

  ASSERT_EQ(err, "") << err;
  EXPECT_NEAR(e_prog, e_seed, 1e-12);
  EXPECT_NEAR((f0_prog - f0_seed).norm(), 0.0, 1e-12);
}

// ── Auto-grow: adding a new element re-ranks slots (Z-sorted) at the next build

TEST(Forcesmith, AutoGrowRerank) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(2.5, 0, 0)));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_CHECK(s.evaluate(c)); // freezes with ntypes == 1

    // Introduce a second element; supply the new pair potentials.
    BOOST_LEAF_CHECK(s.add_atom(c, "Ni", Vec3(0, 2.6, 0)));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Ni", morse_cu()));
    BOOST_LEAF_CHECK(s.set_pair_potential("Ni", "Ni", morse_cu()));
    BOOST_LEAF_CHECK(s.evaluate(c)); // auto-rebuilds with ntypes == 2

    BOOST_LEAF_AUTO(reg, s.species());
    EXPECT_EQ(ntypes(*reg), 2u);
    // Ni (Z=28) ranks before Cu (Z=29): slot 0 == Ni, slot 1 == Cu.
    EXPECT_EQ(species_at(*reg, 0).symbol, std::string_view{"Ni"});
    EXPECT_EQ(species_at(*reg, 1).symbol, std::string_view{"Cu"});
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// ── A missing required pair potential surfaces as a leaf error at build time ──

TEST(Forcesmith, MissingPairPotentialErrors) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Ni", Vec3(2.5, 0, 0)));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));
    // Cu-Ni and Ni-Ni intentionally omitted.
    BOOST_LEAF_CHECK(s.evaluate(c));
    return {};
  });
  EXPECT_NE(err, ""); // expected to fail with a "missing pair potential" error
}

// ── An unknown element symbol is rejected immediately ─────────────────────────

TEST(Forcesmith, UnknownElementRejected) {
  Forcesmith s;
  const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
  std::string err = run([&]() -> leaf::result<void> {
    BOOST_LEAF_CHECK(s.add_atom(c, "Xx", Vec3(0, 0, 0)));
    return {};
  });
  EXPECT_NE(err, "");
}

// ── Richer model families: programmatic build, parity with seeded, decompose ──

// Standard Tersoff (1988) Si parameters.
static TersoffParams si_tersoff() {
  TersoffParams p;
  p.A = 1830.8;
  p.B = 471.18;
  p.lambda = 2.4799;
  p.mu = 1.7322;
  p.beta = 1.1e-6;
  p.n = 0.78734;
  p.c = 1.0039e5;
  p.d = 16.217;
  p.h = -0.59825;
  p.R = 2.7;
  p.S = 3.0;
  return p;
}

// Three Si atoms in an equilateral triangle (side within the Tersoff cutoff).
static leaf::result<std::size_t> si_triangle(Forcesmith &s, double r = 2.35) {
  BOOST_LEAF_CHECK(s.declare_element("Si"));
  const std::size_t c = s.add_configuration(PeriodicBC(cubic(12.0)));
  BOOST_LEAF_CHECK(s.add_atom(c, "Si", Vec3(0, 0, 0)));
  BOOST_LEAF_CHECK(s.add_atom(c, "Si", Vec3(r, 0, 0)));
  BOOST_LEAF_CHECK(s.add_atom(c, "Si", Vec3(r * 0.5, r * 0.8660254, 0)));
  return c;
}

// Building a Tersoff model programmatically (set_tersoff_params) must match
// seeding a directly-constructed TersoffForceCalculator through the same
// pipeline — this validates the analytic-param materialize path and that the
// strategy registry selects the Tersoff family.
TEST(Forcesmith, TersoffProgrammaticMatchesSeeded) {
  double e_prog = 0.0;
  double e_seed = 0.0;

  std::string err = run([&]() -> leaf::result<void> {
    Forcesmith prog;
    BOOST_LEAF_AUTO(cp, si_triangle(prog));
    BOOST_LEAF_CHECK(prog.set_tersoff_params("Si", "Si", si_tersoff()));
    BOOST_LEAF_AUTO(rp, prog.evaluate(cp));
    e_prog = rp.energy;
    BOOST_LEAF_AUTO(m, prog.model());
    EXPECT_TRUE(std::holds_alternative<TersoffForceCalculator>(*m));

    Forcesmith seed;
    BOOST_LEAF_AUTO(cs, si_triangle(seed));
    TersoffForceCalculator calc;
    calc.ntypes = 1;
    calc.params.reserve(1);
    calc.params.emplace_back(si_tersoff());
    BOOST_LEAF_CHECK(seed.seed_force_model(ForceCalculator{std::move(calc)}));
    BOOST_LEAF_AUTO(rs, seed.evaluate(cs));
    e_seed = rs.energy;
    return {};
  });

  ASSERT_EQ(err, "") << err;
  EXPECT_NEAR(e_prog, e_seed, 1e-9);
}

// Stiweb exercises both the analytic params and the per-triplet λ vector.
TEST(Forcesmith, StiwebProgrammaticMatchesSeeded) {
  auto sw_params = [] {
    SWParams p;
    p.A = 7.05;
    p.B = 0.602;
    p.p = 4.0;
    p.q = 0.0;
    p.delta = 1.0;
    p.a1 = 3.0;
    p.gamma = 1.2;
    p.a2 = 3.0;
    return p;
  };
  const Param lambda{21.0};

  double e_prog = 0.0;
  double e_seed = 0.0;

  std::string err = run([&]() -> leaf::result<void> {
    Forcesmith prog;
    BOOST_LEAF_AUTO(cp, si_triangle(prog, 2.5));
    BOOST_LEAF_CHECK(prog.set_stiweb_params("Si", "Si", sw_params()));
    BOOST_LEAF_CHECK(prog.set_stiweb_lambda("Si", "Si", "Si", lambda));
    BOOST_LEAF_AUTO(rp, prog.evaluate(cp));
    e_prog = rp.energy;
    BOOST_LEAF_AUTO(m, prog.model());
    EXPECT_TRUE(std::holds_alternative<StiwebForceCalculator>(*m));

    Forcesmith seed;
    BOOST_LEAF_AUTO(cs, si_triangle(seed, 2.5));
    StiwebForceCalculator calc;
    calc.ntypes = 1;
    calc.params.reserve(1);
    calc.params.emplace_back(sw_params());
    calc.lambda.push_back(lambda); // ntypes*paircol == 1 entry
    BOOST_LEAF_CHECK(seed.seed_force_model(ForceCalculator{std::move(calc)}));
    BOOST_LEAF_AUTO(rs, seed.evaluate(cs));
    e_seed = rs.energy;
    return {};
  });

  ASSERT_EQ(err, "") << err;
  EXPECT_NEAR(e_prog, e_seed, 1e-9);
}

// Decompose round-trip: seed a single-element ADP model, then make an edit that
// detaches the seed (which decomposes the built model back into the symbol-keyed
// spec) and re-materializes. Re-setting one table to its same value must leave
// the energy unchanged — proving every ADP table survived the round-trip.
TEST(Forcesmith, AdpSeededDecomposeRoundTrip) {
  auto phi = [] { return Potential(Morse(0.5, 1.5, 2.5, 0.1, 6.0)); };
  auto dens = [] { return Potential(ExpDecay(1.0, 1.0, 0.1, 6.0)); };
  auto emb = [] { return Potential(ConstFunc(-2.0, 0.0, 100.0)); };
  auto dip = [] { return Potential(ExpDecay(0.3, 0.8, 0.1, 6.0)); };
  auto quad = [] { return Potential(ExpDecay(0.2, 0.9, 0.1, 6.0)); };

  double e_seed = 0.0;
  double e_after = 0.0;

  std::string err = run([&]() -> leaf::result<void> {
    Forcesmith s;
    BOOST_LEAF_CHECK(s.declare_element("Cu"));
    const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(2.5, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(0, 2.7, 0)));

    ADPForceCalculator calc;
    calc.ntypes = 1;
    calc.pair.reserve(1);
    calc.pair.emplace_back(phi());
    calc.density.reserve(1);
    calc.density.emplace_back(dens());
    calc.embedding.reserve(1);
    calc.embedding.emplace_back(emb());
    calc.dipole.reserve(1);
    calc.dipole.emplace_back(dip());
    calc.quadrupole.reserve(1);
    calc.quadrupole.emplace_back(quad());
    BOOST_LEAF_CHECK(s.seed_force_model(ForceCalculator{std::move(calc)}));

    BOOST_LEAF_AUTO(r0, s.evaluate(c));
    e_seed = r0.energy;

    // This edit detaches the seed → decompose into spec → re-materialize ADP.
    BOOST_LEAF_CHECK(s.set_dipole("Cu", "Cu", dip()));
    BOOST_LEAF_AUTO(r1, s.evaluate(c));
    e_after = r1.energy;

    BOOST_LEAF_AUTO(m, s.model());
    EXPECT_TRUE(std::holds_alternative<ADPForceCalculator>(*m));
    return {};
  });

  ASSERT_EQ(err, "") << err;
  EXPECT_NEAR(e_seed, e_after, 1e-12);
}

// A missing analytic-parameter block surfaces as a build-time leaf error.
TEST(Forcesmith, MissingTersoffParamErrors) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Si", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "C", Vec3(1.6, 0, 0)));
    // Diagonal blocks supplied (so both elements enter the registry); the
    // C-Si cross block is intentionally omitted.
    BOOST_LEAF_CHECK(s.set_tersoff_params("Si", "Si", si_tersoff()));
    BOOST_LEAF_CHECK(s.set_tersoff_params("C", "C", si_tersoff()));
    BOOST_LEAF_CHECK(s.evaluate(c));
    return {};
  });
  EXPECT_NE(err, ""); // expected: "missing tersoff parameters for C-Si"
}

// ── Configuration factory round-trip ─────────────────────────────────────────

TEST(Forcesmith, ConfigurationFromText) {
  std::string err = run([&]() -> leaf::result<void> {
    BOOST_LEAF_AUTO(cfg, Configuration::from_text(R"({
      "X": [5.0, 0.0, 0.0],
      "Y": [0.0, 5.0, 0.0],
      "Z": [0.0, 0.0, 5.0],
      "E": -2.25,
      "atoms": [
        {"element": "Cu", "position": [0.0, 0.0, 0.0], "force": [0.1, 0.2, 0.3]},
        {"element": "Cu", "position": [2.0, 0.0, 0.0]}
      ]
    })"));
    EXPECT_EQ(cfg.atoms.size(), 2u);
    EXPECT_DOUBLE_EQ(cfg.ref.energy, -2.25);
    EXPECT_DOUBLE_EQ(cfg.atoms[0].ref.force.x(), 0.1);
    EXPECT_EQ(cfg.atoms[0].type.symbol, std::string_view{"Cu"});
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// A single-Cu cubic cell at a given lattice constant, built via from_text.
static Configuration cu_cell(double a) {
  std::string box = std::to_string(a);
  auto r = Configuration::from_text(R"({
    "X": [)" + box + R"(, 0.0, 0.0],
    "Y": [0.0, )" + box + R"(, 0.0],
    "Z": [0.0, 0.0, )" + box + R"(],
    "E": -1.0,
    "atoms": [
      {"element": "Cu", "position": [0.0, 0.0, 0.0]},
      {"element": "Cu", "position": [2.0, 0.0, 0.0]}
    ]
  })");
  return r.value(); // tests build clean input; surface any parse bug loudly
}

TEST(Forcesmith, AddConfigurationsBulk) {
  // Batch-attach via a std::vector, then again via a std::array, and confirm the
  // returned first-index, config_count growth, and that the lazy re-freeze picks
  // the new configs up (atom slots stamped, neighbor lists built).
  std::string err = run([&]() -> leaf::result<void> {
    Forcesmith s;
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));

    // Pre-seed one config the single-arg way so the batch does not start at 0.
    const std::size_t c0 = s.add_configuration(PeriodicBC(cubic(8.0)));
    BOOST_LEAF_CHECK(s.add_atom(c0, "Cu", Vec3(0, 0, 0)));
    EXPECT_EQ(s.config_count(), 1u);

    // vector batch
    std::vector<Configuration> vec;
    vec.push_back(cu_cell(8.0));
    vec.push_back(cu_cell(8.5));
    const std::size_t first_vec = s.add_configurations(std::move(vec));
    EXPECT_EQ(first_vec, 1u);            // appended after c0
    EXPECT_EQ(s.config_count(), 3u);

    // array batch (exercises the generic-range acceptance, not just vector)
    std::array<Configuration, 2> arr{cu_cell(9.0), cu_cell(9.5)};
    const std::size_t first_arr = s.add_configurations(std::move(arr));
    EXPECT_EQ(first_arr, 3u);
    EXPECT_EQ(s.config_count(), 5u);

    // The re-freeze must see all five: evaluate one of the batched configs.
    BOOST_LEAF_AUTO(r, s.evaluate(first_arr));
    EXPECT_EQ(r.forces.size(), 2u);

    BOOST_LEAF_AUTO(cfgs, s.configurations());
    EXPECT_EQ(cfgs.size(), 5u);
    // Every atom has had its compact type slot stamped at freeze.
    for (const auto &cfg : cfgs) {
      for (const auto &atom : cfg.atoms) {
        EXPECT_EQ(atom.type.symbol, std::string_view{"Cu"});
      }
    }
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

TEST(Forcesmith, AddConfigurationsEquivalentToLoop) {
  // One bulk call must yield the same config_count and per-config atom counts as
  // N single add_configuration calls.
  auto build = [](bool bulk) -> std::size_t {
    Forcesmith s;
    std::vector<Configuration> batch;
    batch.push_back(cu_cell(8.0));
    batch.push_back(cu_cell(8.5));
    batch.push_back(cu_cell(9.0));
    if (bulk) {
      s.add_configurations(std::move(batch));
    } else {
      for (auto &c : batch) {
        s.add_configuration(std::move(c));
      }
    }
    return s.config_count();
  };
  EXPECT_EQ(build(true), build(false));
  EXPECT_EQ(build(true), 3u);
}

// A single-Cu cubic cell carrying an explicit name, built via from_text.
static Configuration named_cell(std::string_view name) {
  auto r = Configuration::from_text(R"({
    "name": ")" + std::string(name) + R"(",
    "X": [8.0, 0.0, 0.0],
    "Y": [0.0, 8.0, 0.0],
    "Z": [0.0, 0.0, 8.0],
    "E": -1.0,
    "atoms": [
      {"element": "Cu", "position": [0.0, 0.0, 0.0]},
      {"element": "Cu", "position": [2.0, 0.0, 0.0]}
    ]
  })");
  return r.value();
}

// index_of(name) resolves a user-supplied name to the index add_configuration
// returned — with no explicit freeze() and without dirtying the session.
TEST(Forcesmith, IndexOfByName) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    const std::size_t a = s.add_configuration(named_cell("alpha"));
    const std::size_t b = s.add_configuration(named_cell("beta"));
    BOOST_LEAF_AUTO(ia, s.get_configuration_index("alpha"));
    BOOST_LEAF_AUTO(ib, s.get_configuration_index("beta"));
    EXPECT_EQ(ia, a);
    EXPECT_EQ(ib, b);
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// Unnamed configs get auto-names ("config-<i>") materialized on demand by
// index_of — proving the lookup works without a full freeze().
TEST(Forcesmith, IndexOfAutoName) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    s.add_configuration(cu_cell(8.0));
    s.add_configuration(cu_cell(8.5));
    BOOST_LEAF_AUTO(i1, s.get_configuration_index("config-1"));
    EXPECT_EQ(i1, 1u);
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// index_of(Configuration&) round-trips a handle from configurations() back to
// its dense index in O(1).
TEST(Forcesmith, IndexOfByRefRoundTrips) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    s.add_configuration(cu_cell(8.0));
    s.add_configuration(cu_cell(8.5));
    s.add_configuration(cu_cell(9.0));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_AUTO(cfgs, s.configurations());
    for (std::size_t k = 0; k < cfgs.size(); ++k) {
      BOOST_LEAF_AUTO(idx, s.get_configuration_index(cfgs[k]));
      EXPECT_EQ(idx, k);
    }
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// Both overloads report errors for unknown name / foreign configuration.
TEST(Forcesmith, IndexOfErrors) {
  Forcesmith s;
  s.add_configuration(cu_cell(8.0));

  // Unknown name.
  std::string e1 = run([&]() -> leaf::result<void> {
    BOOST_LEAF_AUTO(i, s.get_configuration_index("nope"));
    (void)i;
    return {};
  });
  EXPECT_NE(e1, "");

  // A configuration not owned by this session.
  std::string e2 = run([&]() -> leaf::result<void> {
    Configuration foreign = cu_cell(8.0);
    BOOST_LEAF_AUTO(i, s.get_configuration_index(foreign));
    (void)i;
    return {};
  });
  EXPECT_NE(e2, "");
}

// Duplicate user-supplied names surface the dedup error early, via
// get_configuration_index.
TEST(Forcesmith, IndexOfDuplicateName) {
  Forcesmith s;
  s.add_configuration(named_cell("dup"));
  s.add_configuration(named_cell("dup"));
  std::string err = run([&]() -> leaf::result<void> {
    BOOST_LEAF_AUTO(i, s.get_configuration_index("dup"));
    (void)i;
    return {};
  });
  EXPECT_NE(err, "");
}

// get_atom_index(atom) returns the atom's position within its parent config;
// atom.parent is stamped at freeze, so every atom from configurations() resolves.
TEST(Forcesmith, GetAtomIndex) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    s.add_configuration(cu_cell(8.0));
    s.add_configuration(cu_cell(8.5));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_AUTO(cfgs, s.configurations());
    for (const auto &cfg : cfgs) {
      for (std::size_t a = 0; a < cfg.atoms.size(); ++a) {
        BOOST_LEAF_AUTO(ai, s.get_atom_index(cfg.atoms[a]));
        EXPECT_EQ(ai, a);
      }
    }
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// An atom that is not owned by the session errors — even after the implicit
// freeze (which only stamps the session's own atoms, not a foreign one).
TEST(Forcesmith, GetAtomIndexForeignAtom) {
  Forcesmith s;
  s.add_configuration(cu_cell(8.0));
  Configuration foreign = cu_cell(8.0); // not in the session → parent stays null
  std::string err = run([&]() -> leaf::result<void> {
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu())); // freeze ok
    BOOST_LEAF_AUTO(ai, s.get_atom_index(foreign.atoms[0]));
    (void)ai;
    return {};
  });
  EXPECT_NE(err, "");
}

// atom.parent points back at the owning config, and chains to its config index.
TEST(Forcesmith, AtomParentBackRef) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    s.add_configuration(cu_cell(8.0));
    s.add_configuration(cu_cell(8.5));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_AUTO(cfgs, s.configurations());
    for (std::size_t k = 0; k < cfgs.size(); ++k) {
      for (const auto &atom : cfgs[k].atoms) {
        EXPECT_EQ(atom.parent, &cfgs[k]);
        BOOST_LEAF_AUTO(ci, s.get_configuration_index(*atom.parent));
        EXPECT_EQ(ci, k);
      }
    }
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

// The name and atom-handle reference setters land the same values as the
// index-based forms.
TEST(Forcesmith, ReferenceSettersByNameAndHandle) {
  Forcesmith s;
  std::string err = run([&]() -> leaf::result<void> {
    s.add_configuration(named_cell("alpha"));
    BOOST_LEAF_CHECK(s.set_pair_potential("Cu", "Cu", morse_cu()));

    // by name (config-level) and by atom handle (per-atom force)
    BOOST_LEAF_CHECK(s.set_ref_energy("alpha", -3.5));
    BOOST_LEAF_CHECK(s.set_weight("alpha", 2.0));
    BOOST_LEAF_AUTO(cfgs, s.configurations());
    const Vec3 f(0.1, 0.2, 0.3);
    BOOST_LEAF_CHECK(s.set_ref_force(cfgs[0].atoms[1], f));

    EXPECT_EQ(cfgs[0].ref.energy, -3.5);
    EXPECT_EQ(cfgs[0].weight, 2.0);
    EXPECT_TRUE(cfgs[0].atoms[1].ref.force.isApprox(f));
    // The untouched atom keeps its default reference force.
    EXPECT_TRUE(cfgs[0].atoms[0].ref.force.isApprox(Vec3::Zero()));
    return {};
  });
  EXPECT_EQ(err, "") << err;
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
