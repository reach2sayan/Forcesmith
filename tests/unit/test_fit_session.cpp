#include "potfit/api/fit_session.hpp"
#include "potfit/io/config_reader.hpp" // io::ParseError
#include "potfit/potentials/analytic_potential.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

#include <string>

namespace leaf = boost::leaf;
using namespace potfit;

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

TEST(FitSession, ProgrammaticEvaluate) {
  FitSession s;
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
TEST(FitSession, ProgrammaticMatchesSeeded) {
  auto build_cfg = [](FitSession &s) -> leaf::result<std::size_t> {
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
    FitSession prog;
    BOOST_LEAF_AUTO(cp, build_cfg(prog));
    BOOST_LEAF_CHECK(prog.set_pair_potential("Cu", "Cu", morse_cu()));
    BOOST_LEAF_AUTO(rp, prog.evaluate(cp));
    e_prog = rp.energy;
    f0_prog = rp.forces[0];

    FitSession seed;
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

TEST(FitSession, AutoGrowRerank) {
  FitSession s;
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

TEST(FitSession, MissingPairPotentialErrors) {
  FitSession s;
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

TEST(FitSession, UnknownElementRejected) {
  FitSession s;
  const std::size_t c = s.add_configuration(PeriodicBC(cubic(8.0)));
  std::string err = run([&]() -> leaf::result<void> {
    BOOST_LEAF_CHECK(s.add_atom(c, "Xx", Vec3(0, 0, 0)));
    return {};
  });
  EXPECT_NE(err, "");
}

// ── Configuration factory round-trip ─────────────────────────────────────────

TEST(FitSession, ConfigurationFromText) {
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
