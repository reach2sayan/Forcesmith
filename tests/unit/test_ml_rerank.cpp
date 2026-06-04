// Species re-rank for ML force models: when an element is added/removed the
// compact slots shift, so a seeded ML model is remapped — retained element heads
// move to their new slots and their descriptor-blocked coefficients are
// reindexed, while a newly-added element gets a fresh zero head (to be re-fit).
// See MLBase::remap and the per-descriptor descriptor_index_map hooks.

#include "potfit/api/potfit.hpp"
#include "potfit/core/species.hpp"
#include "potfit/force/ml_force.hpp"
#include "potfit/io/config_reader.hpp" // io::ParseError
#include "potfit/potentials/acsf.hpp"
#include "potfit/potentials/lmbtr.hpp"
#include "potfit/potentials/soap.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace leaf = boost::leaf;
using namespace potfit;

namespace {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

template <class F> std::string run(F &&body) {
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

SpeciesRegistry reg_of(std::vector<std::string_view> syms) {
  return build_species_registry(syms).value();
}

// ACSF with 1×G1, 1×G2, 1×G4 (no G3/G5): descriptor_size(S) = 2S + P, with the
// flat layout [G1·S][G2·S][G4·P]. Heads are filled by the caller.
ACSF base_acsf(std::size_t S) {
  ACSF m;
  m.rcut = 6.0;
  m.g1 = 1;
  m.radial = {{1.0, 0.0}};       // 1 G2
  m.g4 = {{1.0, 1.0, 1.0}};      // 1 G4
  m.use_analytic_grads = true;
  m.ntypes = S;
  return m;
}

// A LinearHead whose coeff k carries `base + k` (so each slot is traceable) and
// a distinct bias.
EnergyHead traced_head(Eigen::Index n, double base, double bias) {
  LinearHead h;
  h.coeffs.resize(static_cast<std::size_t>(n));
  for (Eigen::Index k = 0; k < n; ++k) {
    h.coeffs[static_cast<std::size_t>(k)] =
        Param{base + static_cast<double>(k), false};
  }
  h.bias = Param{bias, false};
  return EnergyHead(h);
}

} // namespace

// ── ACSF: add an element (Z-order shift) ─────────────────────────────────────
// Seed {Ni(28), Cu(29)} (slots Ni=0, Cu=1), add Co(27) ⇒ {Co=0, Ni=1, Cu=2}:
// everything shifts up. Verify the retained heads land at the right new slots
// with their descriptor blocks reindexed, all Co-touching blocks zeroed, and a
// fresh zero head for Co.
TEST(MlRerank, AcsfAddElementShift) {
  ACSF m = base_acsf(2);
  const auto D_old = static_cast<Eigen::Index>(m.descriptor_size()); // 2*2+3=7
  ASSERT_EQ(D_old, 7);
  m.heads.emplace_back(traced_head(D_old, 1000.0, 7.0)); // Ni @ slot 0
  m.heads.emplace_back(traced_head(D_old, 2000.0, 9.0)); // Cu @ slot 1

  const auto old_reg = reg_of({"Ni", "Cu"});
  const auto new_reg = reg_of({"Co", "Ni", "Cu"});

  auto res = m.remap(old_reg, new_reg);
  ASSERT_TRUE(res) << "remap failed";
  const ACSF out = res.value();

  EXPECT_EQ(out.ntypes, 3u);
  EXPECT_EQ(out.heads.size(), 3u);
  EXPECT_EQ(out.descriptor_size(), 12u); // 2*3+6

  // New layout (S=3): [G1: 0,1,2][G2: 3,4,5][G4: 6..11], G4 pair ordinals via
  // pair_ordinal(a,b,3): (0,0)=0 (0,1)=1 (0,2)=2 (1,1)=3 (1,2)=4 (2,2)=5.
  const Eigen::VectorXd ni = out.heads[1].all_values(); // 12 coeffs + bias
  ASSERT_EQ(ni.size(), 13);
  EXPECT_DOUBLE_EQ(ni[12], 7.0);     // bias preserved
  EXPECT_DOUBLE_EQ(ni[1], 1000.0);   // G1 s=1 ← old G1 s=0 (old idx 0)
  EXPECT_DOUBLE_EQ(ni[4], 1002.0);   // G2 s=1 ← old G2 s=0 (old idx 2)
  EXPECT_DOUBLE_EQ(ni[9], 1004.0);   // G4 (Ni,Ni)=po3 ← old (0,0)=po0 (old idx 4)
  EXPECT_DOUBLE_EQ(ni[0], 0.0);      // G1 s=0 (Co) — new block
  EXPECT_DOUBLE_EQ(ni[7], 0.0);      // G4 (Co,Ni)=po1 — new block

  const Eigen::VectorXd cu = out.heads[2].all_values();
  EXPECT_DOUBLE_EQ(cu[12], 9.0);     // bias preserved
  EXPECT_DOUBLE_EQ(cu[2], 2001.0);   // G1 s=2 ← old G1 s=1 (old idx 1)
  EXPECT_DOUBLE_EQ(cu[5], 2003.0);   // G2 s=2 ← old G2 s=1 (old idx 3)
  EXPECT_DOUBLE_EQ(cu[11], 2006.0);  // G4 (Cu,Cu)=po5 ← old (1,1)=po2 (old idx 6)
  EXPECT_DOUBLE_EQ(cu[10], 2005.0);  // G4 (Ni,Cu)=po4 ← old (0,1)=po1 (old idx 5)

  const Eigen::VectorXd co = out.heads[0].all_values(); // brand-new element
  EXPECT_TRUE(co.isZero()) << co.transpose();
}

// ── ACSF: remove an element (subset) ─────────────────────────────────────────
// Seed {Co,Ni,Cu}, drop Co ⇒ {Ni,Cu}; retained heads down-reindex, Co blocks
// vanish.
TEST(MlRerank, AcsfRemoveElement) {
  ACSF m = base_acsf(3);
  const auto D_old = static_cast<Eigen::Index>(m.descriptor_size()); // 12
  m.heads.emplace_back(traced_head(D_old, 3000.0, 1.0)); // Co @ slot 0
  m.heads.emplace_back(traced_head(D_old, 1000.0, 7.0)); // Ni @ slot 1
  m.heads.emplace_back(traced_head(D_old, 2000.0, 9.0)); // Cu @ slot 2

  auto res = m.remap(reg_of({"Co", "Ni", "Cu"}), reg_of({"Ni", "Cu"}));
  ASSERT_TRUE(res);
  const ACSF out = res.value();

  EXPECT_EQ(out.ntypes, 2u);
  EXPECT_EQ(out.heads.size(), 2u);
  EXPECT_EQ(out.descriptor_size(), 7u);

  // New {Ni=0, Cu=1}. Ni head was at old slot 1. old layout (S=3):
  // G1 s=1 → idx1, G2 s=1 → idx4, G4 (Ni,Ni)=po3 → base6+3=9.
  // New layout (S=2): G1 s=0 → 0, G2 s=0 → 2, G4 (0,0)=po0 → base4+0=4.
  const Eigen::VectorXd ni = out.heads[0].all_values();
  ASSERT_EQ(ni.size(), 8);
  EXPECT_DOUBLE_EQ(ni[7], 7.0);    // bias
  EXPECT_DOUBLE_EQ(ni[0], 1001.0); // G1 s=0 ← old G1 s=1 (idx1)
  EXPECT_DOUBLE_EQ(ni[2], 1004.0); // G2 s=0 ← old G2 s=1 (idx4)
  EXPECT_DOUBLE_EQ(ni[4], 1009.0); // G4 (Ni,Ni) ← old (Ni,Ni)=po3 (idx9)
}

// ── ACSF: identity (no-op map) ───────────────────────────────────────────────
TEST(MlRerank, AcsfIdentity) {
  ACSF m = base_acsf(2);
  const auto D = static_cast<Eigen::Index>(m.descriptor_size());
  m.heads.emplace_back(traced_head(D, 1000.0, 7.0));
  m.heads.emplace_back(traced_head(D, 2000.0, 9.0));

  const auto reg = reg_of({"Ni", "Cu"});
  auto res = m.remap(reg, reg);
  ASSERT_TRUE(res);
  const ACSF out = res.value();

  ASSERT_EQ(out.heads.size(), 2u);
  for (std::size_t t = 0; t < 2; ++t) {
    const Eigen::VectorXd a = out.heads[t].all_values();
    const Eigen::VectorXd b = m.heads[t].all_values();
    EXPECT_TRUE((a - b).isZero()) << "slot " << t;
  }
}

// ── LMBTR: add an element ────────────────────────────────────────────────────
// k2 (per-species, 4 bins) + k3 (per-pair, 3 bins): descriptor_size(S)=4S+3P.
TEST(MlRerank, LmbtrAddElement) {
  LMBTR m;
  m.k2 = LMBTR::Grid{0.0, 6.0, 4, 0.3};
  m.k3 = LMBTR::Grid{-1.0, 1.0, 3, 0.1};
  m.rcut = 6.0;
  m.normalize_l2 = false;
  m.ntypes = 2;
  const auto D_old = static_cast<Eigen::Index>(m.descriptor_size()); // 8+9=17
  ASSERT_EQ(D_old, 17);
  m.heads.emplace_back(traced_head(D_old, 1000.0, 7.0)); // Ni @ slot 0
  m.heads.emplace_back(traced_head(D_old, 2000.0, 9.0)); // Cu @ slot 1

  auto res = m.remap(reg_of({"Ni", "Cu"}), reg_of({"Co", "Ni", "Cu"}));
  ASSERT_TRUE(res);
  const LMBTR out = res.value();

  EXPECT_EQ(out.ntypes, 3u);
  EXPECT_EQ(out.descriptor_size(), 30u); // 12 + 18

  // New (S=3): k2 base0 (count4, nchan3), k3 base12 (count3, nchan6).
  // Ni new slot 1: k2 s=1 → [4,5,6,7] ← old s=0 → [0,1,2,3].
  // k3 (Ni,Ni)=po3 → 12+3*3=[21,22,23] ← old (0,0)=po0 → base8 → [8,9,10].
  const Eigen::VectorXd ni = out.heads[1].all_values();
  ASSERT_EQ(ni.size(), 31);
  EXPECT_DOUBLE_EQ(ni[30], 7.0);    // bias
  EXPECT_DOUBLE_EQ(ni[4], 1000.0);  // k2 s=1 bin0 ← old k2 s=0 bin0 (idx0)
  EXPECT_DOUBLE_EQ(ni[7], 1003.0);  // k2 s=1 bin3 ← old idx3
  EXPECT_DOUBLE_EQ(ni[21], 1008.0); // k3 (Ni,Ni) bin0 ← old (0,0) bin0 (idx8)
  EXPECT_DOUBLE_EQ(ni[0], 0.0);     // k2 s=0 (Co) — new block

  const Eigen::VectorXd co = out.heads[0].all_values();
  EXPECT_TRUE(co.isZero());
}

// ── SOAP: add an element (variable-length pair blocks) ───────────────────────
// SOAP's power spectrum has variable-length pair blocks (n≤n' on the diagonal),
// so we verify the re-rank is a bijection on retained entries: with all old
// pairs retained (adding Co to {Ni,Cu}), every old coeff must reappear exactly
// once in the remapped head and all Co-touching entries must be zero.
TEST(MlRerank, SoapAddElement) {
  SoapModel m;
  m.n_max = 2;
  m.l_max = 1;
  m.rcut = 6.0;
  m.sigma = 0.5;
  m.ntypes = 2;
  const auto D_old = static_cast<Eigen::Index>(m.descriptor_size());
  ASSERT_EQ(D_old, 20); // 2*( S*nm(nm+1)/2 + (S(S-1)/2)nm² ) = 2*(6+4)
  m.heads.emplace_back(traced_head(D_old, 1.0, 0.0)); // Ni: coeff k = k+1
  m.heads.emplace_back(traced_head(D_old, -100.0, 0.0)); // Cu

  auto res = m.remap(reg_of({"Ni", "Cu"}), reg_of({"Co", "Ni", "Cu"}));
  ASSERT_TRUE(res);
  const SoapModel out = res.value();

  EXPECT_EQ(out.ntypes, 3u);
  EXPECT_EQ(out.descriptor_size(), 42u); // 2*(9+12)

  // New Ni head (slot 1): exactly the 20 old values {1..20} survive, rest zero.
  const Eigen::VectorXd ni = out.heads[1].all_values(); // 42 + bias
  ASSERT_EQ(ni.size(), 43);
  std::vector<double> nz;
  for (Eigen::Index k = 0; k < 42; ++k) {
    if (ni[k] != 0.0) {
      nz.push_back(ni[k]);
    }
  }
  ASSERT_EQ(nz.size(), 20u) << "expected all old entries to survive (bijection)";
  std::ranges::sort(nz);
  for (std::size_t i = 0; i < nz.size(); ++i) {
    EXPECT_DOUBLE_EQ(nz[i], static_cast<double>(i + 1));
  }

  const Eigen::VectorXd co = out.heads[0].all_values();
  EXPECT_TRUE(co.isZero());
}

// ── End-to-end: seed an ML model, add an element, evaluate triggers the re-rank
// through PotFit::ensure_frozen, and the model stays evaluable. ───────────────
TEST(MlRerank, EndToEndSeedThenAddElement) {
  PotFit s;
  double e_before = 0.0;
  double e_after = 0.0;
  std::size_t ntypes_after = 0;

  const std::string err = run([&]() -> leaf::result<void> {
    BOOST_LEAF_CHECK(s.declare_element("Ni"));
    BOOST_LEAF_CHECK(s.declare_element("Cu"));
    const std::size_t c = s.add_configuration(PeriodicBC(20.0 * Mat3::Identity()));
    BOOST_LEAF_CHECK(s.add_atom(c, "Ni", Vec3(0, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Cu", Vec3(2.3, 0, 0)));
    BOOST_LEAF_CHECK(s.add_atom(c, "Ni", Vec3(0, 2.5, 0)));

    ACSF m = base_acsf(2);
    const auto D = static_cast<Eigen::Index>(m.descriptor_size());
    m.heads.emplace_back(traced_head(D, 0.1, 0.0)); // Ni
    m.heads.emplace_back(traced_head(D, 0.2, 0.0)); // Cu
    BOOST_LEAF_CHECK(s.seed_force_model(ForceCalculator{std::move(m)}));

    BOOST_LEAF_AUTO(r0, s.evaluate(c)); // ntypes 2 → seeded model reused
    e_before = r0.energy;

    // Add a Co atom: ntypes 2 → 3, forcing the ML re-rank on next freeze.
    BOOST_LEAF_CHECK(s.add_atom(c, "Co", Vec3(0, 0, 2.4)));
    BOOST_LEAF_AUTO(r1, s.evaluate(c));
    e_after = r1.energy;

    BOOST_LEAF_AUTO(mp, s.model());
    EXPECT_TRUE(std::holds_alternative<ACSF>(*mp));
    ntypes_after = std::visit([](const auto &cc) { return cc.ntypes; }, *mp);
    return {};
  });

  ASSERT_EQ(err, "") << err;
  EXPECT_EQ(ntypes_after, 3u);
  EXPECT_TRUE(std::isfinite(e_before));
  EXPECT_TRUE(std::isfinite(e_after));
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
