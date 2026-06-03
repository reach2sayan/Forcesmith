// Descriptor-cache parity tests for MLBase.
//
// The fit-time cache (MLBase::prepare + the indexed eval_forces / eval_cached)
// must reproduce the legacy per-call eval_forces(cfg) exactly for an
// analytic-gradient model, and to finite-difference tolerance for a model that
// relies on the one-time FD gradient fill. These guard that the cache is a pure
// speedup, not a behaviour change, before the optimizer ever depends on it.

#include "potfit/potentials/soap.hpp"
#include "potfit/potentials/symmetry_functions.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <vector>

using namespace potfit;

namespace {

SymmetryFunctionModel make_sf_model() {
  SymmetryFunctionModel m;
  m.ntypes = 1;
  m.rcut = 6.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{MLPHead::make({3, 5, 1}, MLPHead::Act::Tanh, 2)});
  return m;
}

// A couple of differently sized clusters so the flat-atom precompute and the
// per-config indexing are exercised over more than one geometry.
std::vector<Configuration> make_configs() {
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
  {
    Configuration c;
    c.bc = PeriodicBC(100.0 * Mat3::Identity());
    Atom a0, a1, a2;
    a0.type = 0; a0.pos = {0.0, 0.0, 0.0};
    a1.type = 0; a1.pos = {1.9, -0.4, 0.6};
    a2.type = 0; a2.pos = {-0.7, 1.6, -1.1};
    c.atoms = {a0, a1, a2};
    cfgs.push_back(c);
  }
  return cfgs;
}

} // namespace

// Analytic-gradient model: cached forces/energy must match the legacy path to
// machine precision (same gradients, same assembly — just memoized).
TEST(MLCache, CachedMatchesLegacyAnalytic) {
  auto m = make_sf_model();
  auto cfgs = make_configs();
  m.prepare(std::span<Configuration>(cfgs.data(), cfgs.size()));
  ASSERT_TRUE(m.has_cache());

  for (std::size_t c = 0; c < cfgs.size(); ++c) {
    Configuration ref = cfgs[c];
    m.eval_forces(ref); // uncached reference

    Configuration cached = cfgs[c];
    m.eval_forces(cached, c); // cache-backed

    EXPECT_NEAR(cached.calc_energy, ref.calc_energy, 1e-10) << "config " << c;
    for (std::size_t a = 0; a < ref.atoms.size(); ++a) {
      EXPECT_NEAR(
          (cached.atoms[a].calc_force - ref.atoms[a].calc_force).norm(), 0.0,
          1e-9)
          << "config " << c << " atom " << a;
    }
  }
}

// grad_self = −Σ grad_neigh must hold in the cache (translation invariance), so
// the net force of an isolated cluster is zero on the cached path too.
TEST(MLCache, CachedNewtonThirdLaw) {
  auto m = make_sf_model();
  auto cfgs = make_configs();
  m.prepare(std::span<Configuration>(cfgs.data(), cfgs.size()));

  for (std::size_t c = 0; c < cfgs.size(); ++c) {
    Configuration cached = cfgs[c];
    m.eval_forces(cached, c);
    Vec3 net = Vec3::Zero();
    for (const auto &a : cached.atoms) {
      net += a.calc_force;
    }
    EXPECT_NEAR(net.norm(), 0.0, 1e-9) << "config " << c;
  }
}

// SOAP currently fills the cache by one-time finite difference of the
// descriptor. Energy is exact (same descriptor values); forces match the
// FD-over-energy legacy path to FD tolerance.
TEST(MLCache, SoapCachedMatchesLegacyFD) {
  SoapModel m;
  m.ntypes = 1;
  m.n_max = 3;
  m.l_max = 3;
  m.rcut = 4.0;
  m.sigma = 0.5;
  m.init_radial_basis();
  m.heads.reserve(1);
  m.heads.emplace_back(
      EnergyHead{MLPHead::make({static_cast<int>(m.descriptor_size()), 5, 1},
                               MLPHead::Act::Tanh, 4)});

  auto cfgs = make_configs();
  m.prepare(std::span<Configuration>(cfgs.data(), cfgs.size()));
  ASSERT_TRUE(m.has_cache());

  for (std::size_t c = 0; c < cfgs.size(); ++c) {
    Configuration ref = cfgs[c];
    m.eval_forces(ref); // FD-over-energy legacy path

    Configuration cached = cfgs[c];
    m.eval_forces(cached, c); // cache (FD-once gradients)

    EXPECT_NEAR(cached.calc_energy, ref.calc_energy, 1e-10) << "config " << c;
    for (std::size_t a = 0; a < ref.atoms.size(); ++a) {
      EXPECT_NEAR(
          (cached.atoms[a].calc_force - ref.atoms[a].calc_force).norm(), 0.0,
          1e-3)
          << "config " << c << " atom " << a;
    }
  }
}
