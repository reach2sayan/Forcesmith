#pragma once

#include "forcesmith/core/boundary_conditions.hpp"
#include "forcesmith/core/serialization.hpp"
#include "forcesmith/core/site_id.hpp"
#include "forcesmith/core/species.hpp"
#include "forcesmith/core/types.hpp"

#include <array>
#include <boost/leaf/result.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace forcesmith {

struct Atom;
struct Configuration;
class RadialPotential;

enum NeighborSiteRole : std::size_t {
  kSitePhi = 0, // pair        φ(r)
  kSiteGi,      // density     g_{t(i)}(r)
  kSiteGj,      // density     g_{t(j)}(r)
  kSiteDipole,  // ADP dipole  u(r)
  kSiteQuad,    // ADP quad    w(r)
  kNeighborSiteCount
};

struct NeighborEntry : Serializable<NeighborEntry> {
  const Atom *neighbor = nullptr; // non-owning; restored by build_neighbor_list
  const RadialPotential *pot = nullptr; // non-owning; resolved at build time
  Vec3 dist = Vec3::Zero();
  std::array<SiteId, kNeighborSiteCount> sites = {};
};

struct Atom : Serializable<Atom> {
  Species type{}; // element identity + compact slot (implicitly indexes tables)
  Vec3 pos = Vec3::Zero();

  struct Reference {
    Vec3 force = Vec3::Zero(); // target force from input data
  } ref;

  Vec3 calc_force = Vec3::Zero(); // written by ForceCalculator
  std::vector<NeighborEntry> neighbors;

  const Configuration *parent = nullptr;

  double rho = 0.0;                 // accumulated electron density
  double gradF = 0.0;               // dF/dρ (embedding energy gradient)
  Vec3 mu = Vec3::Zero();           // dipole distortion (ADP)
  SymTens lambda = SymTens::Zero(); // quadrupole distortion (ADP)

  void ZeroScratch() {
    rho = 0.0;
    gradF = 0.0;
    mu = Vec3::Zero();
    lambda = SymTens::Zero();
  }
  void ZeroForce() { calc_force = Vec3::Zero(); }
};

struct NeighborListKey {
  double rcut = -1.0;         // cutoff the list was built for
  const void *pots = nullptr; // potential-table identity (pointer)
  std::size_t sig = 0;        // geometry_signature(cfg)
  bool operator==(const NeighborListKey &) const = default;
};

struct Configuration : Serializable<Configuration> {
  std::vector<Atom> atoms;
  BoundaryConditions bc = PeriodicBC(Mat3::Identity());
  std::string name;

  struct Reference {
    double energy = 0.0;
    SymTens stress = SymTens::Zero();
  } ref;

  double weight = 1.0; // fitting weight (not a target — stays flat)
  double calc_energy = 0.0;
  SymTens calc_stress = SymTens::Zero();
  double calc_limit = 0.0; // F(ρ) out-of-range penalty (RESCALE-style)

  std::optional<NeighborListKey> nl_key;

  [[nodiscard]] static boost::leaf::result<Configuration>
  from_text(std::string_view json_record);
  [[nodiscard]] static boost::leaf::result<Configuration>
  from_file(const std::filesystem::path &path);
};

inline double force_rms(const Configuration &cfg) {
  const double sq = std::transform_reduce(
      cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{},
      [](const Atom &a) { return a.calc_force.squaredNorm(); });
  return cfg.atoms.empty()
             ? 0.0
             : std::sqrt(sq / static_cast<double>(cfg.atoms.size()));
}

template <> struct Serializer<NeighborEntry> {
  template <CArchive Archive>
  static void apply(Archive &ar, NeighborEntry &e, unsigned int) {
    ar & e.dist(0) & e.dist(1) & e.dist(2);
  }
};

template <> struct Serializer<Atom> {
  template <CArchive Archive>
  static void apply(Archive &ar, Atom &a, unsigned int) {
    std::size_t Z = a.type.Z;
    std::size_t idx = a.type.index;
    ar & Z & idx;
    if constexpr (Archive::is_loading::value) {
      if (auto s = Species::find_by_Z(Z)) {
        s->index = idx;
        a.type = *s;
      } else {
        a.type =
            Species{idx}; // synthetic atom (no element) round-trips by slot
      }
    }
    ar & a.pos(0) & a.pos(1) & a.pos(2);
    ar & a.ref.force(0) & a.ref.force(1) & a.ref.force(2);
    ar & a.neighbors;
  }
};

template <> struct Serializer<Configuration> {
  template <CArchive Archive>
  static void apply(Archive &ar, Configuration &c, unsigned int) {
    ar & c.atoms & c.ref.energy & c.weight & c.name;
    Mat3 box = std::holds_alternative<PeriodicBC>(c.bc)
                   ? std::get<PeriodicBC>(c.bc).box()
                   : Mat3::Identity();
    for (auto [i, j] : tensor3D_indices) {
      ar &box(i, j);
    }
    c.bc = PeriodicBC(box); // no-op on save; restores correctly on load
    for (auto [i, j] : tensor3D_indices) {
      ar & c.ref.stress(i, j);
    }
  }
};

} // namespace forcesmith
