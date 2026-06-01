#pragma once

#include "potfit/core/boundary_conditions.hpp"
#include "potfit/core/serialization.hpp"
#include "potfit/core/species.hpp"
#include "potfit/core/types.hpp"

#include <boost/leaf/result.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <filesystem>
#include <functional>
#include <numeric>
#include <string_view>
#include <vector>

namespace potfit {

struct Atom;
class Potential;

struct NeighborEntry : Serializable<NeighborEntry> {
  const Atom *neighbor = nullptr; // non-owning; restored by build_neighbor_list
  const Potential *pot = nullptr; // non-owning; resolved at build time
  Vec3 dist = Vec3::Zero();
};

struct Atom : Serializable<Atom> {
  Species type{}; // element identity + compact slot (implicitly indexes tables)
  std::size_t conf = 0;
  Vec3 pos = Vec3::Zero();

  struct Reference {
    Vec3 force = Vec3::Zero(); // target force from input data
  } ref;

  Vec3 calc_force = Vec3::Zero(); // written by ForceCalculator
  std::vector<NeighborEntry> neighbors;

  // EAM/ADP scratch fields — zeroed before each force evaluation, not
  // serialized.
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

struct Configuration : Serializable<Configuration> {
  std::vector<Atom> atoms;
  BoundaryConditions bc = PeriodicBC(Mat3::Identity());

  struct Reference {
    double energy = 0.0;
    SymTens stress = SymTens::Zero();
  } ref;

  double weight = 1.0; // fitting weight (not a target — stays flat)
  double calc_energy = 0.0;
  SymTens calc_stress = SymTens::Zero();
  double calc_limit =
      0.0; // accumulated F(ρ) out-of-range penalty (RESCALE-style)

  // ── parsing factories (the object owns its parsing) ───────────────────────
  // Build ONE configuration from a single JSON record (the same shape as one
  // element of the config-file array): {X,Y,Z, E, [W], [S], atoms:[…]}. Atoms
  // carry their Species by identity (symbol/Z from the static catalog); the
  // compact table slot (Species::index) is assigned later by FitSession at
  // freeze, so no registry is needed here. Defined in src/io/config_reader.cpp
  // (keeps nlohmann out of this core header). Leaf-returning, not a throwing
  // ctor — see feedback_error_handling.
  [[nodiscard]] static boost::leaf::result<Configuration>
  from_text(std::string_view json_record);
  [[nodiscard]] static boost::leaf::result<Configuration>
  from_file(const std::filesystem::path &path);
};

// Root-mean-square of the per-atom calculated forces over a configuration.
// Returns 0 for an empty configuration.
inline double force_rms(const Configuration &cfg) {
  const double sq = std::transform_reduce(
      cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{},
      [](const Atom &a) { return a.calc_force.squaredNorm(); });
  return cfg.atoms.empty()
             ? 0.0
             : std::sqrt(sq / static_cast<double>(cfg.atoms.size()));
}

template <> struct Serializer<NeighborEntry> {
  template <class Archive>
  static void apply(Archive &ar, NeighborEntry &e, unsigned int) {
    // Pointers are transient — only geometry is persisted.
    // Call build_neighbor_list after deserialization to restore them.
    ar & e.dist(0) & e.dist(1) & e.dist(2);
  }
};

template <> struct Serializer<Atom> {
  template <class Archive>
  static void apply(Archive &ar, Atom &a, unsigned int) {
    // Species::symbol is a string_view into static storage — persist Z + slot
    // and re-derive symbol/mass from the catalog on load.
    std::size_t Z = a.type.Z;
    std::size_t idx = a.type.index;
    ar & Z & idx & a.conf;
    if constexpr (Archive::is_loading::value) {
      if (auto s = Species::find_by_Z(Z)) {
        s->index = idx;
        a.type = *s;
      } else {
        a.type = Species{idx}; // synthetic atom (no element) round-trips by slot
      }
    }
    ar & a.pos(0) & a.pos(1) & a.pos(2);
    ar & a.ref.force(0) & a.ref.force(1) & a.ref.force(2);
    ar & a.neighbors;
  }
};

template <> struct Serializer<Configuration> {
  template <class Archive>
  static void apply(Archive &ar, Configuration &c, unsigned int) {
    ar & c.atoms & c.ref.energy & c.weight;
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

} // namespace potfit
