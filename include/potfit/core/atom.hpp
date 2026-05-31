#pragma once

#include "potfit/core/boundary_conditions.hpp"
#include "potfit/core/serialization.hpp"
#include "potfit/core/types.hpp"

#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <numeric>
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
  std::size_t type = 0;
  std::size_t conf = 0;
  Vec3 pos = Vec3::Zero();
  Vec3 force = Vec3::Zero();      // reference (target) force
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
  double energy = 0.0;
  double weight = 1.0;
  SymTens stress = SymTens::Zero();
  double calc_energy = 0.0;
  SymTens calc_stress = SymTens::Zero();
  double calc_limit = 0.0; // accumulated F(ρ) out-of-range penalty (RESCALE-style)
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
    ar & a.type & a.conf;
    ar & a.pos(0) & a.pos(1) & a.pos(2);
    ar & a.force(0) & a.force(1) & a.force(2);
    ar & a.neighbors;
  }
};

template <> struct Serializer<Configuration> {
  template <class Archive>
  static void apply(Archive &ar, Configuration &c, unsigned int) {
    ar & c.atoms & c.energy & c.weight;
    Mat3 box = std::holds_alternative<PeriodicBC>(c.bc)
                   ? std::get<PeriodicBC>(c.bc).box()
                   : Mat3::Identity();
    for (auto [i, j] : tensor3D_indices) {
      ar &box(i, j);
    }
    c.bc = PeriodicBC(box); // no-op on save; restores correctly on load
    for (auto [i, j] : tensor3D_indices) {
      ar & c.stress(i, j);
    }
  }
};

} // namespace potfit
