#pragma once

#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"
#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace forcesmith {

struct PairBond {
  Atom *ai;                  // central atom
  const Potential *pot;      // i–j pair potential φ
  Vec3 d;                    // pos_j − pos_i
  double r, inv_r;           // |d| and 1/|d|
  double phi = 0.0;          // pair energy φ(r)
  Vec3 force = Vec3::Zero(); // force on i
  SiteId site{};             // φ cache handle (NeighborEntry::sites[kSitePhi])
};

struct PairForceCalculator : WithGlobals {
  std::size_t ntypes = 1;
  std::uint64_t conf_index = 0;
  PotentialPair pair;

  void eval_forces(Configuration &cfg) const;

  // Fit-time setup (single-threaded): see EAMForceCalculator::prepare. Primes
  // the φ spline-cache hint on every bond. Optional — eval_forces falls back to
  // a direct eval/deriv(r) when a bond was not primed.
  void prepare(std::span<Configuration> configs) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;
  void broadcast_globals();
  void finalize_globals();

private:
  // Stage 1 — geometry + cutoff gate. The neighbour list is built with the
  // global max_cutoff(); gate each contribution on this potential's own range
  // [rmin, rmax). Empty for coincident atoms or out-of-range separations.
  static std::optional<PairBond>
  make_pair_bond(Atom &ai, const NeighborEntry &nb, const Potential &pot);
  // Stage 2 — radial force φ′(r).
  static PairBond add_pair_force(PairBond &&pb);
  // Stage 3 — commit energy / force / virial (0.5 for the full neighbor list).
  static PairBond accumulate_pair(Configuration &cfg, PairBond &&pb);
};

static_assert(ForceCalculatorModel<PairForceCalculator>);

// Build a PairForceCalculator that owns the given flat potential list.
// ntypes is inferred from paircol = ntypes*(ntypes+1)/2.
PairForceCalculator
make_pair_force_calculator(std::vector<Potential> potentials);

} // namespace forcesmith
