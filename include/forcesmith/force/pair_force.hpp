#pragma once

#include "forcesmith/core/fields.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"
#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace forcesmith {

struct PairBond {
  Atom *ai;                   // central atom
  const RadialPotential *pot; // i–j pair potential φ
  Vec3 d;                     // pos_j − pos_i
  double r, inv_r;            // |d| and 1/|d|
  double phi = 0.0;           // pair energy φ(r)
  Vec3 force = Vec3::Zero();  // force on i
  SiteId site{}; // φ cache handle (NeighborEntry::sites[kSitePhi])
};

struct PairForceCalculator
    : ForceCalculatorBase<PairForceCalculator>::with_globals<> {
  using Base = ForceCalculatorBase<PairForceCalculator>::with_globals<>;
  RadialPotentialPair pair;

  static constexpr auto tables =
      std::tuple{TableField{&PairForceCalculator::pair, "", true}};

  void eval_forces(Configuration &cfg) const;
  using Base::eval_forces; // indexed (no-cache) overload
  void prepare(std::span<Configuration> configs) const;

  [[nodiscard]] bool has_analytic_jacobian() const;
  void write_param_jacobian(Configuration &cfg, int row0, double energy_weight,
                            double stress_weight, Eigen::MatrixXd &fjac) const;

private:
  static std::optional<PairBond>
  make_pair_bond(Atom &ai, const NeighborEntry &nb, const RadialPotential &pot);
  static PairBond add_pair_force(PairBond &&pb);
  static PairBond accumulate_pair(Configuration &cfg, PairBond &&pb);
};

static_assert(CForceCalculator<PairForceCalculator>);

PairForceCalculator
make_pair_force_calculator(std::vector<RadialPotential> potentials);

} // namespace forcesmith
