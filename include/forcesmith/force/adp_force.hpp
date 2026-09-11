#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>

namespace forcesmith {

struct PairForce {
  const Atom *ai;            // central atom
  const Atom *aj;            // neighbour
  Vec3 d;                    // pos_j − pos_i
  double r, inv_r;           // |d| and 1/|d|
  double phi = 0.0;          // pair energy φ(r) (filled by the EAM stage)
  Vec3 force = Vec3::Zero(); // running F_i
  std::array<SiteId, kNeighborSiteCount> sites = {};
};

struct ADPForceCalculator : ForceCalculatorBase<ADPForceCalculator> {
  using Base = ForceCalculatorBase<ADPForceCalculator>;
  RadialPotentialPair pair;       // φ_{ij}(r)  pair repulsion (1 x num_pairs)
  RadialPotentialArray density;   // g_i(r) electron density (1 x ntypes)
  RadialPotentialArray embedding; // F_i(ρ) embedding energy (1 x ntypes)
  RadialPotentialPair dipole;     // u_{ij}(r)  dipole coupling (1 x num_pairs)
  RadialPotentialPair quadrupole;

  static constexpr auto tables = std::tuple{
      TableField{
          .pointer = &ADPForceCalculator::pair, .name = "pair", .radial = true},
      TableField{.pointer = &ADPForceCalculator::density,
                 .name = "density",
                 .radial = true},
      TableField{.pointer = &ADPForceCalculator::embedding,
                 .name = "embedding",
                 .radial = false},
      TableField{.pointer = &ADPForceCalculator::dipole,
                 .name = "dipole",
                 .radial = true},
      TableField{.pointer = &ADPForceCalculator::quadrupole,
                 .name = "quadrupole",
                 .radial = true}}; // quadrupole coupling (1 x num_pairs)

  void eval_forces(Configuration &cfg) const;
  using Base::eval_forces; // indexed (no-cache) overload

  [[nodiscard]] bool has_analytic_jacobian() const;
  void write_param_jacobian(Configuration &cfg, int row0, double energy_weight,
                            double stress_weight, Eigen::MatrixXd &fjac) const;

  void prepare(std::span<Configuration> configs) const;

private:
  static FORCE_INLINE double quad_nu(const SymTens &M, const Vec3 &d) {
    return d.dot(M * d) - d.squaredNorm() / 3.0 * M.trace();
  }
  static FORCE_INLINE Vec3 quad_xi(const SymTens &M, const Vec3 &d) {
    return M * d - (M.trace() / 3.0) * d;
  }

  static std::optional<PairForce> make_pair_force(const Atom &ai,
                                                  const NeighborEntry &nb);
  PairForce add_eam_force(PairForce &&pf) const;
  PairForce add_dipole_force(PairForce &&pf) const;
  PairForce add_quadrupole_force(PairForce &&pf) const;
  static PairForce accumulate(Atom &ai, Configuration &cfg, PairForce &&pf);
};

static_assert(CForceCalculator<ADPForceCalculator>);

} // namespace forcesmith
