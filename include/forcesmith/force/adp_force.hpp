#pragma once

// ADP (Angular Dependent Potential) force calculator.
// Reference: Mishin et al., Phys. Rev. B 72, 144104 (2005).
//
// Extends EAM with dipole (μ_i) and quadrupole (λ_i) distortion tensors:
//
//   E = Σ_{i<j} φ(r_ij)                        pair
//     + Σ_i F_i(ρ_i)                            EAM embedding
//     + 1/2 Σ_i |μ_i|²                          dipole self-energy
//     + 1/2 Σ_i [||λ_i||_F² − 1/3 (tr λ_i)²]  quadrupole self-energy
//
// with:
//   ρ_i = Σ_j g_{t(j)}(r_ij)
//   μ_i = Σ_j u_{t(i),t(j)}(r_ij) × d_ij
//   λ_i = Σ_j w_{t(i),t(j)}(r_ij) × d_ij⊗d_ij

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace forcesmith {

struct PairForce {
  const Atom *ai;            // central atom
  const Atom *aj;            // neighbour
  Vec3 d;                    // pos_j − pos_i
  double r, inv_r;           // |d| and 1/|d|
  double phi = 0.0;          // pair energy φ(r) (filled by the EAM stage)
  Vec3 force = Vec3::Zero(); // running F_i
  // Spline-cache handles copied from the bond (NeighborEntry::sites), consumed
  // by the add_*_force stages. Default (none) when unset (direct eval/deriv).
  std::array<SiteId, kNeighborSiteCount> sites = {};
};

struct ADPForceCalculator : ForceCalculatorBase<ADPForceCalculator>, NoGlobals {
  // num_pairs = ntypes*(ntypes+1)/2
  PotentialPair pair; // φ_{ij}(r)  pair repulsion (1 x num_pairs)
  PotentialArray density; // g_i(r) electron density (1 x ntypes)
  PotentialArray embedding; // F_i(ρ) embedding energy (1 x ntypes)
  PotentialPair dipole; // u_{ij}(r)  dipole coupling (1 x num_pairs)
  PotentialPair quadrupole; // quadrupole coupling (1 x num_pairs)

  void eval_forces(Configuration &cfg) const;

  // Primes the spline-cache hints for all five radial-table roles a bond drives
  // (φ, g_j, g_i, dipole, quadrupole).
  void prepare(std::span<Configuration> configs) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;

private:
  // nu(M, d) = d^T M d - r²/3 × tr(M)
  static FORCE_INLINE double quad_nu(const SymTens &M, const Vec3 &d) {
    return d.dot(M * d) - d.squaredNorm() / 3.0 * M.trace();
  }
  // xi(M, d) = M d - tr(M)/3 × d
  static FORCE_INLINE Vec3 quad_xi(const SymTens &M, const Vec3 &d) {
    return M * d - (M.trace() / 3.0) * d;
  }

  // Each stage adds one physical contribution to PairForce::force (the force on
  // atom i) and passes the pair on. An empty std::optional means the two atoms
  // are coincident and contribute nothing, so the chain's .transform
  // short-circuits.
  // Stage 1 — geometry. Empty for coincident atoms.
  static std::optional<PairForce> make_pair_force(const Atom &ai,
                                                  const NeighborEntry &nb);
  // Stage 2 — EAM pair + embedding-gradient force (same as EAMForceCalculator):
  //   F_eam = [dφ/dr + gradF_i×dg_{t(j)}/dr + gradF_j×dg_{t(i)}/dr] × r̂
  PairForce add_eam_force(PairForce &&pf) const;
  // Stage 3 — dipole force (Mishin 2005, derived via ∂E_dip/∂r_i):
  //   F_dip = du/r × (μ_i·d − μ_j·d) × d + u × (μ_i − μ_j)
  PairForce add_dipole_force(PairForce &&pf) const;
  // Stage 4 — quadrupole force (Mishin 2005, derived via ∂E_quad/∂r_i):
  //   F_quad = dw/r × [ν(λ_i,d) + ν(λ_j,d)] × d + 2w × [ξ(λ_i,d) + ξ(λ_j,d)]
  //   where ν(M,d)=d^T M d − r²/3 tr(M),  ξ(M,d)=Md − tr(M)/3 d
  PairForce add_quadrupole_force(PairForce &&pf) const;
  // Stage 5 — commit the pair's energy, force and virial to the configuration.
  static PairForce accumulate(Atom &ai, Configuration &cfg, PairForce &&pf);
};

static_assert(ForceCalculatorModel<ADPForceCalculator>);

} // namespace forcesmith
