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

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <optional>

namespace potfit {

struct PairForce {
  const Atom *ai;            // central atom
  const Atom *aj;            // neighbour
  Vec3 d;                    // pos_j − pos_i
  double r, inv_r;           // |d| and 1/|d|
  double phi = 0.0;          // pair energy φ(r) (filled by the EAM stage)
  Vec3 force = Vec3::Zero(); // running F_i
};

// Potential tables for ntypes element types (paircol = ntypes*(ntypes+1)/2):
//   pair       — φ_{ij}(r)  pair repulsion,         paircol entries
//   density    — g_i(r)     electron density,        ntypes entries
//   embedding  — F_i(ρ)     embedding energy,        ntypes entries
//   dipole     — u_{ij}(r)  dipole coupling,         paircol entries
//   quadrupole — w_{ij}(r)  quadrupole coupling,     paircol entries
struct ADPForceCalculator : ForceCalculatorBase<ADPForceCalculator>, NoGlobals {
  PotentialPair pair;
  PotentialArray density;
  PotentialArray embedding;
  PotentialPair dipole;
  PotentialPair quadrupole;

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;

private:
  // Helpers for quadrupole force terms (see eval_forces for derivation refs).
  // nu(M, d) = d^T M d - r²/3 × tr(M)
  static double quad_nu(const SymTens &M, const Vec3 &d);
  // xi(M, d) = M d - tr(M)/3 × d
  static Vec3 quad_xi(const SymTens &M, const Vec3 &d);

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

} // namespace potfit
