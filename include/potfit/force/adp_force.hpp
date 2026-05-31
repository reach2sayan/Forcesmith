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

namespace potfit {

// Potential tables for ntypes element types (paircol = ntypes*(ntypes+1)/2):
//   pair       — φ_{ij}(r)  pair repulsion,         paircol entries
//   density    — g_i(r)     electron density,        ntypes entries
//   embedding  — F_i(ρ)     embedding energy,        ntypes entries
//   dipole     — u_{ij}(r)  dipole coupling,         paircol entries
//   quadrupole — w_{ij}(r)  quadrupole coupling,     paircol entries
struct ADPForceCalculator : ForceCalculatorBase<ADPForceCalculator> {
  PotentialPair  pair;
  PotentialArray density;
  PotentialArray embedding;
  PotentialPair  dipole;
  PotentialPair  quadrupole;

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void        gather_params(Eigen::VectorXd& dst, std::size_t off) const;
  void        scatter_params(const Eigen::VectorXd& src, std::size_t off);
  double      max_cutoff() const;
};

static_assert(ForceCalculatorModel<ADPForceCalculator>);

} // namespace potfit
