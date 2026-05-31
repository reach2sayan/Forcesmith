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

  int param_count() const {
    int count = 0;
    for (const auto& p : pair)       count += p.param_count();
    for (const auto& p : density)    count += p.param_count();
    for (const auto& p : embedding)  count += p.param_count();
    for (const auto& p : dipole)     count += p.param_count();
    for (const auto& p : quadrupole) count += p.param_count();
    return count;
  }

  void gather_params(Eigen::VectorXd& dst, int off) const {
    for (const auto& p : pair)       { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : density)    { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : embedding)  { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : dipole)     { p.gather_params(dst, off); off += p.param_count(); }
    for (const auto& p : quadrupole) { p.gather_params(dst, off); off += p.param_count(); }
  }

  void scatter_params(const Eigen::VectorXd& src, int off) {
    for (auto& p : pair)       { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : density)    { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : embedding)  { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : dipole)     { p.scatter_params(src, off); off += p.param_count(); }
    for (auto& p : quadrupole) { p.scatter_params(src, off); off += p.param_count(); }
  }

  double max_cutoff() const {
    double rcut = 0.0;
    for (const auto& p : pair)    rcut = std::max(rcut, p.span().second);
    for (const auto& p : density) rcut = std::max(rcut, p.span().second);
    return rcut;
  }
};

static_assert(ForceCalculatorModel<ADPForceCalculator>);

} // namespace potfit
