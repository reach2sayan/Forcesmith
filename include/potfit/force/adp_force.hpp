#pragma once

// Step 16 — Angular Dependent Potential (ADP) force calculator.
// Reference: Mishin et al., Phys. Rev. B 72, 144104 (2005).
//
// ADP extends EAM with two pairwise functions u(r) and w(r) that couple to
// per-atom dipole (μ_i) and quadrupole (λ_i) distortion tensors:
//
//   E = Σ_{i<j} φ(r_ij)                        pair
//     + Σ_i F_i(ρ_i)                            EAM embedding
//     + 1/2 Σ_i |μ_i|²                          dipole self-energy
//     + 1/2 Σ_i [||λ_i||_F² − 1/3 (tr λ_i)²]  quadrupole self-energy
//
// with:
//   μ_i   = Σ_j u_{t(i),t(j)}(r_ij) × d_ij       (Vec3)
//   λ_i   = Σ_j w_{t(i),t(j)}(r_ij) × d_ij⊗d_ij  (SymTens)
//   ρ_i   = Σ_j g_{t(j)}(r_ij)
//
// Potential layout (same slot indexing as EAMForceCalculator):
//   pair_pots / u_pots / w_pots — size ntypes*(ntypes+1)/2
//   rho_pots  / F_pots          — size ntypes

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/force/force_calculator.hpp"

#include <cstdint>
#include <vector>

namespace potfit {

struct ADPForceCalculator {
  int ntypes = 1;
  std::vector<Potential> pair_pots; // ntypes*(ntypes+1)/2
  std::vector<Potential> rho_pots;  // ntypes
  std::vector<Potential> F_pots;    // ntypes
  std::vector<Potential> u_pots;    // ntypes*(ntypes+1)/2  dipole
  std::vector<Potential> w_pots;    // ntypes*(ntypes+1)/2  quadrupole
  std::uint64_t conf_index = 0;

  void eval_forces(Configuration &cfg) const;

private:
  static int pair_slot(int a, int b, int n) noexcept {
    if (a > b)
      std::swap(a, b);
    return a * n - a * (a - 1) / 2 + (b - a);
  }
};

static_assert(ForceCalculator<ADPForceCalculator>);

} // namespace potfit
