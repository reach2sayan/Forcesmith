#pragma once

// Step 19 — Stillinger-Weber (SW) two- and three-body force calculator.
// Reference: Stillinger & Weber, Phys. Rev. B 31, 5262 (1985).
//
// Energy:
//   E = Σ_{i<j}   v2(r_ij)
//     + Σ_i Σ_{j<k ∈ neigh(i)} v3(r_ij, r_ik, θ_jik)
//
//   v2(r)            = A [B(σ/r)^p − (σ/r)^q] exp(σ/(r − aσ))   for r < aσ
//   v3(r1, r2, θ)    = λ (cos θ + 1/3)² h(r1) h(r2)
//   h(r)             = exp(γσ/(r − aσ))                           for r < aσ
//
// Parameters per pair type (ntypes*(ntypes+1)/2 entries):
//   A      — 2-body amplitude (energy units; A_SW × ε pre-multiplied)
//   B      — 2-body shape (dimensionless)
//   p, q   — repulsive / attractive exponents
//   a      — cutoff in units of σ
//   sigma  — length scale (Å)
//   lambda — 3-body strength (energy units; λ_SW × ε pre-multiplied)
//   gamma  — 3-body radial damping (dimensionless)

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/force/potential_table.hpp"

namespace potfit {

struct SWParams {
  double A = 1.0;      // 2-body amplitude (eV)
  double B = 1.0;      // 2-body inner shape (dimensionless)
  double p = 4.0;      // repulsive exponent
  double q = 0.0;      // attractive exponent
  double a = 1.8;      // cutoff (in units of σ)
  double sigma = 1.0;  // length scale (Å)
  double lambda = 1.0; // 3-body strength (eV)
  double gamma = 1.0;  // 3-body radial damping
};

// params — one SWParams per unique pair type (paircol = ntypes*(ntypes+1)/2).
// Access via params(ti, tj).
struct StiwebForceCalculator : ForceCalculatorBase<StiwebForceCalculator> {
  SymmetricMatrix<SWParams> params;
  void eval_forces(Configuration &cfg) const;
};

static_assert(ForceCalculatorModel<StiwebForceCalculator>);

} // namespace potfit
