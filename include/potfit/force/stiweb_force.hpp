#pragma once

// Step 19 — Stillinger-Weber (SW) two- and three-body force calculator.
// Reference: Stillinger & Weber, Phys. Rev. B 31, 5262 (1985).
// Matches the original potfit parameterization (force_stiweb.c).
//
// Energy:
//   E = Σ_{i<j}   v2(r_ij)
//     + Σ_i Σ_{j<k ∈ neigh(i)} v3(r_ij, r_ik, θ_jik)
//
//   v2(r)         = (A·r^{−p} − B·r^{−q}) · exp(δ/(r − a1))   for r < a1
//   v3(r1, r2, θ) = λ_ijk (cos θ + 1/3)² h(r1) h(r2)
//   h(r)          = exp(γ/(r − a2))                           for r < a2
//
// 2-body parameters per pair type (paircol = ntypes*(ntypes+1)/2 entries):
//   A, B   — repulsive / attractive amplitudes
//   p, q   — repulsive / attractive exponents (bare r^{−p}, r^{−q}; no σ)
//   delta  — 2-body exponential coefficient
//   a1     — 2-body cutoff
// 3-body parameters per pair type:
//   gamma  — 3-body exponential coefficient
//   a2     — 3-body cutoff (independent of a1)
// 3-body strength λ is per-triplet (central type i, neighbour types j,k;
// symmetric in j,k), stored separately — see StiwebForceCalculator::lambda.

#include "potfit/core/atom.hpp"
#include "potfit/core/param.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace potfit {

struct SWParams {
  Param A = 1.0;     // 2-body repulsive amplitude (eV)
  Param B = 1.0;     // 2-body attractive amplitude (eV)
  Param p = 4.0;     // repulsive exponent
  Param q = 0.0;     // attractive exponent
  Param delta = 1.0; // 2-body exponential coefficient
  Param a1 = 1.8;    // 2-body cutoff
  Param gamma = 1.0; // 3-body exponential coefficient
  Param a2 = 1.8;    // 3-body cutoff
};

// params — one SWParams per unique pair type (paircol = ntypes*(ntypes+1)/2),
// access via params(ti, tj).
// lambda — per-triplet 3-body strength λ[i][j][k] (symmetric in j,k), stored as
// a flat vector of ntypes·paircol entries, indexed i·paircol + pair_slot(j,k).
struct StiwebForceCalculator : ForceCalculatorBase<StiwebForceCalculator>,
                               NoGlobals {
  SymmetricMatrix<SWParams> params;
  std::vector<Param> lambda;

  // Upper-triangular slot for an unordered (j,k) pair, identical to
  // SymmetricMatrix::slot — keeps λ indexing consistent with `params`.
  constexpr std::size_t pair_slot(std::size_t a, std::size_t b) const {
    if (a > b) {
      std::swap(a, b);
    }
    return a * ntypes - a * (a - 1) / 2 + (b - a);
  }
  constexpr std::size_t lambda_index(std::size_t ti, std::size_t tj,
                                     std::size_t tk) const {
    const std::size_t paircol = ntypes * (ntypes + 1) / 2;
    return ti * paircol + pair_slot(tj, tk);
  }
  constexpr const Param &lambda_at(std::size_t ti, std::size_t tj,
                                   std::size_t tk) const {
    return lambda[lambda_index(ti, tj, tk)];
  }

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  double max_cutoff() const;
};

static_assert(ForceCalculatorModel<StiwebForceCalculator>);

} // namespace potfit
