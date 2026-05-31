#pragma once

// Tersoff bond-order force calculator.
// Matches force_tersoff.c from original potfit.
//
// Energy (sum over ordered pairs with factor 1/2):
//   E = Σ_{i<j} f_c(r_ij) [A exp(-λ r_ij) - b_ij × B exp(-μ r_ij)]
//   b_ij = (1 + (β ζ_ij)^n)^{-1/(2n)}
//   ζ_ij = Σ_{k≠j, neighbor of i} f_c(r_ik) × g(cos θ_{ijk})
//   g(c)  = 1 + c²/d² - c²/[d² + (h - c)²]
//   f_c   = smooth cosine cutoff between R and S

#include "potfit/core/atom.hpp"
#include "potfit/core/param.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>

namespace potfit {

struct TersoffParams {
  Param A      = 1.0; // repulsive pre-factor (eV)
  Param B      = 1.0; // attractive pre-factor (eV)
  Param lambda = 1.0; // repulsive decay (1/Å)
  Param mu     = 1.0; // attractive decay (1/Å)
  Param beta   = 1.0; // coordination weight
  Param n      = 1.0; // bond-order exponent
  Param c      = 1.0; // angular function numerator width
  Param d      = 1.0; // angular function denominator width
  Param h      = 0.0; // angular function shift
  Param R      = 2.5; // inner cutoff (Å)
  Param S      = 3.0; // outer cutoff (Å)
  // Bond-order mixing weight ω for the i–k pair in the ζ sum (potfit's omega).
  // Defaults to 1.0 and fixed, matching potfit's diagonal (same-type) pairs;
  // the reader frees it for cross-type pairs that supply an explicit value.
  Param omega{1.0, true};
};

struct Bond {
  const TersoffParams *p; // i–j pair parameters
  Vec3 d1;                // pos_j − pos_i
  double r1;              // |d1|
  double fc, dfc;         // cutoff f_c(r1) and its derivative
  double VR, VA;          // repulsive / attractive pair terms
  double VRp, VAp;        // their radial derivatives
  double zeta = 0.0;      // angular sum ζ_ij
  double b = 1.0;         // bond order b_ij
};

// params — one TersoffParams per unique pair type (paircol =
// ntypes*(ntypes+1)/2). Access via params(ti, tj).
struct TersoffForceCalculator : ForceCalculatorBase<TersoffForceCalculator> {
  SymmetricMatrix<TersoffParams> params;

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void        gather_params(Eigen::VectorXd& dst, std::size_t off) const;
  void        scatter_params(const Eigen::VectorXd& src, std::size_t off);
  double      max_cutoff() const;
};

static_assert(ForceCalculatorModel<TersoffForceCalculator>);

} // namespace potfit
