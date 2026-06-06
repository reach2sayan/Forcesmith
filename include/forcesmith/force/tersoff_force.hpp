#pragma once

// Tersoff bond-order force calculator.
// Matches force_tersoff.c from original forcesmith.
//
// Energy (sum over ordered pairs with factor 1/2):
//   E = Σ_{i<j} f_c(r_ij) [A exp(-λ r_ij) - b_ij × B exp(-μ r_ij)]
//   b_ij = (1 + (β ζ_ij)^n)^{-1/(2n)}
//   ζ_ij = Σ_{k≠j, neighbor of i} f_c(r_ik) × g(cos θ_{ijk})
//   g(c)  = 1 + c²/d² - c²/[d² + (h - c)²]
//   f_c   = smooth cosine cutoff between R and S

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/param.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <optional>

namespace forcesmith {

struct TersoffParams {
  Param A = 1.0;      // repulsive pre-factor (eV)
  Param B = 1.0;      // attractive pre-factor (eV)
  Param lambda = 1.0; // repulsive decay (1/Å)
  Param mu = 1.0;     // attractive decay (1/Å)
  Param beta = 1.0;   // coordination weight
  Param n = 1.0;      // bond-order exponent
  Param c = 1.0;      // angular function numerator width
  Param d = 1.0;      // angular function denominator width
  Param h = 0.0;      // angular function shift
  Param R = 2.5;      // inner cutoff (Å)
  Param S = 3.0;      // outer cutoff (Å)
  // Bond-order mixing weight ω for the i–k pair in the ζ sum (forcesmith's
  // omega). Defaults to 1.0 and fixed, matching forcesmith's diagonal
  // (same-type) pairs; the reader frees it for cross-type pairs that supply an
  // explicit value.
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
struct TersoffForceCalculator : ForceCalculatorBase<TersoffForceCalculator>,
                                NoGlobals {
  SymmetricMatrix<TersoffParams> params;

  void eval_forces(Configuration &cfg) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;

private:
  // The 12 free-able parameter fields of a TersoffParams, in serialization
  // order (used by param_count / gather_params / scatter_params).
  static std::array<Param *, 12> tersoff_fields(TersoffParams &p);
  static std::array<const Param *, 12> tersoff_fields(const TersoffParams &p);

  // Smooth cosine cutoff f_c(r) between R and S, and its derivative.
  static double fc_val(double r, double R, double S) noexcept;
  static double dfc_val(double r, double R, double S) noexcept;
  // Angular function g(cos θ) = 1 + c²/d² − c²/[d² + (h − cos θ)²], and dg/dc.
  static double g_val(double c, const TersoffParams &p) noexcept;
  static double dg_val(double c, const TersoffParams &p) noexcept;
  // Bond order b_ij = (1 + (β ζ)^n)^{−1/(2n)}, and db/dζ.
  static double bond_order(double zeta, const TersoffParams &p) noexcept;
  static double dbond_dzeta(double zeta, const TersoffParams &p) noexcept;

  // Each stage takes a Bond, performs one conceptual step, and passes it on. An
  // empty std::optional means "this bond contributes nothing" (outside the
  // cutoff shell, or no angular neighbours), so the chain short-circuits.
  // Stage 1 — pair terms. Empty unless i–j lies inside the cutoff shell.
  static std::optional<Bond> make_bond(const TersoffParams &p, const Vec3 &d1,
                                       double r1);
  // Stage 2 — angular sum ζ_ij = Σ_{k≠j} ω_ik f_c(r_ik) g(cos θ_ijk).
  std::optional<Bond> add_zeta(const Atom &ai, std::size_t jj,
                               Bond &&bond) const;
  // Stage 3 — bond order b_ij from ζ_ij.
  static Bond add_bond_order(Bond &&bond);
  // Stage 4 — accumulate energy and the (b fixed) pair force / virial.
  static Bond accumulate_pair(Atom &ai, std::size_t jj, Configuration &cfg,
                              Bond &&bond);
  // Stage 5 — three-body force from ∂b_ij/∂ζ × ∂ζ/∂r_n. Empty when ζ = 0.
  std::optional<Bond> accumulate_three_body(Atom &ai, std::size_t jj,
                                            Configuration &cfg,
                                            Bond &&bond) const;
};

static_assert(ForceCalculatorModel<TersoffForceCalculator>);

} // namespace forcesmith
