#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/param.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <boost/describe/class.hpp>

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
  Param omega{1.0, true};
};
BOOST_DESCRIBE_STRUCT(TersoffParams, (),
                      (A, B, lambda, mu, beta, n, c, d, h, R, S, omega))

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

struct TersoffForceCalculator : ForceCalculatorBase<TersoffForceCalculator> {
  using Base = ForceCalculatorBase<TersoffForceCalculator>;
  SymmetricMatrix<TersoffParams> params;

  void eval_forces(Configuration &cfg) const;
  using Base::eval_forces; // indexed (no-cache)

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;

private:
  static std::array<Param *, 12> tersoff_fields(TersoffParams &p);
  static std::array<const Param *, 12> tersoff_fields(const TersoffParams &p);

  static double fc_val(double r, double R, double S) noexcept;
  static double dfc_val(double r, double R, double S) noexcept;
  static double g_val(double c, const TersoffParams &p) noexcept;
  static double dg_val(double c, const TersoffParams &p) noexcept;
  static double bond_order(double zeta, const TersoffParams &p) noexcept;
  static double dbond_dzeta(double zeta, const TersoffParams &p) noexcept;

  static std::optional<Bond> make_bond(const TersoffParams &p, const Vec3 &d1,
                                       double r1);
  std::optional<Bond> add_zeta(const Atom &ai, std::size_t jj,
                               Bond &&bond) const;
  static Bond add_bond_order(Bond &&bond);
  static Bond accumulate_pair(Atom &ai, std::size_t jj, Configuration &cfg,
                              Bond &&bond);
  std::optional<Bond> accumulate_three_body(Atom &ai, std::size_t jj,
                                            Configuration &cfg,
                                            Bond &&bond) const;
};

static_assert(CForceCalculator<TersoffForceCalculator>);

} // namespace forcesmith
