#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/param.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <boost/describe/class.hpp>
#include <cstddef>
#include <utility>

#include <vector>

namespace forcesmith {

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
BOOST_DESCRIBE_STRUCT(SWParams, (), (A, B, p, q, delta, a1, gamma, a2))

struct StiwebForceCalculator : ForceCalculatorBase<StiwebForceCalculator> {
  using Base = ForceCalculatorBase<StiwebForceCalculator>;
  SymmetricMatrix<SWParams> params;
  std::vector<Param> lambda;

  constexpr const Param &lambda_at(std::size_t ti, std::size_t tj,
                                   std::size_t tk) const {
    return lambda[lambda_index(ti, tj, tk)];
  }

  void eval_forces(Configuration &cfg) const;
  using Base::eval_forces; // indexed (no-cache)

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const;
  double max_cutoff() const;

private:
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
};

static_assert(CForceCalculator<StiwebForceCalculator>);

} // namespace forcesmith
