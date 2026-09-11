#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <numeric>
#include <span>
#include <tuple>

namespace forcesmith {

struct EAMForceCalculator
    : ForceCalculatorBase<EAMForceCalculator>::with_globals<> {
  using Base = ForceCalculatorBase<EAMForceCalculator>::with_globals<>;
  RadialPotentialPair pair;
  RadialPotentialArray density;
  RadialPotentialArray embedding;

  static constexpr auto tables = std::tuple{
      TableField{
          .pointer = &EAMForceCalculator::pair, .name = "pair", .radial = true},
      TableField{.pointer = &EAMForceCalculator::density,
                 .name = "density",
                 .radial = true},
      TableField{.pointer = &EAMForceCalculator::embedding,
                 .name = "embedding",
                 .radial = false}};

  void eval_forces(Configuration &cfg) const;
  using Base::eval_forces; // indexed (no-cache) overload

  void prepare(std::span<Configuration> configs) const;

  [[nodiscard]] bool has_analytic_jacobian() const;
  void write_param_jacobian(Configuration &cfg, int row0, double energy_weight,
                            double stress_weight, Eigen::MatrixXd &fjac) const;

};

static_assert(CForceCalculator<EAMForceCalculator>);

} // namespace forcesmith
