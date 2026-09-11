#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <tuple>

namespace forcesmith {

struct AngularForceCalculator : ForceCalculatorBase<AngularForceCalculator> {
  using Base = ForceCalculatorBase<AngularForceCalculator>;
  RadialPotentialPair pair;
  RadialPotentialPair radial;
  RadialPotentialArray angular;

  static constexpr auto tables =
      std::tuple{TableField{.pointer = &AngularForceCalculator::pair,
                            .name = "pair",
                            .radial = true},
                 TableField{.pointer = &AngularForceCalculator::radial,
                            .name = "radial",
                            .radial = true},
                 TableField{.pointer = &AngularForceCalculator::angular,
                            .name = "angular",
                            .radial = false}};
  void eval_forces(Configuration &cfg) const;

  [[nodiscard]] bool has_analytic_jacobian() const;
  void write_param_jacobian(Configuration &cfg, int row0, double energy_weight,
                            double stress_weight, Eigen::MatrixXd &fjac) const;
  using Base::eval_forces; // indexed (no-cache)
};

static_assert(CForceCalculator<AngularForceCalculator>);

} // namespace forcesmith
