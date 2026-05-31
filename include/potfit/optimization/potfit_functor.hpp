#pragma once

// Eigen::LevenbergMarquardt functor.
// Satisfies the DenseFunctor concept required by Eigen's LM implementation.
// Residuals: force components (3*N_atoms per conf) + weighted energy (1 per
// conf) + optional stress (6 per conf). Jacobian: central finite differences.

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/force/force_calculator.hpp"

#include <Eigen/Core>
#include <span>

namespace potfit {

struct PotfitFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  PotfitFunctor(std::span<Configuration> configs, ForceCalculator model,
                double energy_weight = 1.0, double stress_weight = 0.0,
                double smooth_weight = 0.0);

  // Evaluate residual vector fvec given parameter vector x.
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

  // Evaluate Jacobian fjac via central finite differences.
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

  constexpr int inputs() const { return inputs_; }
  constexpr int values() const { return values_; }

  // Access the underlying force calculator (e.g. to gather final params).
  const ForceCalculator &model() const { return model_; }
  ForceCalculator &model() { return model_; }

private:
  std::span<Configuration> configs_;
  mutable ForceCalculator
      model_; // mutable: scatter_params updates params during eval
  double energy_weight_;
  double stress_weight_;
  double smooth_weight_;
  int smooth_count_ = 0; // # curvature residuals (0 when smooth_weight_ == 0)
  int inputs_ = 0;
  int values_ = 0;
  mutable std::uint64_t iter_ = 0;
};

} // namespace potfit
