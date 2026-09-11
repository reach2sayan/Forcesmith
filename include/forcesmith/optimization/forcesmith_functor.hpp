#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/radial_potential.hpp"
#include "forcesmith/force/force_calculator.hpp"

#include <Eigen/Core>
#include <span>
#include <vector>

namespace forcesmith {

struct ForcesmithFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  ForcesmithFunctor(std::span<Configuration> configs, ForceCalculator model,
                    double energy_weight = 1.0, double stress_weight = 0.0,
                    double smooth_weight = 0.0, bool stream_jacobian = false);

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

  constexpr int inputs() const { return inputs_; }
  constexpr int values() const { return values_; }

  const ForceCalculator &model() const { return model_; }
  ForceCalculator &model() { return model_; }

private:
  bool df_cached(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

  std::span<Configuration> configs_;
  mutable ForceCalculator model_;
  double energy_weight_;
  double stress_weight_;
  double smooth_weight_;
  bool stream_jacobian_ = false;
  int smooth_count_ = 0; // # curvature residuals (0 when smooth_weight_ == 0)
  int inputs_ = 0;
  int values_ = 0;
  std::vector<int> row_offset_;
  mutable std::uint64_t iter_ = 0;
  mutable Eigen::VectorXd last_fvec_;
  mutable double grad_norm_ = 0.0;
};

} // namespace forcesmith
