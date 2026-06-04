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
#include <vector>

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
  bool df_cached(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

  std::span<Configuration> configs_;
  mutable ForceCalculator model_;
  double energy_weight_;
  double stress_weight_;
  double smooth_weight_;
  int smooth_count_ = 0; // # curvature residuals (0 when smooth_weight_ == 0)
  int inputs_ = 0;
  int values_ = 0;
  // Prefix sum (size configs+1) of per-config residual counts: config c writes
  // its residuals to fvec[row_offset_[c] .. row_offset_[c+1]).
  // row_offset_.back() is where the (serial) smoothness block begins.
  // Precomputed so the parallel config loop writes disjoint slices without a
  // running counter.
  std::vector<int> row_offset_;
  mutable std::uint64_t iter_ = 0;
  // Most recent residual vector from operator(), and the gradient norm ‖Jᵀf‖
  // computed from it in df(). operator() emits on_iteration before df() runs for
  // the same x, so the logged |grad| reflects the *previous* linearisation —
  // a one-iteration lag that is fine for a progress log.
  mutable Eigen::VectorXd last_fvec_;
  mutable double grad_norm_ = 0.0;
};

} // namespace potfit
