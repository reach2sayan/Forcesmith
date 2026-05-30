#pragma once

// Step 9: Eigen::LevenbergMarquardt functor.
// Satisfies the DenseFunctor concept required by Eigen's LM implementation.
// Residuals: force components (3*N_atoms per conf) + weighted energy (1 per
// conf). Jacobian: central finite differences (Δ=1e-5); switch to analytic
// later per potential type.

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"

#include <Eigen/Core>
#include <span>

namespace potfit {

struct PotfitFunctor {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  PotfitFunctor(std::span<Configuration> configs,
                std::span<Potential> potentials, double energy_weight = 1.0);

  // Evaluate residual vector fvec given parameter vector x.
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

  // Evaluate Jacobian fjac via central finite differences.
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

  int inputs() const; // number of free parameters
  int values() const; // number of residuals

  std::span<const Potential> potentials() const { return potentials_; }
  std::span<Potential> potentials() { return potentials_; }

private:
  std::span<Configuration> configs_;
  std::span<Potential> potentials_;
  double energy_weight_;
  int inputs_ = 0;
  int values_ = 0;
  mutable std::uint64_t iter_ = 0;
};

} // namespace potfit
