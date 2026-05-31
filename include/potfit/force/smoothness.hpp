#pragma once

// Curvature (smoothness) regularization, force-calculator level.
//
// Sums the per-potential curvature residuals (see potfit/potentials/curvature.hpp)
// over every tabulated potential a calculator owns. The default reports zero
// (calculators with no tabulated potentials — e.g. Tersoff/Stiweb, which are
// purely analytic — contribute nothing). Calculators built from `Potential`
// tables overload the two entry points below by iterating their public tables.

#include "potfit/force/force_calculator.hpp"
#include "potfit/force/force_calculator_concept.hpp"

#include <Eigen/Core>

#include <cstddef>

namespace potfit {

// Defaults: a calculator contributes no curvature residuals.
template <typename M>
std::size_t model_smoothness_count(const M &) {
  return 0;
}
template <typename M>
void model_write_smoothness(const M &, Eigen::VectorXd &, std::size_t /*off*/,
                            double /*weight*/) {}

// EAM: pair + density + embedding tables.
inline std::size_t model_smoothness_count(const EAMForceCalculator &m) {
  return smoothness_count_range(m.pair) + smoothness_count_range(m.density) +
         smoothness_count_range(m.embedding);
}
inline void model_write_smoothness(const EAMForceCalculator &m,
                                   Eigen::VectorXd &dst, std::size_t off,
                                   double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
  write_smoothness_range(m.density, dst, off, weight);
  write_smoothness_range(m.embedding, dst, off, weight);
}

// Pair: single pair table.
inline std::size_t model_smoothness_count(const PairForceCalculator &m) {
  return smoothness_count_range(m.pair);
}
inline void model_write_smoothness(const PairForceCalculator &m,
                                   Eigen::VectorXd &dst, std::size_t off,
                                   double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
}

// ADP: pair + density + embedding + dipole + quadrupole tables.
inline std::size_t model_smoothness_count(const ADPForceCalculator &m) {
  return smoothness_count_range(m.pair) + smoothness_count_range(m.density) +
         smoothness_count_range(m.embedding) +
         smoothness_count_range(m.dipole) +
         smoothness_count_range(m.quadrupole);
}
inline void model_write_smoothness(const ADPForceCalculator &m,
                                   Eigen::VectorXd &dst, std::size_t off,
                                   double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
  write_smoothness_range(m.density, dst, off, weight);
  write_smoothness_range(m.embedding, dst, off, weight);
  write_smoothness_range(m.dipole, dst, off, weight);
  write_smoothness_range(m.quadrupole, dst, off, weight);
}

// Angular: pair + radial + angular tables.
inline std::size_t model_smoothness_count(const AngularForceCalculator &m) {
  return smoothness_count_range(m.pair) + smoothness_count_range(m.radial) +
         smoothness_count_range(m.angular);
}
inline void model_write_smoothness(const AngularForceCalculator &m,
                                   Eigen::VectorXd &dst, std::size_t off,
                                   double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
  write_smoothness_range(m.radial, dst, off, weight);
  write_smoothness_range(m.angular, dst, off, weight);
}

} // namespace potfit
