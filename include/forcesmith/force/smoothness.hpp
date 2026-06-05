#pragma once

// Curvature (smoothness) regularization, force-calculator level.
//
// Sums the per-potential curvature residuals (see
// forcesmith/potentials/curvature.hpp) over every tabulated potential a calculator
// owns. The default reports zero (calculators with no tabulated potentials —
// e.g. Tersoff/Stiweb, which are purely analytic — contribute nothing).
// Calculators built from `Potential` tables overload the two entry points below
// by iterating their public tables.

#include "forcesmith/force/force_calculator.hpp"

#include <Eigen/Core>

#include <cstddef>

namespace forcesmith {

// Defaults: a calculator contributes no curvature residuals. Calculators with
// no tabulated potentials (Tersoff/Stiweb) bind here through std::visit.
template <typename M> constexpr std::size_t model_smoothness_count(const M &) {
  return 0;
}
template <typename M>
constexpr void model_write_smoothness(const M &, Eigen::VectorXd &,
                                      std::size_t /*off*/, double /*weight*/) {}

// Concrete overloads for the table-backed calculators (defined in
// smoothness.cpp). These MUST be declared here so the std::visit dispatch in
// forcesmith_functor.cpp binds EAM/Pair/ADP/Angular to these rather than silently
// falling through to the zero-returning template above.

// EAM: pair + density + embedding tables.
std::size_t model_smoothness_count(const EAMForceCalculator &m);
void model_write_smoothness(const EAMForceCalculator &m, Eigen::VectorXd &dst,
                            std::size_t off, double weight);

// Pair: single pair table.
std::size_t model_smoothness_count(const PairForceCalculator &m);
void model_write_smoothness(const PairForceCalculator &m, Eigen::VectorXd &dst,
                            std::size_t off, double weight);

// ADP: pair + density + embedding + dipole + quadrupole tables.
std::size_t model_smoothness_count(const ADPForceCalculator &m);
void model_write_smoothness(const ADPForceCalculator &m, Eigen::VectorXd &dst,
                            std::size_t off, double weight);

// Angular: pair + radial + angular tables.
std::size_t model_smoothness_count(const AngularForceCalculator &m);
void model_write_smoothness(const AngularForceCalculator &m,
                            Eigen::VectorXd &dst, std::size_t off,
                            double weight);

} // namespace forcesmith
