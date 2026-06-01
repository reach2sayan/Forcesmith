#include "potfit/force/smoothness.hpp"

#include "potfit/force/force_calculator_concept.hpp" // *_smoothness_range helpers

namespace potfit {

// EAM: pair + density + embedding tables.
std::size_t model_smoothness_count(const EAMForceCalculator &m) {
  return smoothness_count_range(m.pair) + smoothness_count_range(m.density) +
         smoothness_count_range(m.embedding);
}
void model_write_smoothness(const EAMForceCalculator &m, Eigen::VectorXd &dst,
                            std::size_t off, double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
  write_smoothness_range(m.density, dst, off, weight);
  write_smoothness_range(m.embedding, dst, off, weight);
}

// Pair: single pair table.
std::size_t model_smoothness_count(const PairForceCalculator &m) {
  return smoothness_count_range(m.pair);
}
void model_write_smoothness(const PairForceCalculator &m, Eigen::VectorXd &dst,
                            std::size_t off, double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
}

// ADP: pair + density + embedding + dipole + quadrupole tables.
std::size_t model_smoothness_count(const ADPForceCalculator &m) {
  return smoothness_count_range(m.pair) + smoothness_count_range(m.density) +
         smoothness_count_range(m.embedding) +
         smoothness_count_range(m.dipole) +
         smoothness_count_range(m.quadrupole);
}
void model_write_smoothness(const ADPForceCalculator &m, Eigen::VectorXd &dst,
                            std::size_t off, double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
  write_smoothness_range(m.density, dst, off, weight);
  write_smoothness_range(m.embedding, dst, off, weight);
  write_smoothness_range(m.dipole, dst, off, weight);
  write_smoothness_range(m.quadrupole, dst, off, weight);
}

// Angular: pair + radial + angular tables.
std::size_t model_smoothness_count(const AngularForceCalculator &m) {
  return smoothness_count_range(m.pair) + smoothness_count_range(m.radial) +
         smoothness_count_range(m.angular);
}
void model_write_smoothness(const AngularForceCalculator &m,
                            Eigen::VectorXd &dst, std::size_t off,
                            double weight) {
  write_smoothness_range(m.pair, dst, off, weight);
  write_smoothness_range(m.radial, dst, off, weight);
  write_smoothness_range(m.angular, dst, off, weight);
}

} // namespace potfit
