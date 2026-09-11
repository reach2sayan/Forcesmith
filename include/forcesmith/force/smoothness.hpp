#pragma once

#include "forcesmith/core/fields.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <string_view>

namespace forcesmith {

// Constrained: unconstrained, these defaults would silently answer 0 for any type.
template <class M>
concept CSmoothable = requires(const M &m) { m.ntypes; };

template <CSmoothable M> std::size_t model_smoothness_count(const M &m) {
  std::size_t n = 0;
  for_each_table(m, [&](const auto &t, std::string_view) {
    n += smoothness_count_range(t);
  });
  return n;
}

template <CSmoothable M>
void model_write_smoothness(const M &m, Eigen::VectorXd &dst, std::size_t off,
                            double weight) {
  for_each_table(m, [&](const auto &t, std::string_view) {
    write_smoothness_range(t, dst, off, weight);
  });
}

} // namespace forcesmith
