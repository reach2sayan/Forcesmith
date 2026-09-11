#pragma once

#include <Eigen/Core>

#include <cstddef>

namespace forcesmith {

// Constrained: unconstrained, these defaults would silently answer 0 for any type.
template <class T>
concept CCurvable = requires(const T &t) { t.span(); };

template <CCurvable T> constexpr std::size_t curvature_count(const T &) noexcept {
  return 0;
}

template <CCurvable T>
void write_curvature(const T &, Eigen::VectorXd &, std::size_t /*off*/,
                     double /*weight*/) noexcept {}

} // namespace forcesmith
