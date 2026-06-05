#pragma once

// Curvature (smoothness) regularization customization point, potential level.
//
// A tabulated potential fit from forces/energies alone is rank-deficient
// wherever no training configuration probes that part of the function's domain
// (e.g. pair-potential radii between neighbour shells). The optimizer is free
// to dump arbitrary spikes into that null space. A small Tikhonov penalty on
// the curvature (second difference) of the free knots removes the ambiguity.

#include <Eigen/Core>

#include <cstddef>

namespace forcesmith {

// Number of curvature residuals the potential contributes. Default: none.
template <typename T>
constexpr std::size_t curvature_count(const T &) noexcept {
  return 0;
}

// Write this potential's curvature residuals into dst starting at `off`,
// each scaled by `weight`. Default: nothing to write.
template <typename T>
void write_curvature(const T &, Eigen::VectorXd &, std::size_t /*off*/,
                     double /*weight*/) noexcept {}

} // namespace forcesmith
