#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

namespace potfit {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
// Symmetric stress tensor stored as full 3×3; upper-triangle convention matches
// original sym_tens fields xx/yy/zz/xy/yz/zx.
using SymTens = Eigen::Matrix3d;

enum class PotentialFormat {
  Analytic = 0,           // format 0: function name + parameters
  TabulatedEqDist = 3,    // format 3: equally-spaced table
  TabulatedNonEqDist = 4, // format 4: non-equally-spaced table
  KIM = 5,                // format 5: OpenKIM
  Unknown
};

constexpr auto tensor3D_indices = std::array{
    std::pair{0, 0}, std::pair{0, 1}, std::pair{0, 2},
    std::pair{1, 0}, std::pair{1, 1}, std::pair{1, 2},
    std::pair{2, 0}, std::pair{2, 1}, std::pair{2, 2},
};

} // namespace potfit
