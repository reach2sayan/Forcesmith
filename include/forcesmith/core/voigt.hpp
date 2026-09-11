#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace forcesmith {

inline constexpr std::array kVoigt6{
    std::pair{0, 0}, std::pair{1, 1}, std::pair{2, 2},
    std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2},
};
inline constexpr std::size_t kVoigtCount = kVoigt6.size();

} // namespace forcesmith
