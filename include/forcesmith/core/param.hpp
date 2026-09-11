#pragma once

#include <limits>

namespace forcesmith {

struct Param {
  double value = 0.0;
  bool fixed = false;
  double min = -std::numeric_limits<double>::infinity();
  double max = std::numeric_limits<double>::infinity();

  constexpr Param() = default;
  constexpr Param(double v, bool f = false) noexcept : value{v}, fixed{f} {}
  constexpr Param(double v, double lo, double hi, bool f = false) noexcept
      : value{v}, fixed{f}, min{lo}, max{hi} {}
  constexpr operator double() const noexcept { return value; }
  constexpr Param &operator=(double v) noexcept {
    value = v;
    return *this;
  }
};

} // namespace forcesmith
