#pragma once

namespace potfit {

// A single optimizable scalar that carries a fix/free flag.
// Implicitly converts to double so it can be used directly in arithmetic
// without changing eval code (p.A * exp(...) still compiles when p.A is Param).
struct Param {
  double value = 0.0;
  bool fixed = false;

  constexpr Param() = default;
  constexpr Param(double v, bool f = false) noexcept : value(v), fixed(f) {}
  constexpr operator double() const noexcept { return value; }
  constexpr Param &operator=(double v) noexcept {
    value = v;
    return *this;
  }
};

} // namespace potfit
