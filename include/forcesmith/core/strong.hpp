#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>

namespace forcesmith {

template <typename T, typename Tag, bool Implicit = false, T Default = T{}>
class Strong {
public:
  using value_type = T;
  constexpr Strong() = default;
  explicit constexpr Strong(T v) noexcept : value_{v} {}

  [[nodiscard]] constexpr T index() const noexcept { return value_; }
  [[nodiscard]] constexpr T value() const noexcept { return value_; }

  constexpr operator T() const noexcept
    requires Implicit
  {
    return value_;
  }

  friend constexpr bool operator==(Strong, Strong) = default;
  friend constexpr auto operator<=>(Strong, Strong) = default;

private:
  T value_ = Default;
};

} // namespace forcesmith
