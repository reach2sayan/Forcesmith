#pragma once

#include "potfit/core/types.hpp"
#include <variant>

namespace potfit {

class PeriodicBC {
private:
  template <class RoundOp>
  [[nodiscard]] Vec3 map_fractional(const Vec3 &v,
                                    RoundOp &&round) const noexcept {
    Vec3 frac = inv_box_ * v;
    frac = frac.array() - round(frac.array());
    return box_ * frac;
  }
public:
  explicit PeriodicBC(const Mat3 &box) { set_box(box); }
  void set_box(const Mat3 &box) {
    box_ = box;
    inv_box_ = box.inverse();
    volume_ = std::abs(box.determinant());
  }

  [[nodiscard]] constexpr const Mat3 &box() const noexcept { return box_; }
  [[nodiscard]] constexpr const Mat3 &inv_box() const noexcept {
    return inv_box_;
  }
  [[nodiscard]] constexpr double volume() const noexcept { return volume_; }

  // Wrap a Cartesian position into the unit cell [0,1)³ (frac coords).
  [[nodiscard]] Vec3 wrap(const Vec3 &r) const noexcept {
    return map_fractional(r, [](const auto &x) { return x.floor(); });
  }

  // Mini-imag displacement: maps delta into [-0.5, 0.5)³ in frac coords.
  [[nodiscard]] Vec3 min_image(const Vec3 &d) const noexcept {
    return map_fractional(d, [](const auto &x) { return x.round(); });
  }

private:
  Mat3 box_ = Mat3::Identity();
  Mat3 inv_box_ = Mat3::Identity();
  double volume_ = 1.0;
};

class InfiniteBC {
public:
  constexpr explicit InfiniteBC(double volume = 1.0) noexcept
      : volume_(volume) {}

  [[nodiscard]] Vec3 wrap(const Vec3 &r) const noexcept { return r; }
  [[nodiscard]] Vec3 min_image(const Vec3 &d) const noexcept { return d; }
  [[nodiscard]] constexpr double volume() const noexcept { return volume_; }
  constexpr void set_volume(double v) noexcept { volume_ = v; }

private:
  double volume_;
};

using BoundaryConditions = std::variant<PeriodicBC, InfiniteBC>;

[[nodiscard]] Vec3 bc_wrap(const BoundaryConditions &bc, const Vec3 &r);
[[nodiscard]] Vec3 bc_min_image(const BoundaryConditions &bc, const Vec3 &d);
[[nodiscard]] double bc_volume(const BoundaryConditions &bc);

} // namespace potfit
