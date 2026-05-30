#include "potfit/core/boundary_conditions.hpp"

#include <variant>

namespace potfit {

Vec3 bc_wrap(const BoundaryConditions &bc, const Vec3 &r) {
  return std::visit([&](const auto &b) { return b.wrap(r); }, bc);
}

Vec3 bc_min_image(const BoundaryConditions &bc, const Vec3 &d) {
  return std::visit([&](const auto &b) { return b.min_image(d); }, bc);
}

double bc_volume(const BoundaryConditions &bc) {
  return std::visit([](const auto &b) { return b.volume(); }, bc);
}

} // namespace potfit
