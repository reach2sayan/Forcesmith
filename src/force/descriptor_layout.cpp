#include "forcesmith/force/descriptor_layout.hpp"

#include "forcesmith/force/potential_table.hpp" // upper_triangle

#include <ranges>

namespace forcesmith {

std::vector<std::optional<Eigen::Index>>
remap_layout(const DescriptorLayout &old_L, const DescriptorLayout &new_L,
             const std::vector<std::optional<std::size_t>> &old_of_new,
             std::size_t S_old, std::size_t S_new) {
  std::vector<std::optional<Eigen::Index>> map(
      static_cast<std::size_t>(new_L.size()));

  for (auto [b, blk] : new_L.blocks_ | std::views::enumerate) {
    const auto block = static_cast<std::size_t>(b);
    const std::size_t count = blk.count;
    const bool per_species = blk.nchan == S_new;

    if (per_species) {
      for (auto [s, s_old] : old_of_new | std::views::enumerate) {
        if (!s_old) {
          continue;
        }
        for (std::size_t t = 0; t < count; ++t) {
          map[static_cast<std::size_t>(
              new_L.index(block, static_cast<std::size_t>(s), t))] =
              old_L.index(block, *s_old, t);
        }
      }
    } else { // per-pair
      for (auto [a, c] : upper_triangle(S_new)) {
        const auto a_old = old_of_new[a];
        const auto c_old = old_of_new[c];
        if (!a_old || !c_old) {
          continue;
        }
        const std::size_t po_new = pair_ordinal(a, c, S_new);
        const std::size_t po_old = pair_ordinal(*a_old, *c_old, S_old);
        for (std::size_t t = 0; t < count; ++t) {
          map[static_cast<std::size_t>(new_L.index(block, po_new, t))] =
              old_L.index(block, po_old, t);
        }
      }
    }
  }
  return map;
}

} // namespace forcesmith
