#include "potfit/core/neighbor_list.hpp"
#include "potfit/core/potential_base.hpp"

#include <ranges>
#include <span>

namespace potfit {

static constexpr int pair_slot(int a, int b, int ntypes) noexcept {
  if (a > b)
    std::swap(a, b);
  return a * ntypes - a * (a - 1) / 2 + (b - a);
}

void build_neighbor_list(Configuration &cfg, double rcut,
                         std::span<const Potential> pots) {
  const int ntypes = cfg.atoms.empty()
                         ? 1
                         : std::ranges::max(cfg.atoms | std::views::transform(
                                                            [](const auto &a) {
                                                              return a.type + 1;
                                                            }));

  std::ranges::for_each(cfg.atoms, [](auto &a) { a.neighbors.clear(); });
  const double rcut2 = rcut * rcut;
  auto unique_atom_view = std::views::cartesian_product(cfg.atoms, cfg.atoms) |
                             std::views::filter([](auto &&pair) {
                               return &(std::get<0>(pair)) !=
                                      &(std::get<1>(pair));
                             });

  std::ranges::for_each(unique_atom_view, [&](auto &&pair) {
    auto &&[ai, aj] = pair;

    const Vec3 d = bc_min_image(cfg.bc, aj.pos - ai.pos);
    const double r2 = d.squaredNorm();
    if (r2 >= rcut2)
      return;

    const int s = pair_slot(ai.type, aj.type, ntypes);

    ai.neighbors.push_back(NeighborEntry{
        .neighbor = &aj,
        .pot = (s < static_cast<int>(pots.size())) ? &pots[s] : nullptr,
        .dist = d,
    });
  });
}

} // namespace potfit
