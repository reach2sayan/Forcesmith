#include "potfit/core/neighbor_list.hpp"
#include "potfit/core/potential_base.hpp"

#include <ranges>

namespace potfit {

namespace {

void build_impl(Configuration &cfg, double rcut, const PotentialPair *pots) {
  std::ranges::for_each(cfg.atoms, [](auto &a) { a.neighbors.clear(); });
  const double rcut2 = rcut * rcut;

  std::ranges::for_each(std::views::cartesian_product(cfg.atoms, cfg.atoms) |
                            std::views::filter([](auto &&p) {
                              return &std::get<0>(p) != &std::get<1>(p);
                            }),
                        [&](auto &&pair) {
                          auto &&[ai, aj] = pair;
                          const Vec3 d = bc_min_image(cfg.bc, aj.pos - ai.pos);
                          if (d.squaredNorm() >= rcut2)
                            return;

                          const Potential *pot = nullptr;
                          if (pots && ai.type < pots->ntypes() &&
                              aj.type < pots->ntypes())
                            pot = &(*pots)[ai.type, aj.type];

                          NeighborEntry entry;
                          entry.neighbor = &aj;
                          entry.pot = pot;
                          entry.dist = d;
                          ai.neighbors.push_back(entry);
                        });
}

} // anonymous namespace

void build_neighbor_list(Configuration &cfg, double rcut) {
  build_impl(cfg, rcut, nullptr);
}

void build_neighbor_list(Configuration &cfg, double rcut,
                         const PotentialPair &pots) {
  build_impl(cfg, rcut, &pots);
}

} // namespace potfit
