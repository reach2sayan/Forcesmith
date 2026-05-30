#include "potfit/core/neighbor_list.hpp"
#include "potfit/core/potential_base.hpp"

#include <algorithm>
#include <span>

namespace potfit {

// Upper-triangle column index for pair (a, b): maps to the same slot for (b, a).
static int pair_slot(int a, int b, int ntypes) noexcept {
    if (a > b) std::swap(a, b);
    return a * ntypes - a * (a - 1) / 2 + (b - a);
}

void build_neighbor_list(Configuration& cfg, double rcut,
                         std::span<const Potential> pots) {
    const int n = static_cast<int>(cfg.atoms.size());

    int ntypes = 0;
    for (const auto& atom : cfg.atoms)
        ntypes = std::max(ntypes, atom.type + 1);
    if (ntypes == 0) ntypes = 1;

    for (auto& atom : cfg.atoms)
        atom.neighbors.clear();

    const double rcut2 = rcut * rcut;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            const Vec3   d  = bc_min_image(cfg.bc, cfg.atoms[j].pos - cfg.atoms[i].pos);
            const double r2 = d.squaredNorm();
            if (r2 >= rcut2) continue;

            const int s = pair_slot(cfg.atoms[i].type, cfg.atoms[j].type, ntypes);
            cfg.atoms[i].neighbors.push_back(NeighborEntry{
                .neighbor = &cfg.atoms[j],
                .pot      = (s < static_cast<int>(pots.size())) ? &pots[s] : nullptr,
                .dist     = d,
            });
        }
    }
}

} // namespace potfit
