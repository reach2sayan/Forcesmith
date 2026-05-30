#pragma once

// Step 7: O(N²) build, cell-list upgrade later.

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"

#include <span>

namespace potfit {

// Build the full neighbor list for cfg.
// If pots is provided, each NeighborEntry::pot is resolved to the matching
// element; entries with no potential (out-of-range slot) get pot = nullptr.
// When called without pots (default {}), neighbor and pot are left nullptr —
// useful for geometry-only tests or when potentials are attached separately.
// The span must outlive the Configuration's neighbor list (NeighborEntry::pot
// stores raw pointers into it).
void build_neighbor_list(Configuration& cfg, double rcut,
                         std::span<const Potential> pots = {});

}  // namespace potfit
