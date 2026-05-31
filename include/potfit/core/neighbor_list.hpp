#pragma once

// Step 7: O(N²) build, cell-list upgrade later.

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/force/potential_table.hpp"

namespace potfit {

// Build the full neighbor list for cfg.
// Geometry-only overload: all NeighborEntry::pot are left nullptr.
void build_neighbor_list(Configuration &cfg, double rcut);

// Potential-assigning overload: each NeighborEntry::pot is resolved from pots
// using the (ai.type, aj.type) pair index.  Entries whose type exceeds
// pots.ntypes() get pot = nullptr.  pots must outlive the Configuration's
// neighbor list (NeighborEntry::pot stores raw pointers into pots).
void build_neighbor_list(Configuration &cfg, double rcut,
                         const PotentialPair &pots);

} // namespace potfit
