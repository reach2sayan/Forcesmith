#pragma once

// Step 7: O(N²) build, cell-list upgrade later.

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/potential_base.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <span>

namespace forcesmith {

// Build the full neighbour list for cfg.
// Geometry-only overload: all NeighborEntry::pot are left nullptr.
void build_neighbor_list(Configuration &cfg, double rcut);

// Potential-assigning overload: each NeighborEntry::pot is resolved from pots
// using the (ai.type, aj.type) pair index.  Entries whose type exceeds
// pots.ntypes() get pot = nullptr.  pots must outlive the Configuration's
// neighbour list (NeighborEntry::pot stores raw pointers into pots).
void build_neighbor_list(Configuration &cfg, double rcut,
                         const PotentialPair &pots);

// Build every config's neighbour list in parallel (geometry-only overload).
// Each config is disjoint — it writes only its own atoms and nl_* cache fields —
// so this is embarrassingly parallel and bit-identical to a serial sweep. Must
// be called from inside shared_arena() (every ForceCalculator::prepare() is).
void build_all_neighbor_lists(std::span<Configuration> configs, double rcut);

} // namespace forcesmith
