#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/radial_potential.hpp"
#include "forcesmith/force/potential_table.hpp"

#include <span>

namespace forcesmith {

void build_neighbor_list(Configuration &cfg, double rcut);

void build_neighbor_list(Configuration &cfg, double rcut,
                         const RadialPotentialPair &pots);

// Parallel over disjoint configs, bit-identical to a serial sweep; call inside
// shared_arena().
void build_all_neighbor_lists(std::span<Configuration> configs, double rcut);

} // namespace forcesmith
