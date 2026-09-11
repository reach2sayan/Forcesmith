#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/neighbor_list.hpp" // build_neighbor_list, bc_volume
#include "forcesmith/core/types.hpp"
#include "forcesmith/events/signals.hpp"

#include <concepts>
#include <cstdint>

namespace forcesmith::force {

template <std::invocable Kernel>
FORCE_INLINE void with_eval_scope(Configuration &cfg, double cutoff,
                                  std::uint64_t conf_index, Kernel &&kernel) {
  build_neighbor_list(cfg, cutoff);

  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  cfg.calc_limit = 0.0;
  for (Atom &a : cfg.atoms) {
    a.ZeroForce();
    a.ZeroScratch();
  }

  kernel();

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{
      .conf_index = conf_index, .rms_force = force_rms(cfg), .cfg = cfg});
}

} // namespace forcesmith::force
