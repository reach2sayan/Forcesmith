#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/force/force_calculator.hpp"

#include <cstdint>
#include <vector>

namespace potfit {

// EAM potential layout for ntypes element types:
//   pair_pots  — ntypes*(ntypes+1)/2 pair potentials φ_{ij}(r)
//   rho_pots   — ntypes density functions g_i(r):
//                 g_i(r) = density contributed by an atom of type i to a
//                 neighbor
//   F_pots     — ntypes embedding functions F_i(ρ):
//                 F_i(ρ) = embedding energy of a type-i atom at density ρ
//
// Slot mapping for pair potentials (matches neighbor_list.cpp pair_slot):
//   pair_pots[pair_slot(ti, tj, ntypes)] = φ_{ti,tj}
struct EAMForceCalculator {
  int ntypes = 1;
  std::vector<Potential> pair_pots;
  std::vector<Potential> rho_pots;
  std::vector<Potential> F_pots;
  std::uint64_t conf_index = 0;

  void eval_forces(Configuration &cfg) const;

private:
  static int pair_slot(int a, int b, int n) noexcept {
    if (a > b)
      std::swap(a, b);
    return a * n - a * (a - 1) / 2 + (b - a);
  }
};

static_assert(ForceCalculator<EAMForceCalculator>);

} // namespace potfit
