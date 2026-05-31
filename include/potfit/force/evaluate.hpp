#pragma once

#include "potfit/core/atom.hpp"               // Configuration
#include "potfit/core/types.hpp"              // Vec3, SymTens
#include "potfit/force/force_calculator.hpp" // ForceCalculator variant

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace potfit::force {

// Result of evaluating a potential against a single configuration.
struct EvalResult {
  double energy = 0.0;              // calc_energy
  std::vector<Vec3> forces;         // per-atom, indexed parallel to cfg.atoms
  SymTens stress = SymTens::Zero(); // calc_stress (virial / volume)
  double limit = 0.0;               // EAM/ADP out-of-range penalty (0 for others)
};

EvalResult evaluate(const ForceCalculator &calc, const Configuration &cfg);

// --- Tuple-like (std::get) interface ---------------------------------------
// Field order: <0> energy, <1> forces, <2> stress, <3> limit. Enables
// structured bindings (`auto &[e, f, s, l] = r;`) and `get<I>(r)` — found by
// ADL, or via `std::get` after a `using std::get;`.
template <std::size_t I> constexpr decltype(auto) get(EvalResult &r) noexcept {
  static_assert(I < 4, "EvalResult has 4 fields");
  if constexpr (I == 0)
    return (r.energy);
  else if constexpr (I == 1)
    return (r.forces);
  else if constexpr (I == 2)
    return (r.stress);
  else
    return (r.limit);
}

template <std::size_t I>
constexpr decltype(auto) get(const EvalResult &r) noexcept {
  static_assert(I < 4, "EvalResult has 4 fields");
  if constexpr (I == 0)
    return (r.energy);
  else if constexpr (I == 1)
    return (r.forces);
  else if constexpr (I == 2)
    return (r.stress);
  else
    return (r.limit);
}

template <std::size_t I> constexpr decltype(auto) get(EvalResult &&r) noexcept {
  return std::move(get<I>(r));
}

} // namespace potfit::force

// std::tuple_size / std::tuple_element specializations make EvalResult a
// tuple-like type for structured bindings and the std::get protocol.
template <>
struct std::tuple_size<potfit::force::EvalResult>
    : std::integral_constant<std::size_t, 4> {};

template <std::size_t I>
struct std::tuple_element<I, potfit::force::EvalResult> {
  using type = std::remove_reference_t<decltype(potfit::force::get<I>(
      std::declval<potfit::force::EvalResult &>()))>;
};
