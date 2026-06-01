#pragma once

#include "potfit/core/atom.hpp"

#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index_container.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

// Auxiliary grouping/ordering index over Configuration objects.
//
// The owning store remains `std::vector<Configuration>` — this container holds
// non-owning handles (ConfigRef) so the optimizer's contiguous-span /
// pointer-arithmetic / row_offset machinery
// (src/optimization/potfit_functor.cpp) stays untouched. Only IMMUTABLE
// properties are indexed: the set of distinct atom types present
// ("composition"), reference `energy`, and `weight`. The
// recomputed-every-evaluation fields (calc_energy, calc_stress) are
// deliberately NOT keys — sort on demand if needed.
//
// INVARIANT: the std::vector<Configuration> this index is built from must not
// be resized, reordered, or moved after build_config_index(). It is built once
// after load and thereafter only mutated element-wise — the same invariant the
// optimizer's `&cfg - configs.data()` already relies on.
namespace potfit::config_index {

// Sorted, distinct atom-type ids present in a configuration.
using CompositionKey = std::vector<std::size_t>;

struct ConfigRef {
  const Configuration *cfg = nullptr; // non-owning, into the configs vector
  std::size_t order = 0;              // canonical index == residual block id
  CompositionKey composition;         // sorted distinct atom types
  std::uint64_t comp_mask = 0; // bit t set if type t present; 0 if type>=64
  [[nodiscard]] double energy() const { return cfg->ref.energy; }
  [[nodiscard]] double weight() const { return cfg->weight; }
};

// Sorted, de-duplicated list of the distinct atom types in `c`.
[[nodiscard]] inline CompositionKey composition_of(const Configuration &c) {
  CompositionKey types;
  types.reserve(c.atoms.size());
  for (const Atom &a : c.atoms) {
    types.push_back(a.type);
  }
  std::sort(types.begin(), types.end());
  types.erase(std::unique(types.begin(), types.end()), types.end());
  return types;
}

// Bitmask with bit `t` set when type `t` is present. Returns 0 as a sentinel if
// any type id is >= 64 (caller must fall back to the CompositionKey vector
// path).
[[nodiscard]] inline std::uint64_t composition_mask(const Configuration &c) {
  std::uint64_t mask = 0;
  for (const Atom &a : c.atoms) {
    if (a.type >= 64) {
      return 0;
    }
    mask |= (std::uint64_t{1} << a.type);
  }
  return mask;
}

namespace detail {
struct by_order {};
struct by_composition {};
struct by_energy {};
struct by_weight {};
} // namespace detail

namespace bmi = boost::multi_index;

using ConfigIndex = bmi::multi_index_container<
    ConfigRef,
    bmi::indexed_by<
        // canonical load order — O(1) indexed access, residual block id
        bmi::random_access<bmi::tag<detail::by_order>>,
        // group by element set
        bmi::ordered_non_unique<
            bmi::tag<detail::by_composition>,
            bmi::member<ConfigRef, CompositionKey, &ConfigRef::composition>>,
        // band/range queries on immutable reference energy
        bmi::ordered_non_unique<
            bmi::tag<detail::by_energy>,
            bmi::const_mem_fun<ConfigRef, double, &ConfigRef::energy>>,
        // weight banding
        bmi::ordered_non_unique<
            bmi::tag<detail::by_weight>,
            bmi::const_mem_fun<ConfigRef, double, &ConfigRef::weight>>>>;

// Build the index from the owning configuration store. `order` is the position
// in `configs`, which equals the residual block id used by the optimizer.
[[nodiscard]] inline ConfigIndex
build_config_index(std::span<const Configuration> configs) {
  ConfigIndex idx;
  auto &ordered = idx.get<detail::by_order>();
  for (std::size_t i = 0; i < configs.size(); ++i) {
    const Configuration &c = configs[i];
    ordered.push_back(ConfigRef{.cfg = &c,
                                .order = i,
                                .composition = composition_of(c),
                                .comp_mask = composition_mask(c)});
  }
  return idx;
}

namespace detail {
// Gather matching handles into a plain vector and restore canonical order, so
// callers get a contiguous, deterministic subset (never iterate a node-based
// index under std::execution::par).
template <class It>
[[nodiscard]] std::vector<Configuration *> gather(It first, It last) {
  std::vector<ConfigRef> refs(first, last);
  std::sort(
      refs.begin(), refs.end(),
      [](const ConfigRef &a, const ConfigRef &b) { return a.order < b.order; });
  std::vector<Configuration *> out;
  out.reserve(refs.size());
  for (const ConfigRef &r : refs) {
    out.push_back(const_cast<Configuration *>(r.cfg));
  }
  return out;
}
} // namespace detail

// All configs whose distinct-type set exactly equals `comp`.
[[nodiscard]] inline std::vector<Configuration *>
configs_with_composition(const ConfigIndex &idx, const CompositionKey &comp) {
  const auto &by_comp = idx.get<detail::by_composition>();
  const auto [lo, hi] = by_comp.equal_range(comp);
  return detail::gather(lo, hi);
}

// All configs that contain at least one atom of type `type`.
[[nodiscard]] inline std::vector<Configuration *>
configs_containing_element(const ConfigIndex &idx, std::size_t type) {
  const auto &ordered = idx.get<detail::by_order>();
  std::vector<ConfigRef> hits;
  for (const ConfigRef &r : ordered) {
    const bool present = (type < 64 && r.comp_mask != 0)
                             ? ((r.comp_mask & (std::uint64_t{1} << type)) != 0)
                             : std::binary_search(r.composition.begin(),
                                                  r.composition.end(), type);
    if (present) {
      hits.push_back(r);
    }
  }
  return detail::gather(hits.begin(), hits.end());
}

// All configs whose reference weight lies in [lo, hi].
[[nodiscard]] inline std::vector<Configuration *>
configs_in_weight_band(const ConfigIndex &idx, double lo, double hi) {
  const auto &by_w = idx.get<detail::by_weight>();
  return detail::gather(by_w.lower_bound(lo), by_w.upper_bound(hi));
}

// All configs whose reference energy lies in [lo, hi].
[[nodiscard]] inline std::vector<Configuration *>
configs_in_energy_band(const ConfigIndex &idx, double lo, double hi) {
  const auto &by_e = idx.get<detail::by_energy>();
  return detail::gather(by_e.lower_bound(lo), by_e.upper_bound(hi));
}

} // namespace potfit::config_index
