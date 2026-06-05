#pragma once

#include "forcesmith/core/types.hpp"
#include "forcesmith/core/atom.hpp"

#include <boost/multi_index/global_fun.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index_container.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace forcesmith::config_index {

// Sorted, distinct atom-type ids present in a configuration.
using CompositionKey = std::vector<std::size_t>;

class ConfigRef {
public:
  ConfigRef(const Configuration *cfg, std::size_t order,
            CompositionKey composition, std::uint64_t comp_mask,
            std::string_view name)
      : cfg_(cfg), order_(order), composition_(std::move(composition)),
        comp_mask_(comp_mask), name_(name) {}

  [[nodiscard]] const Configuration *cfg() const { return cfg_; }
  [[nodiscard]] std::size_t order() const { return order_; }
  [[nodiscard]] const CompositionKey &composition() const {
    return composition_;
  }
  [[nodiscard]] std::uint64_t comp_mask() const { return comp_mask_; }
  [[nodiscard]] std::string_view name() const { return name_; }
  [[nodiscard]] double energy() const { return cfg_->ref.energy; }
  [[nodiscard]] double weight() const { return cfg_->weight; }

private:
  const Configuration *cfg_ = nullptr; // non-owning, into the configs vector
  std::size_t order_ = 0;              // canonical index == residual block id
  CompositionKey composition_;         // sorted distinct atom types
  std::uint64_t comp_mask_ = 0; // bit t set if type t present; 0 if type>=64
  std::string_view name_;       // view into cfg->name (owned); unique key
};

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
struct by_name {};
} // namespace detail

namespace bmi = boost::multi_index;

namespace detail {
[[nodiscard]] inline const CompositionKey &ref_composition(const ConfigRef &r) {
  return r.composition();
}
} // namespace detail

using ConfigIndex = bmi::multi_index_container<
    ConfigRef,
    bmi::indexed_by<
        bmi::random_access<bmi::tag<detail::by_order>>,
        // group by element set
        bmi::ordered_non_unique<
            bmi::tag<detail::by_composition>,
            bmi::global_fun<const ConfigRef &, const CompositionKey &,
                            &detail::ref_composition>>,
        // band/range queries on immutable reference energy
        bmi::ordered_non_unique<
            bmi::tag<detail::by_energy>,
            bmi::const_mem_fun<ConfigRef, double, &ConfigRef::energy>>,
        // weight banding
        bmi::ordered_non_unique<
            bmi::tag<detail::by_weight>,
            bmi::const_mem_fun<ConfigRef, double, &ConfigRef::weight>>,
        // unique human-facing identifier; string_view key into the owned name
        bmi::ordered_unique<bmi::tag<detail::by_name>,
                            bmi::const_mem_fun<ConfigRef, std::string_view,
                                               &ConfigRef::name>>>>;

// Build the index from the owning configuration store. `order` is the position
// in `configs`, which equals the residual block id used by the optimizer.
[[nodiscard]] FORCE_INLINE ConfigIndex
build_config_index(std::span<const Configuration> configs) {
  ConfigIndex idx;
  auto &ordered = idx.get<detail::by_order>();
  for (std::size_t i = 0; i < configs.size(); ++i) {
    const Configuration &c = configs[i];
    ordered.push_back(
        ConfigRef{&c, i, composition_of(c), composition_mask(c), c.name});
  }
  return idx;
}

namespace detail {
template <class It>
[[nodiscard]] std::vector<Configuration *> gather(It first, It last) {
  std::vector<ConfigRef> refs(first, last);
  std::ranges::sort(refs, {}, &ConfigRef::order);
  auto out_view = refs | std::views::transform([](const ConfigRef &r) {
                    return const_cast<Configuration *>(r.cfg());
                  });

  std::vector<Configuration *> out(out_view.begin(), out_view.end());
  return out;
}
} // namespace detail

// All configs whose distinct-type set exactly equals `comp`.
[[nodiscard]] FORCE_INLINE std::vector<Configuration *>
configs_with_composition(const ConfigIndex &idx, const CompositionKey &comp) {
  const auto &by_comp = idx.get<detail::by_composition>();
  const auto [lo, hi] = by_comp.equal_range(comp);
  return detail::gather(lo, hi);
}

// All configs that contain at least one atom of type `type`.
[[nodiscard]] FORCE_INLINE std::vector<Configuration *>
configs_containing_element(const ConfigIndex &idx, std::size_t type) {
  const auto &ordered = idx.get<detail::by_order>();
  std::vector<ConfigRef> hits;
  for (const ConfigRef &r : ordered) {
    const bool present =
        (type < 64 && r.comp_mask() != 0)
            ? ((r.comp_mask() & (std::uint64_t{1} << type)) != 0)
            : std::ranges::binary_search(r.composition(), type);
    if (present) {
      hits.push_back(r);
    }
  }
  return detail::gather(hits.begin(), hits.end());
}

// All configs whose reference weight lies in [lo, hi].
[[nodiscard]] FORCE_INLINE std::vector<Configuration *>
configs_in_weight_band(const ConfigIndex &idx, double lo, double hi) {
  const auto &by_w = idx.get<detail::by_weight>();
  return detail::gather(by_w.lower_bound(lo), by_w.upper_bound(hi));
}

// All configs whose reference energy lies in [lo, hi].
[[nodiscard]] FORCE_INLINE std::vector<Configuration *>
configs_in_energy_band(const ConfigIndex &idx, double lo, double hi) {
  const auto &by_e = idx.get<detail::by_energy>();
  return detail::gather(by_e.lower_bound(lo), by_e.upper_bound(hi));
}

// The config with the given unique name, or nullptr if none matches.
[[nodiscard]] FORCE_INLINE Configuration *config_by_name(const ConfigIndex &idx,
                                                   std::string_view name) {
  const auto &by_name = idx.get<detail::by_name>();
  const auto it = by_name.find(name);
  return it == by_name.end() ? nullptr : const_cast<Configuration *>(it->cfg());
}

} // namespace forcesmith::config_index
