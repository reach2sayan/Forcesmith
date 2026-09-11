#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/core/fit_params.hpp" // CFittable
#include "forcesmith/core/param.hpp"
#include "forcesmith/core/radial_potential.hpp" // CRadialPotential
#include <Eigen/Core>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace forcesmith {

struct GlobalParam {
  Param value;
  struct Link {
    enum class LinkRegion {
      PAIR,
      DENSITY,
      EMBEDDING
    } region;          // which calculator sub-table the linked slot lives in
    std::size_t index; // flat position within that table (gather/scatter order)
    std::size_t param; // parameter slot inside that potential
  };
  std::vector<Link> links;
};

struct NoGlobals {
  constexpr void set_globals(const std::vector<GlobalParam> &) noexcept {}
  [[nodiscard]] constexpr bool globals_empty() const noexcept { return true; }
};

struct WithGlobals {
  std::vector<GlobalParam> globals;
  void set_globals(std::vector<GlobalParam> g) { globals = std::move(g); }
  [[nodiscard]] bool globals_empty() const noexcept { return globals.empty(); }
};

template <class Derived> struct NoCache {
  [[nodiscard]] constexpr bool has_cache() const noexcept { return false; }
  [[nodiscard]] constexpr bool has_param_jacobian() const noexcept {
    return false;
  }
  [[nodiscard]] constexpr bool has_standardization() const noexcept {
    return false;
  }
  void eval_forces(Configuration &cfg, std::size_t /*cache_index*/) const {
    static_cast<const Derived &>(*this).eval_forces(cfg);
  }
  void prepare(std::span<Configuration> /*configs*/) const {}
  void eval_cached(std::size_t /*cache_index*/, std::span<Vec3> /*forces*/,
                   double & /*energy*/, SymTens & /*stress*/) const {
    std::unreachable(); // gated by has_cache()
  }
  void eval_cached_jacobian(std::size_t /*cache_index*/, int /*row0*/,
                            const std::vector<int> & /*col_off*/,
                            double /*energy_weight*/, double /*stress_weight*/,
                            Eigen::MatrixXd & /*fjac*/) const {
    std::unreachable(); // gated by has_param_jacobian()
  }
  [[nodiscard]] std::vector<std::size_t> head_param_counts() const {
    return {};
  }

  [[nodiscard]] bool has_analytic_jacobian() const noexcept { return false; }
  void write_param_jacobian(Configuration & /*cfg*/, int /*row0*/,
                            double /*energy_weight*/, double /*stress_weight*/,
                            Eigen::MatrixXd & /*fjac*/) const {
    std::unreachable(); // gated by has_analytic_jacobian()
  }
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

inline constexpr double kDummyWeight = 100.0;

template <class P>
concept CEvaluable = requires(const P &p, double r, SiteId site) {
  { p.span() } -> std::convertible_to<std::pair<double, double>>;
  { p.eval(r) } -> std::convertible_to<double>;
  { p.deriv(r) } -> std::convertible_to<double>;
};

template <CEvaluable Pot>
FORCE_INLINE bool in_range(const Pot &p, double r) {
  const auto [rmin, rmax] = p.span();
  return r >= rmin && r <= rmax;
}

template <CEvaluable Pot>
FORCE_INLINE double eval_gated(const Pot &p, SiteId site, double r) {
  return site.cacheable() ? p.eval_at(site)
                          : (in_range(p, r) ? p.eval(r) : 0.0);
}
template <CEvaluable Pot>
FORCE_INLINE double deriv_gated(const Pot &p, SiteId site, double r) {
  return site.cacheable() ? p.deriv_at(site)
                          : (in_range(p, r) ? p.deriv(r) : 0.0);
}

template <CEvaluable Pot>
FORCE_INLINE std::pair<double, double> eval_deriv_gated(const Pot &p,
                                                        SiteId site, double r) {
  if (site.cacheable()) {
    return p.eval_and_deriv_at(site);
  }
  return in_range(p, r) ? p.eval_and_deriv(r) : std::pair{0.0, 0.0};
}

template <class R>
concept CFittableRange =
    std::ranges::input_range<R> && CFittable<std::ranges::range_value_t<R>>;

template <CFittableRange Range>
void gather_range(const Range &range, Eigen::VectorXd &dst, std::size_t &off) {
  for (const auto &p : range) {
    p.gather_params(dst, off);
    off += p.param_count();
  }
}

template <CFittableRange Range>
void scatter_range(Range &range, const Eigen::VectorXd &src, std::size_t &off) {
  for (auto &p : range) {
    p.scatter_params(src, off);
    off += p.param_count();
  }
}

template <CFittableRange Range>
void gather_bounds_range(const Range &range, Eigen::VectorXd &lo,
                         Eigen::VectorXd &hi, std::size_t &off) {
  for (const auto &p : range) {
    p.gather_bounds(lo, hi, off);
    off += p.param_count();
  }
}

template <std::ranges::input_range Range>
std::size_t smoothness_count_range(const Range &range) {
  return std::transform_reduce(
      range.begin(), range.end(), std::size_t{0}, std::plus<>{},
      [](const auto &p) { return p.smoothness_count(); });
}

template <std::ranges::input_range Range>
void write_smoothness_range(const Range &range, Eigen::VectorXd &dst,
                            std::size_t &off, double weight) {
  for (const auto &p : range) {
    p.write_smoothness(dst, off, weight);
    off += p.smoothness_count();
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// Table-driven optimizer surface. ORDER IS PART OF THE CONTRACT: tables in
// declaration order, then globals, identically in gather/scatter/bounds.
template <typename Derived, bool HasGlobals> struct TableParams {
  std::size_t param_count() const {
    std::size_t n = 0;
    for_each_table(self(), [&](const auto &t, std::string_view) {
      n += std::transform_reduce(t.begin(), t.end(), std::size_t{0},
                                 std::plus<>{},
                                 [](const auto &p) { return p.param_count(); });
    });
    return n + free_globals();
  }

  void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    for_each_table(self(), [&](const auto &t, std::string_view) {
      gather_range(t, dst, off);
    });
    if constexpr (HasGlobals) {
      detail::gather_params_impl(global_values(), dst, off);
    }
  }

  void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    for_each_table(
        mut(), [&](auto &t, std::string_view) { scatter_range(t, src, off); });
    if constexpr (HasGlobals) {
      detail::scatter_params_impl(mutable_global_values(), src, off);
      mut().broadcast_globals();
    }
  }

  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    for_each_table(self(), [&](const auto &t, std::string_view) {
      gather_bounds_range(t, lo, hi, off);
    });
    if constexpr (HasGlobals) {
      detail::gather_bounds_impl(global_values(), lo, hi, off);
    }
  }

  double max_cutoff() const {
    double m = 0.0;
    for_each_table_field(self(), [&](const auto &t, const auto &field) {
      if (!field.radial) {
        return;
      }
      for (const auto &p : t) {
        m = std::max(m, p.span().second);
      }
    });
    return m;
  }

  void broadcast_globals() {
    if constexpr (HasGlobals) {
      each_link([](auto &pot, const auto &lk, double v) {
        pot.set_param(lk.param, v);
      });
    }
  }

  void finalize_globals() {
    if constexpr (HasGlobals) {
      each_link([](auto &pot, const auto &lk, double) {
        pot.set_fixed(lk.param, true);
      });
      broadcast_globals();
    }
  }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }
  Derived &mut() { return static_cast<Derived &>(*this); }

  std::size_t free_globals() const {
    if constexpr (HasGlobals) {
      return static_cast<std::size_t>(std::ranges::count_if(
          self().globals, [](const GlobalParam &g) { return !g.value.fixed; }));
    } else {
      return 0;
    }
  }
  auto global_values() const {
    return self().globals |
           std::views::transform(
               [](const GlobalParam &g) -> const Param & { return g.value; });
  }
  auto mutable_global_values() {
    return mut().globals | std::views::transform([](GlobalParam &g) -> Param & {
             return g.value;
           });
  }

  template <class Act>
  void each_link(Act act)
    requires HasGlobals
  {
    using enum GlobalParam::Link::LinkRegion;
    for (const GlobalParam &g : mut().globals) {
      for (const auto &lk : g.links) {
        const std::string_view want = lk.region == DENSITY     ? "density"
                                      : lk.region == EMBEDDING ? "embedding"
                                                               : "pair";
        for_each_table(mut(), [&](auto &t, std::string_view name) {
          const bool match = name == want || (name.empty() && want == "pair");
          if (!match) {
            return;
          }
          act(*std::next(t.begin(), static_cast<std::ptrdiff_t>(lk.index)), lk,
              g.value.value);
        });
      }
    }
  }
};

template <typename T>
concept CForceCalculator =
    CFittable<T> && requires(T calc, Configuration &cfg) {
      { calc.eval_forces(cfg) } -> std::same_as<void>;
      { calc.max_cutoff() } -> std::same_as<double>;
      { calc.ntypes } -> std::convertible_to<std::size_t>;
    };

template <typename Derived, bool HasGlobals = false>
struct ForceCalculatorBase
    : std::conditional_t<HasGlobals, WithGlobals, NoGlobals>,
      NoCache<Derived>,
      TableParams<Derived, HasGlobals> {
  std::size_t ntypes = 1;
  std::uint64_t conf_index = 0;

  template <bool On = true>
  using with_globals = ForceCalculatorBase<Derived, On>;
};

} // namespace forcesmith
