#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/param.hpp"
#include <Eigen/Core>
#include <concepts>
#include <cstdint>
#include <utility>
#include <vector>

namespace forcesmith {

// A single optimizer parameter shared across several potentials (forcesmith's
// global parameters, e.g. a smooth-cutoff h declared once via makeapot -g).
// `value` is the one slot exposed to the optimizer; each Link names a potential
// slot it is broadcast into before every force evaluation. The linked slots are
// held fixed so the per-potential gather/scatter skip them — only `value`
// participates in the optimizer vector.
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
  constexpr void finalize_globals() noexcept {}
  constexpr void broadcast_globals() noexcept {}
  [[nodiscard]] constexpr bool globals_empty() const noexcept { return true; }
};

struct WithGlobals {
  std::vector<GlobalParam> globals;
  void set_globals(std::vector<GlobalParam> g) { globals = std::move(g); }
  [[nodiscard]] bool globals_empty() const noexcept { return globals.empty(); }
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

// forcesmith's DUMMY_WEIGHT (defines.h): weight on the EAM/ADP embedding F(ρ)
// out-of-range punishment residual.
inline constexpr double kDummyWeight = 100.0;

// True iff r lies in the radial table p's own cutoff range [rmin, rmax]. The
// neighbour list is built with the global max_cutoff() over all tables, so a
// table with a shorter cutoff would otherwise be fed neighbours past its last
// knot — where spline potentials linearly *extrapolate* (nonzero) instead of
// vanishing. Gate every radial table eval/deriv on this. Do NOT use it for the
// EAM/ADP embedding F(ρ) (argument ρ legitimately falls outside the table,
// where it is clamped instead) nor for the angular g(cosθ) table (cosθ ∈ [-1,1]
// is always within the table's own domain). The upper bound is inclusive to
// match C forcesmith's `r <= end[col]` and the inclusive neighbour-list cutoff.
template <typename Pot> bool in_range(const Pot &p, double r) {
  const auto [rmin, rmax] = p.span();
  return r >= rmin && r <= rmax;
}

// Cached + cutoff-gated radial-table evaluation for the force hot loops.
// A primed bond (site.cacheable()) was already cutoff-checked in the
// calculator's prepare() pass, so it evaluates straight through the spline cache
// with NO in_range/span() virtual call; an unprimed bond (none — rescale, tests,
// or out-of-range-at-prime) falls back to the in_range gate + direct eval,
// yielding 0 outside the table's own cutoff. Templated so it instantiates only
// where the full Potential type is visible.
template <typename Pot>
FORCE_INLINE double eval_gated(const Pot &p, SiteId site, double r) {
  return site.cacheable() ? p.eval_at(site)
                          : (in_range(p, r) ? p.eval(r) : 0.0);
}
template <typename Pot>
FORCE_INLINE double deriv_gated(const Pot &p, SiteId site, double r) {
  return site.cacheable() ? p.deriv_at(site)
                          : (in_range(p, r) ? p.deriv(r) : 0.0);
}

template <typename Pot>
FORCE_INLINE std::pair<double, double>
eval_deriv_gated(const Pot &p, SiteId site, double r) {
  if (site.cacheable()) {
    return p.eval_and_deriv_at(site);
  }
  return in_range(p, r) ? p.eval_and_deriv(r) : std::pair{0.0, 0.0};
}

template <typename Range>
void gather_range(const Range &range, Eigen::VectorXd &dst, std::size_t &off) {
  for (const auto &p : range) {
    p.gather_params(dst, off);
    off += p.param_count();
  }
}

template <typename Range>
void scatter_range(Range &range, const Eigen::VectorXd &src, std::size_t &off) {
  for (auto &p : range) {
    p.scatter_params(src, off);
    off += p.param_count();
  }
}

// Per-free-param box constraints, in the same order as gather_range so the
// bound vectors align element-for-element with the gathered parameter vector.
template <typename Range>
void gather_bounds_range(const Range &range, Eigen::VectorXd &lo,
                         Eigen::VectorXd &hi, std::size_t &off) {
  for (const auto &p : range) {
    p.gather_bounds(lo, hi, off);
    off += p.param_count();
  }
}

template <typename Range>
std::size_t smoothness_count_range(const Range &range) {
  return std::transform_reduce(
      range.begin(), range.end(), std::size_t{0}, std::plus<>{},
      [](const auto &p) { return p.smoothness_count(); });
}

template <typename Range>
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

template <typename T>
concept ForceCalculatorModel =
    requires(T calc, Configuration &cfg, Eigen::VectorXd &v, std::size_t off) {
      { calc.eval_forces(cfg) } -> std::same_as<void>;
      { calc.param_count() } -> std::same_as<std::size_t>;
      { calc.gather_params(v, off) } -> std::same_as<void>;
      { calc.scatter_params(v, off) } -> std::same_as<void>;
      { calc.gather_bounds(v, v, off) } -> std::same_as<void>;
      { calc.max_cutoff() } -> std::same_as<double>;
    };

template <typename Derived> struct ForceCalculatorBase {
  std::size_t ntypes = 1;
  std::uint64_t conf_index = 0;
};

} // namespace forcesmith
