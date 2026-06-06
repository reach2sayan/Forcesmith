#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/fit_params.hpp" // CFittable
#include "forcesmith/core/param.hpp"
#include <Eigen/Core>
#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>
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

// Descriptor-cache fit fast-path surface, mixed in like NoGlobals/WithGlobals.
// These methods are part of every calculator's contract so the erased
// ForceCalculator can call the whole interface unconditionally (no `if
// constexpr` probing in its Model<T> adapter).
//
// NoCache is what every analytic calculator wants: has_cache() == false, the
// indexed eval just recomputes from scratch (CRTP forward to the plain
// eval_forces), and the cached residual/Jacobian hooks are unreachable. Every
// calculator inherits it through ForceCalculatorBase. The one cache owner is
// MLBase, which overrides this whole surface with a real (runtime) descriptor
// cache — so there is no separate compile-time "has cache" mixin to opt into.
template <class Derived> struct NoCache {
  [[nodiscard]] constexpr bool has_cache() const noexcept { return false; }
  [[nodiscard]] constexpr bool has_param_jacobian() const noexcept {
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
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

// forcesmith's DUMMY_WEIGHT (defines.h): weight on the EAM/ADP embedding F(ρ)
// out-of-range punishment residual.
inline constexpr double kDummyWeight = 100.0;

template <typename Pot> FORCE_INLINE bool in_range(const Pot &p, double r) {
  const auto [rmin, rmax] = p.span();
  return r >= rmin && r <= rmax;
}

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
FORCE_INLINE std::pair<double, double> eval_deriv_gated(const Pot &p,
                                                        SiteId site, double r) {
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

// The leaf-side contract for a type stored in a ForceCalculator. Subsumes
// CFittable (every calculator supplies the full optimizer param surface,
// gather_bounds included) and adds the force-evaluation surface. The remaining
// fit fast-path methods (eval_forces(cfg,idx), prepare, eval_cached…) come from
// the NoCache mixin in ForceCalculatorBase, so they are always present and need
// not be re-listed here.
template <typename T>
concept CForceCalculator =
    CFittable<T> && requires(T calc, Configuration &cfg) {
      { calc.eval_forces(cfg) } -> std::same_as<void>;
      { calc.max_cutoff() } -> std::same_as<double>;
      { calc.ntypes } -> std::convertible_to<std::size_t>;
    };

// The shared identity + capability base every concrete force calculator derives
// from. It carries the type count / config index, always folds in the no-cache
// fit fast-path surface (NoCache — overridden by ML, which is the only cache
// owner), and selects the globals axis (NoGlobals|WithGlobals). A calculator
// names the globals choice with a fluent clause instead of listing extra bases:
//
//   struct Tersoff : ForceCalculatorBase<Tersoff> { ... };               // NoGlobals
//   struct EAM     : ForceCalculatorBase<EAM>::with_globals<> { ... };   // WithGlobals
//   struct Pair    : ForceCalculatorBase<Pair>::with_globals<> { ... };
//
// Omitting the clause keeps the NoGlobals default. A calculator that defines its
// own eval_forces(cfg) still re-exposes the inherited indexed overload with
// `using Base::eval_forces;` (C++ name hiding) — see the concrete calculators.
template <typename Derived, bool HasGlobals = false>
struct ForceCalculatorBase
    : std::conditional_t<HasGlobals, WithGlobals, NoGlobals>,
      NoCache<Derived> {
  std::size_t ntypes = 1;
  std::uint64_t conf_index = 0;

  // Flip the globals axis on. `On` defaults to true so the call site reads
  // `with_globals<>`.
  template <bool On = true>
  using with_globals = ForceCalculatorBase<Derived, On>;
};

} // namespace forcesmith
