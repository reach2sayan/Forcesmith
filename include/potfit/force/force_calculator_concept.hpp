#pragma once

#include "potfit/core/atom.hpp"
#include <Eigen/Core>
#include <concepts>
#include <cstdint>

namespace potfit {

// NOTE: GCC 13's -Wdangling-reference is a false positive in these loops. The
// ranges are always lvalue member containers and their iterators' operator*
// returns a real T& into the underlying storage, so the loop reference never
// dangles. See GCC PR#107532. Suppressed locally rather than weakening the
// warning project-wide.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

// True iff r lies in the radial table p's own cutoff range [rmin, rmax). The
// neighbor list is built with the global max_cutoff() over all tables, so a
// table with a shorter cutoff would otherwise be fed neighbors past its last
// knot — where spline potentials linearly *extrapolate* (nonzero) instead of
// vanishing. Gate every radial table eval/deriv on this. Do NOT use it for the
// EAM/ADP embedding F(ρ) (argument ρ legitimately falls outside the table, where
// extrapolation is intended) nor for the angular g(cosθ) table (cosθ ∈ [-1,1]
// is always within the table's own domain).
template <typename Pot> bool in_range(const Pot &p, double r) {
  const auto [rmin, rmax] = p.span();
  return r >= rmin && r < rmax;
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

// Total number of curvature (smoothness) residuals contributed by a range of
// potentials, and writing those residuals — mirrors gather_range/scatter_range.
template <typename Range>
std::size_t smoothness_count_range(const Range &range) {
  std::size_t n = 0;
  for (const auto &p : range) {
    n += p.smoothness_count();
  }
  return n;
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
        { calc.eval_forces(cfg) }       -> std::same_as<void>;
        { calc.param_count() }          -> std::same_as<std::size_t>;
        { calc.gather_params(v, off) }  -> std::same_as<void>;
        { calc.scatter_params(v, off) } -> std::same_as<void>;
        { calc.max_cutoff() }           -> std::same_as<double>;
    };

template <typename Derived> struct ForceCalculatorBase {
    std::size_t ntypes = 1;
    std::uint64_t conf_index = 0;
};

} // namespace potfit
