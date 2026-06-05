#pragma once

#include "forcesmith/core/types.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

// boost makima is kept only as an independent cross-check oracle for the
// hand-rolled slope/Hermite math, compiled in when FORCESMITH_SPLINE_VERIFY is
// defined. Release builds evaluate entirely from our own slopes (s_).
#ifdef FORCESMITH_SPLINE_VERIFY
#include <boost/assert.hpp>
#include <boost/math/interpolators/makima.hpp>
#include <cmath>
#endif

namespace forcesmith {

class SplinePotential {
public:
  SplinePotential(std::vector<double> x, std::vector<double> y);
  double eval(double r) const;
  double deriv(double r) const;
  constexpr std::pair<double, double> span() const {
    return {x_.front(), x_.back()};
  }

  constexpr void set_fixed(std::size_t i, bool f) { fixed_[i] = f; }
  constexpr bool is_fixed(std::size_t i) const { return fixed_[i]; }

  // Direct knot setter (interface parity with AnalyticParams::set_param,
  // required by the erased Potential). Global parameters only ever bind to
  // analytic _sc potentials, never to tabulated ones, so this path is unused in
  // practice.
  void set_param(std::size_t i, double v) { y_[i] = v; }

  constexpr std::size_t param_count() const {
    return static_cast<std::size_t>(std::ranges::count(fixed_, false));
  }
  void gather_params(Eigen::VectorXd &dst, std::size_t offset) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t offset);

  // ── Fit-time evaluation cache ─────────────────────────────────────────────
  // During a fit the query distances r are frozen (atom geometry is fixed)
  // while the knot values y_ move every iteration. prepare_site(r) memoizes the
  // geometry-fixed part of the evaluation — the bracketing knot interval and
  // the Hermite basis weights at r — and returns an index; eval_at/deriv_at
  // then evaluate against the CURRENT y_/slopes with NO binary search. This is
  // the spline analog of the ML descriptor cache. prepare_site MUST be called
  // single-threaded (it mutates the site table); eval_at/deriv_at are read-only
  // and safe under the parallel Jacobian. Returns a non-negative index.
  int prepare_site(double r) const;
  // Hot per-bond accessors: defined inline (FORCE_INLINE) so they fuse into the
  // force calculators' neighbor loops — the cached EvalSite + knot loads feed a
  // register-resident Hermite FMA chain with no call/spill across a TU boundary.
  // Bodies operate only on this object's own EvalSite/y_/s_ state.
  FORCE_INLINE double eval_at(int site) const {
    const EvalSite &st = sites_[static_cast<std::size_t>(site)];
    const std::size_t i = st.i;
    double v;
    if (st.kind == EvalSite::Kind::Linear) {
      v = y_[i] + st.off * st.inv_dx * (y_[i + 1] - y_[i]);
    } else {
      v = st.v0 * y_[i] + st.v1 * y_[i + 1] + st.vs0 * s_[i] +
          st.vs1 * s_[i + 1];
    }
#ifdef FORCESMITH_SPLINE_VERIFY
    const double ref = eval(st.r);
    BOOST_ASSERT(std::abs(v - ref) <= 1e-9 * (1.0 + std::abs(ref)));
#endif
    return v;
  }
  FORCE_INLINE double deriv_at(int site) const {
    const EvalSite &st = sites_[static_cast<std::size_t>(site)];
    const std::size_t i = st.i;
    double d;
    if (st.kind == EvalSite::Kind::Linear) {
      d = (y_[i + 1] - y_[i]) * st.inv_dx;
    } else {
      d = st.d0 * y_[i] + st.d1 * y_[i + 1] + st.ds0 * s_[i] +
          st.ds1 * s_[i + 1];
    }
#ifdef FORCESMITH_SPLINE_VERIFY
    const double ref = deriv(st.r);
    BOOST_ASSERT(std::abs(d - ref) <= 1e-9 * (1.0 + std::abs(ref)));
#endif
    return d;
  }
  // Fused value+derivative. Returns {eval_at(site), deriv_at(site)} but fetches
  // the cached EvalSite and the knot values/slopes once instead of twice; the
  // per-bond force loops want both. Each expression is copied verbatim from
  // eval_at()/deriv_at() (including the Linear branch's multiply order) so the
  // components are bit-identical to the separate calls.
  FORCE_INLINE std::pair<double, double> eval_and_deriv_at(int site) const {
    const EvalSite &st = sites_[static_cast<std::size_t>(site)];
    const std::size_t i = st.i;
    double v, d;
    if (st.kind == EvalSite::Kind::Linear) {
      v = y_[i] + st.off * st.inv_dx * (y_[i + 1] - y_[i]);
      d = (y_[i + 1] - y_[i]) * st.inv_dx;
    } else {
      const double yi = y_[i], yi1 = y_[i + 1], si = s_[i], si1 = s_[i + 1];
      v = st.v0 * yi + st.v1 * yi1 + st.vs0 * si + st.vs1 * si1;
      d = st.d0 * yi + st.d1 * yi1 + st.ds0 * si + st.ds1 * si1;
    }
#ifdef FORCESMITH_SPLINE_VERIFY
    const double vref = eval(st.r), dref = deriv(st.r);
    BOOST_ASSERT(std::abs(v - vref) <= 1e-9 * (1.0 + std::abs(vref)));
    BOOST_ASSERT(std::abs(d - dref) <= 1e-9 * (1.0 + std::abs(dref)));
#endif
    return {v, d};
  }
  // Uncached fused value+derivative: {eval(r), deriv(r)} sharing a single
  // interval search. Used on the non-primed path (analytic-free angular tables,
  // rescale, tests). Bit-identical to the two separate calls.
  std::pair<double, double> eval_and_deriv(double r) const;

  // Curvature (smoothness) regularization: one second-difference residual per
  // interior knot, active only when the potential has free knots. The knots are
  // on a uniform grid, so the plain second difference is proportional to the
  // discrete curvature.
  std::size_t curvature_count() const {
    return (param_count() > 0 && x_.size() >= 3) ? x_.size() - 2 : 0;
  }
  void write_curvature(Eigen::VectorXd &dst, std::size_t offset,
                       double weight) const;

private:
  // One cached evaluation site (one distinct fit-time distance r). Stores only
  // geometry (depends on x_ and r, never on y_), so it survives scatter_params.
  struct EvalSite {
    enum class Kind : std::uint8_t { Cubic, Linear } kind = Kind::Linear;
    std::size_t i = 0;  // interval: x_[i] <= r < x_[i+1]  (Linear: anchor knot)
    double r = 0.0;     // original query (fallback / cross-check)
    double off = 0.0;   // r - x_[i]
    double inv_dx = 0.0;// 1/(x_[i+1]-x_[i])
    // Cubic value basis:  y  = v0*y_[i] + v1*y_[i+1] + vs0*s_[i] + vs1*s_[i+1]
    double v0 = 0, v1 = 0, vs0 = 0, vs1 = 0;
    // Cubic deriv basis:  y' = d0*y_[i] + d1*y_[i+1] + ds0*s_[i] + ds1*s_[i+1]
    double d0 = 0, d1 = 0, ds0 = 0, ds1 = 0;
  };

  std::vector<double> x_, y_;
  std::vector<bool> fixed_;
  std::vector<double> s_; // makima knot slopes (size n when n>=4; else empty)
  mutable std::vector<EvalSite> sites_;           // indexed by the returned hint
  mutable std::unordered_map<double, int> site_of_r_;

  void recompute_slopes_();              // port of boost makima slope formula
  double hermite_eval_(std::size_t i, double r) const;  // exact boost formula
  double hermite_deriv_(std::size_t i, double r) const; // exact boost formula

#ifdef FORCESMITH_SPLINE_VERIFY
  using Makima = boost::math::interpolators::makima<std::vector<double>>;
  std::optional<Makima> interp_;
  void rebuild_interp_();
#endif
};

// Curvature customization-point overloads (found by ADL from the erased
// Potential); these make a tabulated potential participate in the Tikhonov
// smoothness regularization. See forcesmith/potentials/curvature.hpp.
FORCE_INLINE std::size_t curvature_count(const SplinePotential &p) {
  return p.curvature_count();
}
FORCE_INLINE void write_curvature(const SplinePotential &p,
                                  Eigen::VectorXd &dst, std::size_t off,
                                  double weight) {
  p.write_curvature(dst, off, weight);
}

} // namespace forcesmith
