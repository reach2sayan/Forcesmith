#include "forcesmith/potentials/spline.hpp"

#include "forcesmith/core/fit_params.hpp"

#include <algorithm>
#include <boost/assert.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include <ranges>

namespace forcesmith {

namespace {

// Locate the interval x[i] <= r < x[i+1], clamped to [0, n-2]. Assumes
// x.front() < r < x.back() (callers handle the boundaries separately).
std::size_t locate_interval(const std::vector<double> &x, double r) {
  BOOST_ASSERT_MSG(std::ranges::is_sorted(x), "Expected sorted x values");
  const std::size_t n = x.size();
  std::size_t i = static_cast<std::size_t>(
      std::upper_bound(x.begin(), x.end(), r) - x.begin());
  if (i > 0) {
    --i;
  }
  if (i > n - 2) {
    i = n - 2;
  }
  return i;
}

} // namespace

SplinePotential::SplinePotential(std::vector<double> x, std::vector<double> y)
    : x_(std::move(x)), y_(std::move(y)), fixed_(x_.size(), false) {
  assert(x_.size() == y_.size() && x_.size() >= 2);
  recompute_slopes_();
#ifdef FORCESMITH_SPLINE_VERIFY
  rebuild_interp_();
#endif
}

// Port of boost::math::interpolators::makima's slope computation (makima.hpp).
// We own the slopes so the per-bond hot path can evaluate the Hermite cubic
// directly from (y_, s_) with no interpolator object and no binary search.
void SplinePotential::recompute_slopes_() {
  using std::abs;
  using std::isnan;
  const std::size_t n = x_.size();
  if (n < 4) {
    s_.clear(); // piecewise-linear fallback; no cubic slopes needed
    return;
  }
  s_.assign(n, 0.0);

  double m2 = (y_[3] - y_[2]) / (x_[3] - x_[2]);
  double m1 = (y_[2] - y_[1]) / (x_[2] - x_[1]);
  double m0 = (y_[1] - y_[0]) / (x_[1] - x_[0]);
  double mm1 = 2 * m0 - m1;  // quadratic extrapolation m_{-1}
  double mm2 = 2 * mm1 - m0; // quadratic extrapolation m_{-2}
  double w1 = abs(m1 - m0) + abs(m1 + m0) / 2;
  double w2 = abs(mm1 - mm2) + abs(mm1 + mm2) / 2;
  s_[0] = (w1 * mm1 + w2 * m0) / (w1 + w2);
  if (isnan(s_[0])) {
    s_[0] = 0;
  }

  w1 = abs(m2 - m1) + abs(m2 + m1) / 2;
  w2 = abs(m0 - mm1) + abs(m0 + mm1) / 2;
  s_[1] = (w1 * m0 + w2 * m1) / (w1 + w2);
  if (isnan(s_[1])) {
    s_[1] = 0;
  }

  for (std::size_t i = 2; i < n - 2; ++i) {
    double mim2 = (y_[i - 1] - y_[i - 2]) / (x_[i - 1] - x_[i - 2]);
    double mim1 = (y_[i] - y_[i - 1]) / (x_[i] - x_[i - 1]);
    double mi = (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]);
    double mip1 = (y_[i + 2] - y_[i + 1]) / (x_[i + 2] - x_[i + 1]);
    w1 = abs(mip1 - mi) + abs(mip1 + mi) / 2;
    w2 = abs(mim1 - mim2) + abs(mim1 + mim2) / 2;
    s_[i] = (w1 * mim1 + w2 * mi) / (w1 + w2);
    if (isnan(s_[i])) {
      s_[i] = 0;
    }
  }

  double mnm4 = (y_[n - 3] - y_[n - 4]) / (x_[n - 3] - x_[n - 4]);
  double mnm3 = (y_[n - 2] - y_[n - 3]) / (x_[n - 2] - x_[n - 3]);
  double mnm2 = (y_[n - 1] - y_[n - 2]) / (x_[n - 1] - x_[n - 2]);
  double mnm1 = 2 * mnm2 - mnm3;
  double mn = 2 * mnm1 - mnm2;
  w1 = abs(mnm1 - mnm2) + abs(mnm1 + mnm2) / 2;
  w2 = abs(mnm3 - mnm4) + abs(mnm3 + mnm4) / 2;
  s_[n - 2] = (w1 * mnm3 + w2 * mnm2) / (w1 + w2);
  if (isnan(s_[n - 2])) {
    s_[n - 2] = 0;
  }

  w1 = abs(mn - mnm1) + abs(mn + mnm1) / 2;
  w2 = abs(mnm2 - mnm3) + abs(mnm2 + mnm3) / 2;
  s_[n - 1] = (w1 * mnm2 + w2 * mnm1) / (w1 + w2);
  if (isnan(s_[n - 1])) {
    s_[n - 1] = 0;
  }
}

#ifdef FORCESMITH_SPLINE_VERIFY
void SplinePotential::rebuild_interp_() {
  if (x_.size() >= 4) {
    interp_.emplace(std::vector<double>(x_), std::vector<double>(y_));
  } else {
    interp_.reset();
  }
}
#endif

// Cubic Hermite value/derivative on interval i, written exactly as boost's
// cubic_hermite_detail (cubic_hermite_detail.hpp) so eval/deriv stay
// bit-identical to the previous boost-backed implementation given matching
// slopes (which FORCESMITH_SPLINE_VERIFY checks).
double SplinePotential::hermite_eval_(std::size_t i, double r) const {
  const double x0 = x_[i], x1 = x_[i + 1];
  const double y0 = y_[i], y1 = y_[i + 1];
  const double s0 = s_[i], s1 = s_[i + 1];
  const double dx = x1 - x0;
  const double t = (r - x0) / dx;
  return (1 - t) * (1 - t) * (y0 * (1 + 2 * t) + s0 * (r - x0)) +
         t * t * (y1 * (3 - 2 * t) + dx * s1 * (t - 1));
}

double SplinePotential::hermite_deriv_(std::size_t i, double r) const {
  const double x0 = x_[i], x1 = x_[i + 1];
  const double y0 = y_[i], y1 = y_[i + 1];
  const double s0 = s_[i], s1 = s_[i + 1];
  const double dx = x1 - x0;
  const double d1 = (y1 - y0 - s0 * dx) / (dx * dx);
  const double d2 = (s1 - s0) / (2 * dx);
  const double c2 = 3 * d1 - 2 * d2;
  const double c3 = 2 * (d2 - d1) / dx;
  return s0 + 2 * c2 * (r - x0) + 3 * c3 * (r - x0) * (r - x0);
}

void SplinePotential::gather_params(Eigen::VectorXd &dst,
                                    std::size_t offset) const {
  // Spline knots are stored as parallel value/fixed arrays (not Param objects),
  // so use the (values, fixed) overload of the shared leaf loop.
  detail::gather_params_impl(y_, fixed_, dst, offset);
}

void SplinePotential::scatter_params(const Eigen::VectorXd &src,
                                     std::size_t offset) {
  detail::scatter_params_impl(y_, fixed_, src, offset);
  // Only the knot VALUES changed; the x-grid (and therefore every cached site)
  // is untouched. Recompute the cheap per-knot slopes; do NOT clear sites_.
  recompute_slopes_();
#ifdef FORCESMITH_SPLINE_VERIFY
  rebuild_interp_();
#endif
}

void SplinePotential::write_curvature(Eigen::VectorXd &dst, std::size_t offset,
                                      double weight) const {
  if (curvature_count() == 0) {
    return;
  }
  // One residual per interior knot: weight * (y[k-1] - 2 y[k] + y[k+1]).
  for (std::size_t k = 1; k + 1 < y_.size(); ++k) {
    dst[offset++] = weight * (y_[k - 1] - 2.0 * y_[k] + y_[k + 1]);
  }
}

double SplinePotential::eval(double r) const {
  // Out of range: linear extrapolation using the boundary slope, consistent
  // with deriv()
  const std::size_t n = x_.size();
  if (r <= x_.front()) {
    const double slope = (y_[1] - y_[0]) / (x_[1] - x_[0]);
    return y_.front() + slope * (r - x_.front());
  } else if (r >= x_.back()) {
    const double slope = (y_[n - 1] - y_[n - 2]) / (x_[n - 1] - x_[n - 2]);
    return y_.back() + slope * (r - x_.back());
  } else if (s_.empty()) {
    const std::size_t i = locate_interval(x_, r); // n < 4: piecewise-linear
    const double t = (r - x_[i]) / (x_[i + 1] - x_[i]);
    return y_[i] + t * (y_[i + 1] - y_[i]);
  }
  const double v = hermite_eval_(locate_interval(x_, r), r);
#ifdef FORCESMITH_SPLINE_VERIFY
  BOOST_ASSERT(interp_ && std::abs(v - std::invoke(*interp_, r)) <=
                              1e-9 * (1.0 + std::abs(v)));
#endif
  return v;
}

double SplinePotential::deriv(double r) const {
  const std::size_t n = x_.size();
  if (r <= x_.front()) {
    return (y_[1] - y_[0]) / (x_[1] - x_[0]);
  } else if (r >= x_.back()) {
    return (y_[n - 1] - y_[n - 2]) / (x_[n - 1] - x_[n - 2]);
  } else if (s_.empty()) {
    const std::size_t i = locate_interval(x_, r); // n < 4: piecewise-linear
    return (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]);
  }
  const double d = hermite_deriv_(locate_interval(x_, r), r);
#ifdef FORCESMITH_SPLINE_VERIFY
  BOOST_ASSERT(interp_ && std::abs(d - interp_->prime(r)) <=
                              1e-9 * (1.0 + std::abs(d)));
#endif
  return d;
}

// Fused eval(r)+deriv(r): the branch ladder mirrors eval()/deriv() exactly, but
// the interior cubic case does ONE locate_interval and feeds the shared index to
// both hermite helpers — saving the second binary search. Each returned
// component is bit-identical to the standalone call.
std::pair<double, double> SplinePotential::eval_and_deriv(double r) const {
  const std::size_t n = x_.size();
  if (r <= x_.front()) {
    const double slope = (y_[1] - y_[0]) / (x_[1] - x_[0]);
    return {y_.front() + slope * (r - x_.front()), slope};
  } else if (r >= x_.back()) {
    const double slope = (y_[n - 1] - y_[n - 2]) / (x_[n - 1] - x_[n - 2]);
    return {y_.back() + slope * (r - x_.back()), slope};
  } else if (s_.empty()) {
    const std::size_t i = locate_interval(x_, r); // n < 4: piecewise-linear
    const double t = (r - x_[i]) / (x_[i + 1] - x_[i]);
    return {y_[i] + t * (y_[i + 1] - y_[i]),
            (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i])};
  }
  const std::size_t i = locate_interval(x_, r);
  const double v = hermite_eval_(i, r);
  const double d = hermite_deriv_(i, r);
#ifdef FORCESMITH_SPLINE_VERIFY
  BOOST_ASSERT(interp_ && std::abs(v - std::invoke(*interp_, r)) <=
                              1e-9 * (1.0 + std::abs(v)));
  BOOST_ASSERT(interp_ && std::abs(d - interp_->prime(r)) <=
                              1e-9 * (1.0 + std::abs(d)));
#endif
  return {v, d};
}

int SplinePotential::prepare_site(double r) const {
  if (auto it = site_of_r_.find(r); it != site_of_r_.end()) {
    return it->second;
  }
  const std::size_t n = x_.size();
  EvalSite st;
  st.r = r;
  if (r <= x_.front()) {
    st.kind = EvalSite::Kind::Linear;
    st.i = 0; // y_[0] + slope01 * (r - x_[0])
  } else if (r >= x_.back()) {
    st.kind = EvalSite::Kind::Linear;
    st.i = n - 2; // anchor on the last interval (equivalent to back-slope form)
  } else if (s_.empty()) {
    st.kind = EvalSite::Kind::Linear; // n < 4 interior: piecewise-linear
    st.i = locate_interval(x_, r);
  } else {
    st.kind = EvalSite::Kind::Cubic;
    st.i = locate_interval(x_, r);
  }

  const std::size_t i = st.i;
  const double dx = x_[i + 1] - x_[i];
  st.inv_dx = 1.0 / dx;
  st.off = r - x_[i];

  if (st.kind == EvalSite::Kind::Cubic) {
    const double t = st.off * st.inv_dx; // (r - x_[i]) / dx
    const double omt = 1.0 - t;
    // Value basis (matches boost cubic_hermite operator()): the s0 term is
    // (1-t)^2 * s0 * (r-x0) = (1-t)^2 * t * dx * s0.
    st.v0 = omt * omt * (1.0 + 2.0 * t);
    st.v1 = t * t * (3.0 - 2.0 * t);
    st.vs0 = omt * omt * t * dx;
    st.vs1 = t * t * (t - 1.0) * dx;
    // Derivative basis: d/dx of the value basis (chain rule, dt/dx = 1/dx).
    st.d0 = (6.0 * t * t - 6.0 * t) * st.inv_dx;
    st.d1 = (-6.0 * t * t + 6.0 * t) * st.inv_dx;
    st.ds0 = 3.0 * t * t - 4.0 * t + 1.0;
    st.ds1 = 3.0 * t * t - 2.0 * t;
  }

  const int idx = static_cast<int>(sites_.size());
  sites_.push_back(st);
  site_of_r_.emplace(r, idx);
  return idx;
}

// eval_at / deriv_at / eval_and_deriv_at are defined FORCE_INLINE in spline.hpp
// so they fuse into the force calculators' hot neighbor loops.

} // namespace forcesmith
