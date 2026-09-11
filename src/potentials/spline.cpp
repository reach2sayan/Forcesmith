#include "forcesmith/potentials/spline.hpp"

#include "forcesmith/core/symbolic.hpp"

#include "forcesmith/core/fit_params.hpp"

#include <algorithm>
#include <boost/assert.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include <ranges>

namespace forcesmith {

namespace {

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

namespace hermite {

using ddx::named;
using ddx::var;

inline constexpr auto r = var<"r">;
inline constexpr auto x0 = var<"x0">;
inline constexpr auto dx = var<"dx">;
inline constexpr auto y0 = var<"y0">;
inline constexpr auto y1 = var<"y1">;
inline constexpr auto s0 = var<"s0">;
inline constexpr auto s1 = var<"s1">;

// Boost's cubic_hermite_detail spelling, kept verbatim so the VALUE stays
// bit-identical to the boost-backed implementation this port replaced.
inline constexpr auto kExpr = [] {
  const auto t = (r - x0) / dx;
  const auto omt = 1.0 - t;
  return omt * omt * (y0 * (1.0 + 2.0 * t) + s0 * (r - x0)) +
         t * t * (y1 * (3.0 - 2.0 * t) + dx * s1 * (t - 1.0));
}();

inline constexpr auto kValue = ddx::Equation{kExpr};
inline constexpr auto kDeriv = symbolic::derivative_of<"r">(kValue);

// Slots by name PER EQUATION: a derivative may carry fewer symbols, so another
// equation's slots would silently read the wrong partial.
template <class Eq, ddx::impl::FixedString N>
inline constexpr std::size_t slot = symbolic::slot_of<Eq>(N.view());

inline auto point(const std::vector<double> &x, const std::vector<double> &y,
                  const std::vector<double> &s, std::size_t i, double rr) {
  return decltype(kValue)::point(
      named<"dx">(x[i + 1] - x[i]), named<"r">(rr), named<"s0">(s[i]),
      named<"s1">(s[i + 1]), named<"x0">(x[i]), named<"y0">(y[i]),
      named<"y1">(y[i + 1]));
}

// Value basis = d(value)/d(y0,y1,s0,s1); the derivative basis must come from
// derivative_tensor<2>: jacobian() of the stored r-derivative silently returns
// zeros (ddx freezes a partial tree's symbols as constants).
struct Basis {
  double c_y0, c_y1, c_s0, c_s1;
};

template <ddx::impl::FixedString N>
inline constexpr std::size_t vslot = slot<decltype(kValue), N>;

inline Basis value_basis(double x0v, double dxv, double rr) {
  const auto j = kValue.jacobian(named<"dx">(dxv), named<"r">(rr),
                                 named<"s0">(0.0), named<"s1">(0.0),
                                 named<"x0">(x0v), named<"y0">(0.0),
                                 named<"y1">(0.0));
  return Basis{j[vslot<"y0">], j[vslot<"y1">], j[vslot<"s0">],
               j[vslot<"s1">]};
}

inline Basis deriv_basis(double x0v, double dxv, double rr) {
  const auto t = kValue.derivative_tensor<2>(
      named<"dx">(dxv), named<"r">(rr), named<"s0">(0.0), named<"s1">(0.0),
      named<"x0">(x0v), named<"y0">(0.0), named<"y1">(0.0));
  constexpr std::size_t R = vslot<"r">;
  return Basis{t[R, vslot<"y0">], t[R, vslot<"y1">], t[R, vslot<"s0">],
               t[R, vslot<"s1">]};
}

} // namespace hermite

double SplinePotential::hermite_eval_(std::size_t i, double r) const {
  return hermite::kValue.evaluate(hermite::point(x_, y_, s_, i, r));
}

double SplinePotential::hermite_deriv_(std::size_t i, double r) const {
  return hermite::kDeriv.evaluate(hermite::point(x_, y_, s_, i, r));
}

void SplinePotential::gather_params(Eigen::VectorXd &dst,
                                    std::size_t offset) const {
  detail::gather_params_impl(y_, fixed_, dst, offset);
}

void SplinePotential::scatter_params(const Eigen::VectorXd &src,
                                     std::size_t offset) {
  detail::scatter_params_impl(y_, fixed_, src, offset);
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
  for (std::size_t k = 1; k + 1 < y_.size(); ++k) {
    dst[offset++] = weight * (y_[k - 1] - 2.0 * y_[k] + y_[k + 1]);
  }
}

double SplinePotential::eval(double r) const {
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

// Fused eval(r)+deriv(r), bit-identical to the separate calls.
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
    const auto v = hermite::value_basis(x_[i], dx, r);
    st.v0 = v.c_y0;
    st.v1 = v.c_y1;
    st.vs0 = v.c_s0;
    st.vs1 = v.c_s1;
    const auto d = hermite::deriv_basis(x_[i], dx, r);
    st.d0 = d.c_y0;
    st.d1 = d.c_y1;
    st.ds0 = d.c_s0;
    st.ds1 = d.c_s1;
  }

  const int idx = static_cast<int>(sites_.size());
  sites_.push_back(st);
  site_of_r_.emplace(r, idx);
  return idx;
}

} // namespace forcesmith
