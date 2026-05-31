#include "potfit/potentials/spline.hpp"

#include <algorithm>
#include <boost/assert.hpp>
#include <cassert>
#include <ranges>

namespace potfit {

namespace {

// Piecewise-linear fallback used when n < 4 (makima needs >= 4 points).
double pw_linear_eval(const std::vector<double> &x,
                      const std::vector<double> &y, double r) {
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
  const double t = (r - x[i]) / (x[i + 1] - x[i]);
  return y[i] + t * (y[i + 1] - y[i]);
}

double pw_linear_deriv(const std::vector<double> &x,
                       const std::vector<double> &y, double r) {
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
  return (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
}

} // namespace

SplinePotential::SplinePotential(std::vector<double> x, std::vector<double> y)
    : x_(std::move(x)), y_(std::move(y)), fixed_(x_.size(), false) {
  assert(x_.size() == y_.size() && x_.size() >= 2);
  rebuild_interp_();
}

void SplinePotential::rebuild_interp_() {
  if (x_.size() >= 4) {
    interp_.emplace(std::vector<double>(x_), std::vector<double>(y_));
  } else {
    interp_.reset(); // n < 4: piecewise-linear fallback in eval/deriv
  }
}

void SplinePotential::gather_params(Eigen::VectorXd &dst,
                                    std::size_t offset) const {
  for (auto [y, fixed] : std::views::zip(y_, fixed_)) {
    if (!fixed) {
      dst[offset++] = y;
    }
  }
}

void SplinePotential::scatter_params(const Eigen::VectorXd &src,
                                     std::size_t offset) {
  for (auto &&[y, fixed] : std::views::zip(y_, fixed_)) {
    if (!fixed) {
      y = src[offset++];
    }
  }
  rebuild_interp_();
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
  // with deriv() and with potfit's splint (e.g. EAM embedding F(ρ) sampled
  // outside the tabulated density range).
  const std::size_t n = x_.size();
  if (r <= x_.front()) {
    const double slope = (y_[1] - y_[0]) / (x_[1] - x_[0]);
    return y_.front() + slope * (r - x_.front());
  } else if (r >= x_.back()) {
    const double slope = (y_[n - 1] - y_[n - 2]) / (x_[n - 1] - x_[n - 2]);
    return y_.back() + slope * (r - x_.back());
  } else if (!interp_) {
    return pw_linear_eval(x_, y_, r);
  }
  return std::invoke(*interp_, r);
}

double SplinePotential::deriv(double r) const {
  const std::size_t n = x_.size();
  if (r <= x_.front()) {
    return (y_[1] - y_[0]) / (x_[1] - x_[0]);
  } else if (r >= x_.back()) {
    return (y_[n - 1] - y_[n - 2]) / (x_[n - 1] - x_[n - 2]);
  } else if (!interp_) {
    return pw_linear_deriv(x_, y_, r);
  }
  return interp_->prime(r);
}

} // namespace potfit
