#include "forcesmith/optimization/line_search.hpp"

#include <boost/math/tools/minima.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace forcesmith {

std::pair<double, double>
detail::linmin_fn::bracket_minimum(const std::function<double(double)> &g,
                                   double initial_step, double max_step,
                                   int max_iter) {
  using std::numbers::phi;

  double a = 0.0;
  double fa = g(0.0);
  double b = initial_step;
  double fb = g(b);

  if (fb >= fa) {
    return {0.0, b};
  }

  for (int i = 0; i < max_iter; ++i) {
    double c = std::min(b * phi, max_step);
    double fc = g(c);
    if (fc >= fb) {
      return {a, c};
    }
    a = b;
    fa = fb;
    b = c;
    fb = fc;
    if (b >= max_step) {
      break;
    }
  }
  return {a, b};
}

double detail::linmin_fn::operator()(
    Eigen::VectorXd &x, const Eigen::VectorXd &dir,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &F,
    const LinSearchOptions &opts) const {
  auto g = [&](double alpha) -> double {
    return 0.5 * F(x + alpha * dir).squaredNorm();
  };

  auto [lo, hi] =
      bracket_minimum(g, opts.initial_step, opts.max_step, opts.max_iter);

  if (hi - lo <= opts.tol) {
    return 0.0;
  }

  std::uintmax_t iters = static_cast<std::uintmax_t>(opts.max_iter);
  auto [alpha, fval] = boost::math::tools::brent_find_minima(
      g, lo, hi, std::numeric_limits<double>::digits / 2, iters);

  x += alpha * dir;
  return alpha;
}

} // namespace forcesmith
