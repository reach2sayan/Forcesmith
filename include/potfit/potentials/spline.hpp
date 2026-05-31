#pragma once

#include <Eigen/Core>
#include <boost/math/interpolators/makima.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace potfit {

class SplinePotential {
public:
  SplinePotential(std::vector<double> x, std::vector<double> y);
  double eval(double r) const;
  double deriv(double r) const;
  std::pair<double, double> span() const { return {x_.front(), x_.back()}; }

  void set_fixed(std::size_t i, bool f) { fixed_[i] = f; }
  bool is_fixed(std::size_t i) const { return fixed_[i]; }

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t offset) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t offset);

private:
  using Makima = boost::math::interpolators::makima<std::vector<double>>;

  std::vector<double> x_, y_;
  std::vector<bool> fixed_;
  std::optional<Makima> interp_; // null for the degenerate 2-knot (linear) case
  void rebuild_interp_();
};

} // namespace potfit
