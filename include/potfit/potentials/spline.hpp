#pragma once

#include <Eigen/Core>
#include <utility>
#include <vector>

namespace potfit {

class SplinePotential {
public:
  SplinePotential(std::vector<double> x, std::vector<double> y);
  double eval(double r) const;
  double deriv(double r) const;
  constexpr std::pair<double, double> span() const {
    return {x_[0], x_[x_.size() - 1]};
  }

  // Param interface: free parameters are the knot y-values (x-knots are fixed).
  constexpr int param_count() const { return static_cast<int>(x_.size()); }
  void gather_params(Eigen::VectorXd &dst, int offset) const {
    dst.segment(offset, y_.size()) = y_;
  }

  void scatter_params(const Eigen::VectorXd &src, int offset) {
    y_ = src.segment(offset, y_.size());
    compute_d2y();
  }

private:
  Eigen::VectorXd x_, y_, d2y_; // knots, values, second derivatives
  void compute_d2y(); // recomputes d2y_ from x_ and y_ (Thomas algorithm)
};

} // namespace potfit
