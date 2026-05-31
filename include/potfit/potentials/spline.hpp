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

  constexpr void set_fixed(std::size_t i, bool f) { fixed_[i] = f; }
  constexpr bool is_fixed(std::size_t i) const { return fixed_[i]; }

  constexpr std::size_t param_count() const {
    return static_cast<std::size_t>(std::ranges::count(fixed_, false));
  }
  void gather_params(Eigen::VectorXd &dst, std::size_t offset) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t offset);

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
  using Makima = boost::math::interpolators::makima<std::vector<double>>;

  std::vector<double> x_, y_;
  std::vector<bool> fixed_;
  std::optional<Makima> interp_; // null for the degenerate 2-knot (linear) case
  void rebuild_interp_();
};

// Curvature customization-point overloads (found by ADL from the erased
// Potential); these make a tabulated potential participate in the Tikhonov
// smoothness regularization. See potfit/potentials/curvature.hpp.
inline std::size_t curvature_count(const SplinePotential &p) {
  return p.curvature_count();
}
inline void write_curvature(const SplinePotential &p, Eigen::VectorXd &dst,
                            std::size_t off, double weight) {
  p.write_curvature(dst, off, weight);
}

} // namespace potfit
