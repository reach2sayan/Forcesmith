#include "potfit/potentials/symmetry_functions.hpp"

#include <cmath>
#include <numbers>

namespace potfit {

DescriptorValue SymmetryFunctionModel::get_descriptor(const Atom &a) const {
  const auto S = static_cast<Eigen::Index>(radial.size());
  const std::size_t nn = a.neighbors.size();

  DescriptorValue out;
  out.values = Eigen::VectorXd::Zero(S);
  out.has_grad = true;
  out.grad_self = DescriptorGrad::Zero(S, 3);
  out.grad_neigh.assign(nn, DescriptorGrad::Zero(S, 3));

  for (std::size_t jj = 0; jj < nn; ++jj) {
    const Vec3 &dist = a.neighbors[jj].dist; // r_j − r_i
    const double r = dist.norm();
    if (r < 1e-14 || r >= rcut) {
      continue;
    }

    const double x = std::numbers::pi * r / rcut;
    const double fc = 0.5 * (1.0 + std::cos(x));
    const double dfc = -0.5 * std::numbers::pi / rcut * std::sin(x);
    const Vec3 rhat = dist / r; // dr/dr_j

    for (Eigen::Index k = 0; k < S; ++k) {
      const G2 &g2 = radial[static_cast<std::size_t>(k)];
      const double dr = r - g2.rs;
      const double gauss = std::exp(-g2.eta * dr * dr);
      const double g = gauss * fc;                              // term value
      const double dg = gauss * (-2.0 * g2.eta * dr * fc + dfc); // dterm/dr

      out.values[k] += g;
      // dD_k/dr_j = dg · r̂ ; dD_k/dr_i = −dg · r̂ (translational invariance).
      out.grad_neigh[jj].row(k) = dg * rhat.transpose();
      out.grad_self.row(k) -= dg * rhat.transpose();
    }
  }

  return out;
}

} // namespace potfit
