#pragma once

// Behler-style atom-centered symmetry functions — the first concrete ML
// descriptor, used to exercise the MLBaseImpl pipeline end to end.
//
// Per atom i, each radial G2 descriptor is
//   D_k(i) = Σ_{j∈neighbors(i)} exp(−η_k (r_ij − Rs_k)²) · f_c(r_ij)
// with the cosine cutoff f_c(r) = ½(1 + cos(π r / rcut)) for r < rcut, else 0.
// η_k and Rs_k are fixed hyperparameters; only the head coefficients are
// fitted. dD_k/dr is closed-form, so this model reports analytic gradients.

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/ml_force.hpp"

#include <cstddef>
#include <vector>

namespace potfit {

struct SymmetryFunctionModel : MLBase<SymmetryFunctionModel> {
  struct G2 {
    double eta = 1.0; // Gaussian width
    double rs = 0.0;  // Gaussian centre (shift)
  };

  std::vector<G2> radial; // descriptor_size = radial.size()
  double rcut = 6.0;
  bool use_analytic_grads = true;

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] constexpr double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] constexpr bool analytic_grads() const {
    return use_analytic_grads;
  }
  [[nodiscard]] constexpr std::size_t descriptor_size() const {
    return radial.size();
  }
};

static_assert(ForceCalculatorModel<SymmetryFunctionModel>);

} // namespace potfit
