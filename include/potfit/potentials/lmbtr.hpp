#pragma once

// Local Many-Body Tensor Representation (LMBTR) — a local per-atom ML
// descriptor (DScribe's LMBTR). Around each central atom it builds:
//   k2 (distance):  for every neighbour j, the geometry value r_ij is broadened
//                   by a unit-area Gaussian onto a fixed distance grid, weighted
//                   by w = exp(−r_ij / weight_scale), channelled per neighbour
//                   species s∈[0,ntypes).
//   k3 (cosine):    for every neighbour pair (j,k), the geometry value cosθ_ijk
//                   is broadened onto a fixed cosine grid, weighted by
//                   w = exp(−(r_ij+r_ik+r_jk) / weight_scale), channelled per
//                   unordered species pair (upper_triangle(ntypes), as SOAP).
// The full vector is optionally L2-normalized. All hyperparameters are fixed;
// only the head coefficients are fitted. Gradients use the framework's
// finite-difference fallback (analytic_grads() == false).

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/ml_force.hpp"

#include <cstddef>

namespace potfit {

struct LMBTR : MLBase<LMBTR> {
  struct Grid {
    double min = 0.0;
    double max = 6.0;
    int n = 50;
    double sigma = 0.3;
  };

  bool use_k2 = true;
  bool use_k3 = true;
  Grid k2{0.0, 6.0, 50, 0.3};  // geometry = distance r_ij
  Grid k3{-1.0, 1.0, 50, 0.1}; // geometry = cos θ_ijk
  double rcut = 6.0;
  double weight_scale = 3.0; // exp weighting decay length
  bool normalize_l2 = true;

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] constexpr double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] constexpr bool analytic_grads() const { return false; }
  [[nodiscard]] std::size_t descriptor_size() const {
    const std::size_t S = ntypes;
    const std::size_t P = S * (S + 1) / 2;
    return (use_k2 ? S * static_cast<std::size_t>(k2.n) : 0) +
           (use_k3 ? P * static_cast<std::size_t>(k3.n) : 0);
  }
};

static_assert(ForceCalculatorModel<LMBTR>);

} // namespace potfit
