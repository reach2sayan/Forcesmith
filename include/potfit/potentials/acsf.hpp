#pragma once

// Behler-style atom-centered symmetry functions (ACSF) — a local per-atom ML
// descriptor that feeds the MLBaseImpl pipeline.
//
// Families (DScribe ACSF conventions), all using the cosine cutoff
//   f_c(r) = ½(1 + cos(π r / rcut)) for r < rcut, else 0:
//   G1  (per species)       Σ_j f_c(r_ij)
//   G2  (per species)       Σ_j exp(−η (r_ij − Rs)²) f_c(r_ij)
//   G3  (per species)       Σ_j cos(κ r_ij) f_c(r_ij)
//   G4  (per species pair)  2^{1−ζ} Σ_{j<k} (1+λ cosθ)^ζ
//                              exp(−η(r_ij²+r_ik²+r_jk²)) f_c(r_ij)f_c(r_ik)f_c(r_jk)
//   G5  (per species pair)  2^{1−ζ} Σ_{j<k} (1+λ cosθ)^ζ
//                              exp(−η(r_ij²+r_ik²)) f_c(r_ij)f_c(r_ik)
// cosθ = (d_j·d_k)/(r_ij r_ik). Radial families channel over the neighbour
// species s∈[0,ntypes); angular families channel over unordered species pairs
// (the upper_triangle(ntypes) enumeration, matching SOAP). All hyperparameters
// are fixed; only the head coefficients are fitted. dD/dr is closed-form for
// every family, so the model reports analytic gradients.

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/ml_force.hpp"

#include <cstddef>
#include <vector>

namespace potfit {

struct ACSF : MLBase<ACSF> {
  struct G2 {
    double eta = 1.0; // Gaussian width
    double rs = 0.0;  // Gaussian centre (shift)
  };
  struct G3 {
    double kappa = 1.0; // cosine wavevector
  };
  struct G4 {
    double eta = 1.0;
    double zeta = 1.0;
    double lambda = 1.0; // ±1
  };
  struct G5 {
    double eta = 1.0;
    double zeta = 1.0;
    double lambda = 1.0; // ±1
  };

  std::size_t g1 = 0;     // number of parameterless G1 channels (Σ f_c)
  std::vector<G2> radial; // G2 functions
  std::vector<G3> g3;     // G3 functions
  std::vector<G4> g4;     // G4 angular functions
  std::vector<G5> g5;     // G5 angular functions
  double rcut = 6.0;
  bool use_analytic_grads = true;

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] constexpr double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] constexpr bool analytic_grads() const {
    return use_analytic_grads;
  }
  [[nodiscard]] std::size_t descriptor_size() const {
    const std::size_t S = ntypes;
    const std::size_t P = S * (S + 1) / 2;
    return S * (g1 + radial.size() + g3.size()) + P * (g4.size() + g5.size());
  }
};

static_assert(ForceCalculatorModel<ACSF>);

} // namespace potfit
