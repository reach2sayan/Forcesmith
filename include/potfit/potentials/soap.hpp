#pragma once

// SOAP (Smooth Overlap of Atomic Positions) descriptor — the standard
// power-spectrum representation of a local atomic environment, as referenced in
// dscribe and the VASP ML force-field theory.
//
// Per atom i, the neighbour density of each species channel α is expanded onto an
// orthonormal radial basis g_n(r) and (complex) spherical harmonics Y_lm:
//   c^α_{nlm} = K Σ_{j∈α} f_c(r_j) · I^j_{nl} · Y*_{lm}(r̂_j)
// where the radial projection I^j_{nl} comes from the atomic-Gaussian overlap and
// uses the modified spherical Bessel function i_l (boost::math::cyl_bessel_i).
// The rotationally invariant power spectrum is
//   p^{αβ}_{n n' l} = w_l Σ_m c^α_{nlm} · conj(c^β_{n'nlm})
// flattened (α≤β, n≤n' for α=β) and L2-normalised to form the descriptor.
//
// First pass: descriptor VALUES only (analytic_grads()==false), so MLBase's
// finite-difference force fallback handles forces. Analytic dD/dr is a later pass.
// boost::math provides Y_lm / i_l / quadrature; Eigen provides the radial-basis
// orthonormalisation (S^{-1/2}) and all matrix algebra.

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/ml_force.hpp"

#include <Eigen/Core>

#include <cstddef>

namespace potfit {

struct SoapModel : MLBase<SoapModel> {
  using MLBase<SoapModel>::ntypes;
  int n_max = 6;       // radial basis size
  int l_max = 6;       // max spherical-harmonic degree
  double rcut = 6.0;   // environment cutoff (Å)
  double sigma = 0.5;  // atomic Gaussian width (Å)

  // n_max × n_max radial orthonormalisation matrix β = S^{-1/2}. Optional cache:
  // get_descriptor auto-computes β on demand if this is empty, so callers never
  // need to. init_radial_basis() just precomputes it once (single-threaded) as a
  // performance optimisation.
  mutable Eigen::MatrixXd beta;

  void init_radial_basis(); // optional precompute; not required before use

  // ── CRTP hooks consumed by MLBase ──────────────────────────────────────────
  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] bool analytic_grads() const { return false; }
  [[nodiscard]] std::size_t descriptor_size() const;
};

static_assert(ForceCalculatorModel<SoapModel>);

} // namespace potfit
