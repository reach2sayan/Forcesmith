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
#include <memory>

namespace potfit {

// Tabulated radial projection J(a,l)(r) as cubic splines in r — the heavy part of
// the SOAP descriptor (an adaptive Bessel quadrature per (a,l) per neighbor). Built
// once over [0,rcut]; get_descriptor then evaluates a spline per neighbor instead.
// Opaque here (definition + boost spline live in soap.cpp) to keep boost::math out
// of this header. Shared read-only via shared_ptr so model copies (e.g. the
// parallel Jacobian) never duplicate it.
struct SoapRadialTable;

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

  // Lazily-built radial spline table (see SoapRadialTable). Auto-built on first
  // get_descriptor; init_radial_basis() also builds it up front.
  mutable std::shared_ptr<const SoapRadialTable> radial_;
  void init_radial_basis();

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] bool analytic_grads() const { return true; }
  [[nodiscard]] std::size_t descriptor_size() const;
};

static_assert(ForceCalculatorModel<SoapModel>);

} // namespace potfit
