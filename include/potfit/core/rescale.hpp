#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/eam_force.hpp"

#include <span>
#include <vector>

namespace potfit {

// Compute average electron density per element type across all configurations.
// Calls eval_forces on every config (neighbor lists must be built beforehand).
// Returns a vector of length calc.ntypes.
std::vector<double> compute_rho_ref(EAMForceCalculator &calc,
                                    std::span<Configuration> configs);

// Gauge-invariant linear shift of the EAM embedding functions.
// For each type t:
//   F_t(ρ) → F_t(ρ) − F_t′(rho_ref[t]) × ρ
// The pair potentials are updated to preserve total energy exactly:
//   φ_{αβ}(r) → φ_{αβ}(r) + slope_α × g_β(r) + slope_β × g_α(r)
// After this call F_t′(rho_ref[t]) == 0 for all t.
void embed_shift(EAMForceCalculator &calc, std::span<const double> rho_ref);

// Full EAM rescaling: gauge-normalize + zero embedding at reference density.
//   1. Computes rho_ref from configs and calls embed_shift.
//   2. Re-evaluates rho_ref and shifts each F_t so F_t(rho_ref_t) = 0.
// Intended to be called once after loading potentials (and once more after
// fitting is complete to produce a normalized parameter set).
void rescale_eam(EAMForceCalculator &calc, std::span<Configuration> configs);

} // namespace potfit
