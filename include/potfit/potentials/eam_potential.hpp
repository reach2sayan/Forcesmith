#pragma once

#include "potfit/core/potential_base.hpp"

namespace potfit {

// EAM = pair repulsion φ(r) + electron density ρ(r) + embedding F(ρ).
// Each component is a type-erased Potential (typically SplinePotential).
struct EAMPotential {
  Potential phi; // pair repulsion
  Potential rho; // electron density
  Potential F;   // embedding function
};

} // namespace potfit
