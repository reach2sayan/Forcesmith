#pragma once

// Step 5: two-body pair potential backed by a SplinePotential.

#include "potfit/potentials/spline.hpp"
namespace potfit {

class PairPotential : public SplinePotential {
  using SplinePotential::SplinePotential;
};

} // namespace potfit
