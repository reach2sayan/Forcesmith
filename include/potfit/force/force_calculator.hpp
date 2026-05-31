#pragma once

#include "potfit/force/adp_force.hpp"
#include "potfit/force/angular_force.hpp"
#include "potfit/force/eam_force.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/force/stiweb_force.hpp"
#include "potfit/force/tersoff_force.hpp"

#include <variant>

namespace potfit {

using ForceCalculator = std::variant<
    PairForceCalculator,
    EAMForceCalculator,
    ADPForceCalculator,
    AngularForceCalculator,
    TersoffForceCalculator,
    StiwebForceCalculator
>;

} // namespace potfit
