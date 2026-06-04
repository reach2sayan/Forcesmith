#pragma once

#include "potfit/force/adp_force.hpp"
#include "potfit/force/angular_force.hpp"
#include "potfit/force/eam_force.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/force/stiweb_force.hpp"
#include "potfit/force/tersoff_force.hpp"
#include "potfit/potentials/acsf.hpp"
#include "potfit/potentials/lmbtr.hpp"
#include "potfit/potentials/soap.hpp"

#include <variant>

namespace potfit {
// clang-format off
using ForceCalculator = std::variant<
    PairForceCalculator,
    EAMForceCalculator,
    ADPForceCalculator,
    AngularForceCalculator,
    TersoffForceCalculator,
    StiwebForceCalculator,
    ACSF,
    SoapModel,
    LMBTR
>;
// clang-format on

} // namespace potfit
