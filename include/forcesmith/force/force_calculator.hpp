#pragma once

#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/angular_force.hpp"
#include "forcesmith/force/eam_force.hpp"
#include "forcesmith/force/pair_force.hpp"
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/potentials/acsf.hpp"
#include "forcesmith/potentials/lmbtr.hpp"
#include "forcesmith/potentials/soap.hpp"

#include <variant>

namespace forcesmith {
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

} // namespace forcesmith
