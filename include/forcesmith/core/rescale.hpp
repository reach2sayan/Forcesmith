#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/eam_force.hpp"

#include <span>
#include <vector>

namespace forcesmith {

std::vector<double> compute_rho_ref(EAMForceCalculator &calc,
                                    std::span<Configuration> configs);

double rescale_rho_axis(EAMForceCalculator &calc,
                        std::span<Configuration> configs);

void embed_shift(EAMForceCalculator &calc, std::span<const double> rho_ref);

void rescale_eam(EAMForceCalculator &calc, std::span<Configuration> configs);

} // namespace forcesmith
