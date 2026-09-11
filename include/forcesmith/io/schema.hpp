#pragma once

#include "forcesmith/core/json.hpp"
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"

#include <array>
#include <string_view>

namespace forcesmith {

template <>
inline constexpr std::array optional_members<TersoffParams>{
    std::string_view("omega")};

} // namespace forcesmith
