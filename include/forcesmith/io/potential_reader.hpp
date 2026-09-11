#pragma once

#include "forcesmith/core/radial_potential.hpp"
#include "forcesmith/io/config_reader.hpp" // ParseError

#include <boost/leaf/result.hpp>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace forcesmith::io {

boost::leaf::result<std::vector<RadialPotential>>
parse_potential(std::string_view input);

std::optional<std::size_t> analytic_param_index(std::string_view type,
                                                std::string_view param);

} // namespace forcesmith::io
