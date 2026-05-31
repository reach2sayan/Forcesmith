#pragma once

#include "potfit/core/potential_base.hpp"
#include "potfit/io/config_reader.hpp" // ParseError

#include <boost/leaf/result.hpp>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace potfit::io {

boost::leaf::result<std::vector<Potential>>
parse_potential(std::string_view input);

// Index of parameter `param` within analytic function `type` (registry order,
// matching the struct's params[] layout). Empty if the type/param is unknown.
// Used to resolve a global-parameter reference (e.g. "h") to its slot.
std::optional<std::size_t> analytic_param_index(std::string_view type,
                                                std::string_view param);

} // namespace potfit::io
