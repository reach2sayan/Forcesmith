#pragma once

#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/io/config_reader.hpp" // ParseError, indirectly pulls in core types

#include <boost/leaf/result.hpp>
#include <string_view>

namespace forcesmith::io {

boost::leaf::result<ForceCalculator> parse_force_model(std::string_view input);

} // namespace forcesmith::io
