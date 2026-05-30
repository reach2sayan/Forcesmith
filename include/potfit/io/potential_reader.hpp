#pragma once

#include "potfit/core/potential_base.hpp"
#include "potfit/io/config_reader.hpp" // ParseError

#include <boost/leaf/result.hpp>
#include <string_view>
#include <vector>

namespace potfit::io {

boost::leaf::result<std::vector<Potential>>
parse_potential(std::string_view input);

} // namespace potfit::io
