#pragma once

// JSON-based config reader: top-level array of configuration objects.

#include "potfit/core/atom.hpp"

#include <boost/leaf/result.hpp>
#include <string_view>
#include <vector>

namespace potfit::io {
struct ParseError {
  std::string message;
  std::size_t line = 0;
};

boost::leaf::result<std::vector<Configuration>>
parse_config(std::string_view input);

} // namespace potfit::io
