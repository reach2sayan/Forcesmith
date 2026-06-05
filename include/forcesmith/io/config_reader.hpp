#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/species.hpp"

#include <boost/leaf/result.hpp>
#include <string_view>
#include <vector>

namespace forcesmith::io {
struct ParseError {
  std::string message;
  std::size_t line = 0;
};

// Parsed configurations together with the element↔slot registry that stamped
// their atom types. The registry's Z-sorted slot order is authoritative: the
// potential tables a force model is built with must use the same layout.
struct ParsedConfig {
  std::vector<Configuration> configs;
  SpeciesRegistry registry;
};

boost::leaf::result<ParsedConfig> parse_config(std::string_view input);

} // namespace forcesmith::io
