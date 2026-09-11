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
  std::size_t column = 0; // 1-based; 0 == unknown (filled by io/grammar.hpp)
};

struct ParsedConfig {
  std::vector<Configuration> configs;
  SpeciesRegistry registry;
};

boost::leaf::result<ParsedConfig> parse_config(std::string_view input);

} // namespace forcesmith::io
