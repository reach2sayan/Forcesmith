#pragma once

#include "forcesmith/io/config_reader.hpp" // ParseError

#include <boost/leaf/error.hpp>
#include <boost/leaf/result.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace forcesmith::io {

[[nodiscard]] inline boost::leaf::result<std::string>
read_file(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        ParseError{"cannot open " + path.string(), 0});
  }
  return std::string(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
}

[[nodiscard]] inline boost::leaf::result<std::ofstream>
open_out(const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        ParseError{"cannot open " + path.string(), 0});
  }
  return f;
}

} // namespace forcesmith::io
