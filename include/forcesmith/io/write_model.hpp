#pragma once

#include "forcesmith/force/force_calculator.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>
#include <string_view>

namespace forcesmith::io {

boost::leaf::result<void> write_model(const ForceCalculator &model,
                                      const std::filesystem::path &path,
                                      std::string_view format = "native");

} // namespace forcesmith::io
