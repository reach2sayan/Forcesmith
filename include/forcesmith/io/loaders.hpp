#pragma once

#include "forcesmith/api/forcesmith.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>

namespace forcesmith::io {

boost::leaf::result<void> load_configs(const std::filesystem::path &path,
                                       Forcesmith &session);

boost::leaf::result<void> load_model(const std::filesystem::path &path,
                                     Forcesmith &session);

} // namespace forcesmith::io
