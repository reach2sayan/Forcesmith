#pragma once

#include "potfit/api/potfit.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>

namespace potfit::io {

boost::leaf::result<void> load_configs(const std::filesystem::path &path,
                                       PotFit &session);

boost::leaf::result<void> load_model(const std::filesystem::path &path,
                                     PotFit &session);

} // namespace potfit::io
