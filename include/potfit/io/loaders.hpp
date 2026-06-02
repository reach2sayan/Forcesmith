#pragma once

// Thin file-driven front-ends for the PotFit API. These are the only
// difference between "the user builds a fit by hand" and "the CLI loads files":
// they parse individual objects and push them through the very same public
// PotFit methods. No privileged bulk path.

#include "potfit/api/potfit.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>

namespace potfit::io {

boost::leaf::result<void> load_configs(const std::filesystem::path &path,
                                       PotFit &session);

boost::leaf::result<void> load_model(const std::filesystem::path &path,
                                     PotFit &session);

} // namespace potfit::io
