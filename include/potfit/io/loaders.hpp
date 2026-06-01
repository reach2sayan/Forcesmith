#pragma once

// Thin file-driven front-ends for the FitSession API. These are the only
// difference between "the user builds a fit by hand" and "the CLI loads files":
// they parse individual objects and push them through the very same public
// FitSession methods. No privileged bulk path.

#include "potfit/api/fit_session.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>

namespace potfit::io {

// Parse a config file (top-level JSON array of records) and feed each record to
// session.add_configuration(). Element symbols join the model at freeze.
boost::leaf::result<void> load_configs(const std::filesystem::path &path,
                                       FitSession &session);

// Parse a force-model file and seed the session with the resulting model
// (delegates to the battle-tested parse_force_model, then seed_force_model).
boost::leaf::result<void> load_model(const std::filesystem::path &path,
                                     FitSession &session);

} // namespace potfit::io
