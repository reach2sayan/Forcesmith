#ifndef POTFIT_CLI_APP_HPP
#define POTFIT_CLI_APP_HPP

#include "potfit/cli/options.hpp"

namespace potfit::cli {

// Drive the PotFit API from parsed CLI options: build the session, resume or
// load configs/model, then evaluate / optimize / write through it. Owns the
// structured (boost::leaf) error handling. Returns the process exit code.
int run(const CliOptions &o);

} // namespace potfit::cli

#endif // POTFIT_CLI_APP_HPP
