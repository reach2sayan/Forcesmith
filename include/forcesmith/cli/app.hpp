#ifndef FORCESMITH_CLI_APP_HPP
#define FORCESMITH_CLI_APP_HPP

#include "forcesmith/cli/options.hpp"

namespace forcesmith::cli {

// Drive the Forcesmith API from parsed CLI options: build the session, resume or
// load configs/model, then evaluate / optimize / write through it. Owns the
// structured (boost::leaf) error handling. Returns the process exit code.
int run(const CliOptions &o);

} // namespace forcesmith::cli

#endif // FORCESMITH_CLI_APP_HPP
