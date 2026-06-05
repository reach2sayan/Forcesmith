#ifndef FORCESMITH_CLI_INIT_HPP
#define FORCESMITH_CLI_INIT_HPP

namespace forcesmith::cli::init {

// `forcesmith init …` — scaffold a fresh, loadable startpot for any model type
// (pair/eam/adp/angular/tersoff/stiweb/acsf/soap/lmbtr). Parses its own argv
// (argv[0] is "init"), writes the startpot JSON, and returns a process code.
// Self-contained: needs neither --config nor --startpot, so it is dispatched
// from main() before the fitting CLI's required-option parsing.
int run(int argc, char *argv[]);

} // namespace forcesmith::cli::init

#endif // FORCESMITH_CLI_INIT_HPP
