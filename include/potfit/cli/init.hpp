#ifndef POTFIT_CLI_INIT_HPP
#define POTFIT_CLI_INIT_HPP

namespace potfit::cli::init {

// `potfit init …` — scaffold a fresh, loadable startpot for any model type
// (pair/eam/adp/angular/tersoff/stiweb/acsf/soap/lmbtr). Parses its own argv
// (argv[0] is "init"), writes the startpot JSON, and returns a process code.
// Self-contained: needs neither --config nor --startpot, so it is dispatched
// from main() before the fitting CLI's required-option parsing.
int run(int argc, char *argv[]);

} // namespace potfit::cli::init

#endif // POTFIT_CLI_INIT_HPP
