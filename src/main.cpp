#include "forcesmith/cli/app.hpp"
#include "forcesmith/cli/init.hpp"
#include "forcesmith/cli/options.hpp"

#include <string_view>

// Entry point. All work lives in the cli/ module. The `init` subcommand
// scaffolds a fresh startpot and needs neither --config nor --startpot, so it
// is dispatched here before the fitting CLI's required-option parsing. Anything
// else falls through to: parse the arguments, then — unless help was shown or
// parsing failed — drive the Forcesmith API via cli::run.
int main(int argc, char *argv[]) {
  if (argc > 1 && std::string_view{argv[1]} == "init") {
    return forcesmith::cli::init::run(argc - 1, argv + 1);
  }

  const forcesmith::cli::ParseResult parsed = forcesmith::cli::parse(argc, argv);
  switch (parsed.outcome) {
  case forcesmith::cli::ParseOutcome::ExitOk:
    return 0;
  case forcesmith::cli::ParseOutcome::ExitError:
    return 1;
  case forcesmith::cli::ParseOutcome::Run:
    return forcesmith::cli::run(parsed.options);
  }
  return 0;
}
