#include "potfit/cli/app.hpp"
#include "potfit/cli/init.hpp"
#include "potfit/cli/options.hpp"

#include <string_view>

// Entry point. All work lives in the cli/ module. The `init` subcommand
// scaffolds a fresh startpot and needs neither --config nor --startpot, so it
// is dispatched here before the fitting CLI's required-option parsing. Anything
// else falls through to: parse the arguments, then — unless help was shown or
// parsing failed — drive the PotFit API via cli::run.
int main(int argc, char *argv[]) {
  if (argc > 1 && std::string_view{argv[1]} == "init") {
    return potfit::cli::init::run(argc - 1, argv + 1);
  }

  const potfit::cli::ParseResult parsed = potfit::cli::parse(argc, argv);
  switch (parsed.outcome) {
  case potfit::cli::ParseOutcome::ExitOk:
    return 0;
  case potfit::cli::ParseOutcome::ExitError:
    return 1;
  case potfit::cli::ParseOutcome::Run:
    return potfit::cli::run(parsed.options);
  }
  return 0;
}
