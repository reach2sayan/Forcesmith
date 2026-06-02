#include "cli/app.hpp"
#include "cli/options.hpp"

// Entry point. All work lives in the cli/ module: parse the arguments, then —
// unless help was shown or parsing failed — drive the PotFit API via cli::run.
int main(int argc, char *argv[]) {
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
