#include "forcesmith/cli/app.hpp"
#include "forcesmith/cli/init.hpp"
#include "forcesmith/cli/options.hpp"

#include <string_view>

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
