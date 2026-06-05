#ifndef POTFIT_CLI_OPTIONS_HPP
#define POTFIT_CLI_OPTIONS_HPP

#include <optional>
#include <string>

namespace potfit::cli {

// Parsed command-line options. A plain struct decoupled from the parser library
// so the rest of the CLI never touches boost::program_options.
struct CliOptions {
  std::string config;                  // --config,c  (required)
  std::string startpot;                // --startpot,s (required)
  std::optional<std::string> endpot;   // --endpot,e  (required unless evaluate)
  std::optional<std::string> evaluate; // --evaluate
  std::string format = "native";       // --format,f
  std::optional<std::string> checkpoint; // --checkpoint,k

  int max_iter = 500;           // --maxiter
  double energy_weight = 1.0;   // --eweight
  double stress_weight = 0.0;   // --stress-weight
  double smooth_weight = 0.0;   // --smooth-weight
  std::string algorithm = "lm"; // --algorithm,a

  unsigned seed = 42;  // --seed
  double de_F = 0.65; // --de-F
  double de_CR = 0.5; // --de-CR
  int de_np = 15;
  int de_gen = 1000;  // --de-gen
};

enum class ParseOutcome {
  Run,
  ExitOk,
  ExitError,
};

struct ParseResult {
  ParseOutcome outcome;
  CliOptions options;
};

ParseResult parse(int argc, char *argv[]);

} // namespace potfit::cli

#endif // POTFIT_CLI_OPTIONS_HPP
