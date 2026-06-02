#ifndef POTFIT_CLI_EVALUATE_REPORT_HPP
#define POTFIT_CLI_EVALUATE_REPORT_HPP

#include "potfit/cli/options.hpp"
#include "potfit/api/potfit.hpp"

#include <boost/leaf/result.hpp>

namespace potfit::cli {

// Evaluate-only mode: a single force evaluation of the (unoptimized) start
// potential against every configuration, dumping per-config computed vs
// reference forces / energy / stress as JSON to o.evaluate. Parity artifact vs
// the original C potfit. Caller guarantees o.evaluate has a value. On a failure
// to open the output file, prints to stderr and sets exit_code to 1.
boost::leaf::result<void> write_evaluate_report(potfit::PotFit &session,
                                                const CliOptions &o,
                                                int &exit_code);

} // namespace potfit::cli

#endif // POTFIT_CLI_EVALUATE_REPORT_HPP
