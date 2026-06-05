#ifndef FORCESMITH_CLI_EVALUATE_REPORT_HPP
#define FORCESMITH_CLI_EVALUATE_REPORT_HPP

#include "forcesmith/cli/options.hpp"
#include "forcesmith/api/forcesmith.hpp"

#include <boost/leaf/result.hpp>

namespace forcesmith::cli {

// Evaluate-only mode: a single force evaluation of the (unoptimized) start
// potential against every configuration, dumping per-config computed vs
// reference forces / energy / stress as JSON to o.evaluate. Parity artifact vs
// the original C forcesmith. Caller guarantees o.evaluate has a value. A failure to
// open the output file is reported as a leaf ParseError.
boost::leaf::result<void> write_evaluate_report(forcesmith::Forcesmith &session,
                                                const CliOptions &o);

} // namespace forcesmith::cli

#endif // FORCESMITH_CLI_EVALUATE_REPORT_HPP
