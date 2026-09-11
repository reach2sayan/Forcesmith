#ifndef FORCESMITH_CLI_EVALUATE_REPORT_HPP
#define FORCESMITH_CLI_EVALUATE_REPORT_HPP

#include "forcesmith/cli/options.hpp"
#include "forcesmith/api/forcesmith.hpp"

#include <boost/leaf/result.hpp>

namespace forcesmith::cli {

boost::leaf::result<void> write_evaluate_report(forcesmith::Forcesmith &session,
                                                const CliOptions &o);

} // namespace forcesmith::cli

#endif // FORCESMITH_CLI_EVALUATE_REPORT_HPP
