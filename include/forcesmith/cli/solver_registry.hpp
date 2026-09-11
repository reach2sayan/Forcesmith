#ifndef FORCESMITH_CLI_SOLVER_REGISTRY_HPP
#define FORCESMITH_CLI_SOLVER_REGISTRY_HPP

#include "forcesmith/cli/options.hpp"
#include "forcesmith/optimization/solver.hpp"

#include <optional>
#include <string>

namespace forcesmith::cli {

std::optional<forcesmith::Solver> build_solver(const CliOptions &o);

std::string solver_help_text();

std::string solver_name_list();

} // namespace forcesmith::cli

#endif // FORCESMITH_CLI_SOLVER_REGISTRY_HPP
