#include "forcesmith/cli/solver_registry.hpp"

#include "forcesmith/optimization/ipopt_solver.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace forcesmith::cli {
namespace {

struct SolverEntry {
  std::string_view name;
  std::string_view help;
  forcesmith::Solver (*make)(const CliOptions &);
  bool hidden = false;
};

constexpr std::array kSolvers{
    SolverEntry{"lm", "Levenberg-Marquardt",
                [](const CliOptions &o) {
                  return forcesmith::Solver{
                      forcesmith::EigenLMSolver{o.max_iter}};
                }},
    SolverEntry{"powell", "dogleg",
                [](const CliOptions &o) {
                  return forcesmith::Solver{
                      forcesmith::EigenHybridSolver{o.max_iter}};
                }},
    SolverEntry{"de", "differential evolution",
                [](const CliOptions &o) {
                  forcesmith::BoostDESolver s;
                  s.mutation_factor = o.de_F;
                  s.crossover_probability = o.de_CR;
                  s.NP_factor = static_cast<std::size_t>(o.de_np);
                  s.max_generations = static_cast<std::size_t>(o.de_gen);
                  s.seed = o.seed;
                  return forcesmith::Solver{std::move(s)};
                }},
    SolverEntry{"ls", "Powell direction-set line search",
                [](const CliOptions &o) {
                  return forcesmith::Solver{
                      forcesmith::LineSearchSolver{o.max_iter}};
                }},
    SolverEntry{"ipopt", "L-BFGS",
                [](const CliOptions &o) {
                  return forcesmith::Solver{
                      forcesmith::IpoptSolver{o.max_iter}};
                }},
    SolverEntry{"lsq",
                "closed-form least squares; exact + low-memory for linear ML "
                "heads",
                [](const CliOptions &o) {
                  forcesmith::NormalEquationsSolver s;
                  s.ridge = o.smooth_weight;
                  return forcesmith::Solver{std::move(s)};
                }},
    SolverEntry{"linear", "alias of lsq",
                [](const CliOptions &o) {
                  forcesmith::NormalEquationsSolver s;
                  s.ridge = o.smooth_weight;
                  return forcesmith::Solver{std::move(s)};
                },
                /*hidden=*/true},
};

} // namespace

std::optional<forcesmith::Solver> build_solver(const CliOptions &o) {
  const auto it = std::ranges::find(kSolvers, o.algorithm, &SolverEntry::name);
  if (it == kSolvers.end()) {
    return std::nullopt;
  }
  return it->make(o);
}

std::string solver_help_text() {
  std::string s;
  for (const SolverEntry &e : kSolvers) {
    if (e.hidden) {
      continue;
    }
    if (!s.empty()) {
      s += " | ";
    }
    s += e.name;
    s += " (";
    s += e.help;
    s += ")";
  }
  return s;
}

std::string solver_name_list() {
  std::string s;
  for (const SolverEntry &e : kSolvers) {
    if (e.hidden) {
      continue;
    }
    if (!s.empty()) {
      s += " | ";
    }
    s += e.name;
  }
  return s;
}

} // namespace forcesmith::cli
