#include "potfit/cli/app.hpp"
#include "potfit/cli/evaluate_report.hpp"

#include "potfit/api/potfit.hpp"
#include "potfit/core/checkpoint.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/io/config_reader.hpp"
#include "potfit/io/loaders.hpp"
#include "potfit/optimization/ipopt_solver.hpp"
#include "potfit/optimization/solver.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace leaf = boost::leaf;

// BOOST_LEAF_CHECK expands to a GNU statement-expression ({ ... }); silence the
// pedantic complaint about that Boost idiom for this translation unit.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

namespace potfit::cli {
namespace {

// Build the configured solver named by --algorithm. The solver carries its own
// tuning (iteration cap, DE parameters, seed), so the choice is materialised
// here as a Solver and injected into the session. nullopt → unknown name.
std::optional<potfit::Solver> build_solver(const CliOptions &o) {
  if (o.algorithm == "lm") {
    return potfit::Solver{potfit::EigenLMSolver{o.max_iter}};
  } else if (o.algorithm == "powell") {
    return potfit::Solver{potfit::EigenHybridSolver{o.max_iter}};
  } else if (o.algorithm == "ls") {
    return potfit::Solver{potfit::LineSearchSolver{o.max_iter}};
  } else if (o.algorithm == "ipopt") {
    return potfit::Solver{potfit::IpoptSolver{o.max_iter}};
  } else if (o.algorithm == "de") {
    potfit::BoostDESolver s;
    s.mutation_factor = o.de_F;
    s.crossover_probability = o.de_CR;
    s.NP_factor = static_cast<std::size_t>(o.de_np);
    s.max_generations = static_cast<std::size_t>(o.de_gen);
    s.seed = o.seed;
    return potfit::Solver{std::move(s)};
  }
  return std::nullopt;
}

} // namespace

int run(const CliOptions &o) {

  int ret = 0;
  auto iter_conn = potfit::events::on_iteration.connect(
      [](const potfit::events::IterationStats &s) {
        std::cout << "iter " << s.iteration << "  obj=" << s.objective << "\n";
      });

  auto checkpoint_error_fn = [&](const potfit::CheckpointError &e) {
    std::cerr << "checkpoint error: " << e.message << "\n";
    ret = 1;
  };

  auto parse_error_fn = [&](const potfit::io::ParseError &e) {
    std::cerr << "parse error (line " << e.line << "): " << e.message << "\n";
    ret = 1;
  };

  auto default_exception_fn = [&](const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    ret = 1;
  };

  auto species_registry_error_fn = [&](const std::string &msg) {
    std::cerr << "error: " << msg << "\n";
    ret = 1;
  };

  auto unknown_error_fn = [&]() {
    std::cerr << "unknown error\n";
    ret = 1;
  };

  auto main_runner = [&]() -> leaf::result<void> {
    potfit::PotFit session;

    const bool has_checkpoint = o.checkpoint.has_value();
    const std::string ckpt_prefix =
        has_checkpoint ? *o.checkpoint : std::string{};

    bool resumed = false;
    if (has_checkpoint) {
      std::vector<potfit::Configuration> cfgs;
      potfit::ForceCalculator model;
      if (potfit::CheckpointReader(ckpt_prefix).read(cfgs, model)) {
        for (auto &c : cfgs) {
          session.add_configuration(std::move(c));
        }
        BOOST_LEAF_CHECK(session.seed_force_model(std::move(model)));
        std::cout << "resumed from checkpoint " << ckpt_prefix << "\n";
        resumed = true;
      }
    }

    if (!resumed) {
      BOOST_LEAF_CHECK(potfit::io::load_configs(o.config, session));
      BOOST_LEAF_CHECK(potfit::io::load_model(o.startpot, session));
    }

    if (session.config_count() == 0) {
      std::cerr << "error: no configurations loaded\n";
      ret = 1;
      return {};
    }

    if (o.evaluate.has_value()) {
      return write_evaluate_report(session, o, ret);
    }

    if (!o.endpot.has_value()) {
      std::cerr << "error: --endpot is required unless --evaluate is given\n";
      ret = 1;
      return {};
    }

    potfit::OptimizerOptions &opts = session.options();
    opts.energy_weight = o.energy_weight;
    opts.stress_weight = o.stress_weight;
    opts.smooth_weight = o.smooth_weight;

    if (auto solver = build_solver(o)) {
      session.set_solver(std::move(*solver));
    } else {
      std::cerr << "unknown algorithm '" << o.algorithm
                << "'; choose: lm | powell | de | ls | ipopt\n";
      ret = 1;
      return {};
    }

    BOOST_LEAF_AUTO(status, session.optimize());
    std::cout << "optimizer finished with status " << status << "\n";

    if (has_checkpoint) {
      BOOST_LEAF_AUTO(configs, session.configurations());
      BOOST_LEAF_AUTO(model, session.model());
      const std::vector<potfit::Configuration> cfg_copy(configs.begin(),
                                                        configs.end());
      BOOST_LEAF_CHECK(potfit::CheckpointWriter(ckpt_prefix)
                           .configs(cfg_copy)
                           .model(*model)
                           .write());
      std::cout << "checkpoint saved to " << ckpt_prefix << "\n";
    }

    BOOST_LEAF_CHECK(session.write(*o.endpot, o.format));
    std::cout << "wrote potential to " << *o.endpot << "\n";
    return {};
  };

  leaf::try_handle_all(main_runner, parse_error_fn, checkpoint_error_fn,
                       default_exception_fn, species_registry_error_fn,
                       unknown_error_fn);
  return ret;
}

} // namespace potfit::cli

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
