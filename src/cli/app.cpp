#include "forcesmith/cli/app.hpp"
#include "forcesmith/cli/evaluate_report.hpp"
#include "forcesmith/cli/solver_registry.hpp"

#include "forcesmith/api/forcesmith.hpp"
#include "forcesmith/core/checkpoint.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/io/config_reader.hpp"
#include "forcesmith/io/loaders.hpp"
#include "forcesmith/io/logging.hpp"

#include "forcesmith/core/leaf_macros.hpp"
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace leaf = boost::leaf;

namespace forcesmith::cli {

int run(const CliOptions &o) {

  int ret = 0;
  forcesmith::log::init();
  auto log_sinks = forcesmith::log::connect_signals(); // kept alive for the run

  const auto report = [&](std::string_view prefix, std::string_view msg) {
    std::cerr << prefix << msg << "\n";
    ret = 1;
  };

  auto main_runner = [&]() -> leaf::result<void> {
    forcesmith::Forcesmith session;

    const bool has_checkpoint = o.checkpoint.has_value();
    const std::string ckpt_prefix =
        has_checkpoint ? *o.checkpoint : std::string{};

    bool resumed = false;
    if (has_checkpoint) {
      std::vector<forcesmith::Configuration> cfgs;
      forcesmith::ForceCalculator model;
      if (forcesmith::CheckpointReader(ckpt_prefix).read(cfgs, model)) {
        for (auto &c : cfgs) {
          session.add_configuration(std::move(c));
        }
        BOOST_LEAF_CHECK(session.seed_force_model(std::move(model)));
        std::cout << "resumed from checkpoint " << ckpt_prefix << "\n";
        resumed = true;
      }
    }

    if (!resumed) {
      BOOST_LEAF_CHECK(forcesmith::io::load_configs(o.config, session));
      BOOST_LEAF_CHECK(forcesmith::io::load_model(o.startpot, session));
    }

    if (session.config_count() == 0) {
      std::cerr << "error: no configurations loaded\n";
      ret = 1;
      return {};
    }

    if (o.evaluate.has_value()) {
      return write_evaluate_report(session, o);
    }

    if (!o.endpot.has_value()) {
      std::cerr << "error: --endpot is required unless --evaluate is given\n";
      ret = 1;
      return {};
    }

    forcesmith::OptimizerOptions &opts = session.options();
    opts.energy_weight = o.energy_weight;
    opts.stress_weight = o.stress_weight;
    opts.smooth_weight = o.smooth_weight;

    if (auto solver = build_solver(o)) {
      session.set_solver(std::move(*solver));
    } else {
      std::cerr << "unknown algorithm '" << o.algorithm
                << "'; choose: " << solver_name_list() << "\n";
      ret = 1;
      return {};
    }

    BOOST_LEAF_AUTO(status, session.optimize());
    std::cout << "optimizer finished with status " << status << "\n";

    if (has_checkpoint) {
      BOOST_LEAF_AUTO(configs, session.configurations());
      BOOST_LEAF_AUTO(model, session.model());
      const std::vector<forcesmith::Configuration> cfg_copy(configs.begin(),
                                                            configs.end());
      BOOST_LEAF_CHECK(forcesmith::CheckpointWriter(ckpt_prefix)
                           .configs(cfg_copy)
                           .model(*model)
                           .write());
      std::cout << "checkpoint saved to " << ckpt_prefix << "\n";
    }

    BOOST_LEAF_CHECK(session.write(*o.endpot, o.format));
    std::cout << "wrote potential to " << *o.endpot << "\n";
    return {};
  };

  leaf::try_handle_all(
      main_runner,
      [&](const forcesmith::io::ParseError &e) {
        report("parse error (line " + std::to_string(e.line) + "): ",
               e.message);
      },
      [&](const forcesmith::CheckpointError &e) {
        report("checkpoint error: ", e.message);
      },
      [&](const std::exception &e) { report("error: ", e.what()); },
      [&](const std::string &msg) { report("error: ", msg); },
      [&] { report("", "unknown error"); });
  return ret;
}

} // namespace forcesmith::cli
