#include "cli/app.hpp"
#include "cli/evaluate_report.hpp"

#include "potfit/api/potfit.hpp"
#include "potfit/core/checkpoint.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/io/config_reader.hpp"
#include "potfit/io/loaders.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace leaf = boost::leaf;

namespace potfit::cli {
namespace {

// Map the --algorithm string onto the API enum. Returns nullopt for an unknown
// name (the caller reports the error).
std::optional<potfit::Algorithm> select_algorithm(const std::string &alg) {
  if (alg == "lm") {
    return potfit::Algorithm::LM;
  }
  if (alg == "powell") {
    return potfit::Algorithm::Powell;
  }
  if (alg == "de") {
    return potfit::Algorithm::DE;
  }
  if (alg == "ls") {
    return potfit::Algorithm::LineSearch;
  }
  return std::nullopt;
}

} // namespace

int run(const CliOptions &o) {
  // Progress logger.
  auto iter_conn = potfit::events::on_iteration.connect(
      [](const potfit::events::IterationStats &s) {
        std::cout << "iter " << s.iteration << "  obj=" << s.objective << "\n";
      });

  int ret = 0;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        // The CLI is a thin client of the PotFit API: build the session,
        // then evaluate / optimize / write through it.
        potfit::PotFit session;

        const bool has_checkpoint = o.checkpoint.has_value();
        const std::string ckpt_prefix =
            has_checkpoint ? *o.checkpoint : std::string{};

        // ── Resume from checkpoint if it exists ──────────────────────────
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

        // ── Evaluate-only mode ───────────────────────────────────────────
        if (o.evaluate.has_value()) {
          return write_evaluate_report(session, o, ret);
        }

        if (!o.endpot.has_value()) {
          std::cerr
              << "error: --endpot is required unless --evaluate is given\n";
          ret = 1;
          return {};
        }

        // ── Optimize ─────────────────────────────────────────────────────
        potfit::OptimizerOptions &opts = session.options();
        opts.max_iter = o.max_iter;
        opts.energy_weight = o.energy_weight;
        opts.stress_weight = o.stress_weight;
        opts.smooth_weight = o.smooth_weight;
        opts.seed = o.seed;
        opts.de.mutation_factor = o.de_F;
        opts.de.crossover_probability = o.de_CR;
        opts.de.NP_factor = static_cast<std::size_t>(o.de_np);
        opts.de.max_generations = static_cast<std::size_t>(o.de_gen);

        if (auto alg = select_algorithm(o.algorithm)) {
          opts.algorithm = *alg;
        } else {
          std::cerr << "unknown algorithm '" << o.algorithm
                    << "'; choose: lm | powell | de | ls\n";
          ret = 1;
          return {};
        }

        BOOST_LEAF_AUTO(status, session.optimize());
        std::cout << "optimizer finished with status " << status << "\n";

        // ── Save checkpoint (any force-model family) ──────────────────────
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

        // ── Write endpot ─────────────────────────────────────────────────
        BOOST_LEAF_CHECK(session.write(*o.endpot, o.format));
        std::cout << "wrote potential to " << *o.endpot << "\n";
        return {};
      },
      [&](const potfit::io::ParseError &e) {
        std::cerr << "parse error (line " << e.line << "): " << e.message
                  << "\n";
        ret = 1;
      },
      [&](const potfit::CheckpointError &e) {
        std::cerr << "checkpoint error: " << e.message << "\n";
        ret = 1;
      },
      [&](const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        ret = 1;
      },
      // Some subsystems (e.g. the species registry) raise leaf errors carrying a
      // plain std::string message rather than a typed payload.
      [&](const std::string &msg) {
        std::cerr << "error: " << msg << "\n";
        ret = 1;
      },
      [&]() {
        std::cerr << "unknown error\n";
        ret = 1;
      });
  return ret;
}

} // namespace potfit::cli
