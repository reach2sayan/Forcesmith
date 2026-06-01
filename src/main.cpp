#include "potfit/api/fit_session.hpp"
#include "potfit/core/checkpoint.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/io/config_reader.hpp"
#include "potfit/io/loaders.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace po = boost::program_options;
namespace leaf = boost::leaf;

int main(int argc, char *argv[]) {
  po::options_description desc("potfit — interatomic potential fitter");
  desc.add_options()("help,h", "show this message")(
      "config,c", po::value<std::string>()->required(),
      "atomic configuration file")("startpot,s",
                                   po::value<std::string>()->required(),
                                   "initial potential file")(
      "endpot,e", po::value<std::string>(),
      "output potential file (required unless --evaluate)")(
      "evaluate", po::value<std::string>(),
      "evaluate the start potential against the configs and write a "
      "per-config\n"
      "forces/energy/stress JSON report to the given file, then exit (no "
      "optimization)")("format,f",
                       po::value<std::string>()->default_value("native"),
                       "output format: native | lammps | imd")(
      "checkpoint,k", po::value<std::string>(),
      "checkpoint prefix: save after each run, resume if present")(
      "maxiter", po::value<int>()->default_value(500),
      "max optimizer iterations")("eweight",
                                  po::value<double>()->default_value(1.0),
                                  "energy residual weight")(
      "stress-weight", po::value<double>()->default_value(0.0),
      "stress tensor residual weight (0 = disabled)")(
      "smooth-weight", po::value<double>()->default_value(0.0),
      "curvature (Tikhonov) regularization weight on free knots (0 = "
      "disabled)")("algorithm,a", po::value<std::string>()->default_value("lm"),
                   "optimization algorithm: lm | powell (dogleg) | de | ls "
                   "(Powell direction-set line search)")(
      "seed", po::value<unsigned>()->default_value(0),
      "RNG seed for DE (0 = random_device)")(
      "de-F", po::value<double>()->default_value(0.65),
      "DE mutation factor F ∈ (0,1)")("de-CR",
                                      po::value<double>()->default_value(0.5),
                                      "DE crossover probability CR ∈ (0,1)")(
      "de-np", po::value<int>()->default_value(15),
      "DE population factor: NP = de-np × D")(
      "de-gen", po::value<int>()->default_value(1000),
      "DE maximum number of generations");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || argc == 1) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "error: " << e.what() << "\n\n" << desc << "\n";
    return 1;
  }

  // Progress logger.
  auto iter_conn = potfit::events::on_iteration.connect(
      [](const potfit::events::IterationStats &s) {
        std::cout << "iter " << s.iteration << "  obj=" << s.objective << "\n";
      });

  int ret = 0;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        // The CLI is a thin client of the FitSession API: build the session,
        // then evaluate / optimize / write through it.
        potfit::FitSession session;

        const bool has_checkpoint = vm.count("checkpoint") > 0;
        const std::string ckpt_prefix =
            has_checkpoint ? vm["checkpoint"].as<std::string>() : "";

        // ── Resume from checkpoint if it exists ──────────────────────────
        bool resumed = false;
        if (has_checkpoint) {
          std::vector<potfit::Configuration> cfgs;
          std::vector<potfit::Potential> pots;
          if (potfit::CheckpointReader(ckpt_prefix).read(cfgs, pots)) {
            for (auto &c : cfgs) {
              session.add_configuration(std::move(c));
            }
            BOOST_LEAF_CHECK(session.seed_force_model(
                potfit::make_pair_force_calculator(std::move(pots))));
            std::cout << "resumed from checkpoint " << ckpt_prefix << "\n";
            resumed = true;
          }
        }

        if (!resumed) {
          BOOST_LEAF_CHECK(
              potfit::io::load_configs(vm["config"].as<std::string>(), session));
          BOOST_LEAF_CHECK(potfit::io::load_model(
              vm["startpot"].as<std::string>(), session));
        }

        if (session.config_count() == 0) {
          std::cerr << "error: no configurations loaded\n";
          ret = 1;
          return {};
        }

        // ── Evaluate-only mode ───────────────────────────────────────────
        // Single force evaluation of the (unoptimized) start potential against
        // every configuration; dumps per-config computed vs reference forces /
        // energy / stress as JSON. Parity artifact vs the original C potfit.
        if (vm.count("evaluate")) {
          const std::string ev_path = vm["evaluate"].as<std::string>();
          const double ew = vm["eweight"].as<double>();
          const double sw = vm["stress-weight"].as<double>();

          BOOST_LEAF_AUTO(configs, session.configurations());

          std::ofstream out(ev_path);
          if (!out) {
            std::cerr << "error: cannot open evaluate output: " << ev_path
                      << "\n";
            ret = 1;
            return {};
          }
          out << std::setprecision(17);

          auto stress6 = [](const potfit::SymTens &s, std::ostream &o) {
            o << s(0, 0) << ", " << s(1, 1) << ", " << s(2, 2) << ", "
              << s(0, 1) << ", " << s(1, 2) << ", " << s(0, 2);
          };

          double total_sumsq = 0.0;
          out << "{\n  \"energy_weight\": " << ew
              << ",\n  \"stress_weight\": " << sw
              << ",\n  \"nconf\": " << configs.size()
              << ",\n  \"configs\": [\n";

          for (std::size_t i = 0; i < configs.size(); ++i) {
            const potfit::Configuration &cfg = configs[i];
            BOOST_LEAF_AUTO(r, session.evaluate(i));

            double csq = 0.0;
            out << "    {\n      \"index\": " << i
                << ",\n      \"name\": \"" << cfg.name << "\""
                << ",\n      \"natoms\": " << cfg.atoms.size()
                << ",\n      \"calc_energy\": " << r.energy
                << ",\n      \"ref_energy\": " << cfg.ref.energy
                << ",\n      \"calc_stress\": [";
            stress6(r.stress, out);
            out << "],\n      \"ref_stress\": [";
            stress6(cfg.ref.stress, out);
            out << "],\n      \"atoms\": [\n";
            for (std::size_t a = 0; a < cfg.atoms.size(); ++a) {
              const auto &cf = r.forces[a];
              const auto &rf = cfg.atoms[a].ref.force;
              for (int k = 0; k < 3; ++k) {
                const double d = cf[k] - rf[k];
                csq += d * d;
              }
              out << "        {\"calc_force\": [" << cf[0] << ", " << cf[1]
                  << ", " << cf[2] << "], \"ref_force\": [" << rf[0] << ", "
                  << rf[1] << ", " << rf[2] << "]}"
                  << (a + 1 < cfg.atoms.size() ? "," : "") << "\n";
            }
            const double de = ew * (r.energy - cfg.ref.energy);
            csq += de * de;
            if (sw > 0.0) {
              const potfit::SymTens ds = r.stress - cfg.ref.stress;
              const double comps[6] = {ds(0, 0), ds(1, 1), ds(2, 2),
                                       ds(0, 1), ds(1, 2), ds(0, 2)};
              for (double c : comps) {
                csq += (sw * c) * (sw * c);
              }
            }
            csq += r.limit * r.limit;
            total_sumsq += csq;

            out << "      ],\n      \"limit\": " << r.limit
                << ",\n      \"sumsq\": " << csq << "\n    }"
                << (i + 1 < configs.size() ? "," : "") << "\n";
          }

          out << "  ],\n  \"total_sumsq\": " << total_sumsq << "\n}\n";
          std::cout << "evaluated " << configs.size()
                    << " configurations, total error sum = "
                    << std::setprecision(10) << total_sumsq
                    << "\nwrote evaluation report to " << ev_path << "\n";
          return {};
        }

        if (!vm.count("endpot")) {
          std::cerr
              << "error: --endpot is required unless --evaluate is given\n";
          ret = 1;
          return {};
        }

        // ── Optimize ─────────────────────────────────────────────────────
        potfit::OptimizerOptions &opts = session.options();
        opts.max_iter = vm["maxiter"].as<int>();
        opts.energy_weight = vm["eweight"].as<double>();
        opts.stress_weight = vm["stress-weight"].as<double>();
        opts.smooth_weight = vm["smooth-weight"].as<double>();
        opts.seed = vm["seed"].as<unsigned>();
        opts.de.mutation_factor = vm["de-F"].as<double>();
        opts.de.crossover_probability = vm["de-CR"].as<double>();
        opts.de.NP_factor = static_cast<std::size_t>(vm["de-np"].as<int>());
        opts.de.max_generations =
            static_cast<std::size_t>(vm["de-gen"].as<int>());

        const std::string alg = vm["algorithm"].as<std::string>();
        if (alg == "powell") {
          opts.algorithm = potfit::Algorithm::Powell;
        } else if (alg == "de") {
          opts.algorithm = potfit::Algorithm::DE;
        } else if (alg == "ls") {
          opts.algorithm = potfit::Algorithm::LineSearch;
        } else if (alg != "lm") {
          std::cerr << "unknown algorithm '" << alg
                    << "'; choose: lm | powell | de | ls\n";
          ret = 1;
          return {};
        }

        BOOST_LEAF_AUTO(status, session.optimize());
        std::cout << "optimizer finished with status " << status << "\n";

        // ── Save checkpoint (pair model only) ─────────────────────────────
        if (has_checkpoint) {
          BOOST_LEAF_AUTO(configs, session.configurations());
          BOOST_LEAF_AUTO(model, session.model());
          std::vector<potfit::Potential> result_pots;
          std::visit(
              [&](const auto &calc) {
                using T = std::decay_t<decltype(calc)>;
                if constexpr (std::is_same_v<T, potfit::PairForceCalculator>) {
                  result_pots.assign(calc.pair.begin(), calc.pair.end());
                }
              },
              *model);
          const std::vector<potfit::Configuration> cfg_copy(configs.begin(),
                                                            configs.end());
          BOOST_LEAF_CHECK(potfit::CheckpointWriter(ckpt_prefix)
                               .configs(cfg_copy)
                               .potentials(result_pots)
                               .write());
          std::cout << "checkpoint saved to " << ckpt_prefix << "\n";
        }

        // ── Write endpot ─────────────────────────────────────────────────
        const std::string fmt = vm["format"].as<std::string>();
        const std::string outp = vm["endpot"].as<std::string>();
        BOOST_LEAF_CHECK(session.write(outp, fmt));
        std::cout << "wrote potential to " << outp << "\n";
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
      [&]() {
        std::cerr << "unknown error\n";
        ret = 1;
      });
  return ret;
}
