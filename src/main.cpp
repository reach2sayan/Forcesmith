#include "potfit/core/checkpoint.hpp"
#include "potfit/core/config_index.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/evaluate.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/io/config_reader.hpp"
#include "potfit/io/force_model_reader.hpp"
#include "potfit/io/output_writer.hpp"
#include "potfit/optimization/optimizer.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace po = boost::program_options;
namespace leaf = boost::leaf;

static std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot open: " + path);
  return std::string(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
}

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
        std::vector<potfit::Configuration> configs_vec;
        potfit::ForceCalculator model = potfit::PairForceCalculator{};

        const bool has_checkpoint = vm.count("checkpoint") > 0;
        const std::string ckpt_prefix =
            has_checkpoint ? vm["checkpoint"].as<std::string>() : "";

        // ── Resume from checkpoint if it exists ──────────────────────────
        bool resumed = false;
        if (has_checkpoint) {
          std::vector<potfit::Potential> pots;
          auto r =
              potfit::CheckpointReader(ckpt_prefix).read(configs_vec, pots);
          if (r) {
            model = potfit::make_pair_force_calculator(std::move(pots));
            std::cout << "resumed from checkpoint " << ckpt_prefix << "\n";
            resumed = true;
          }
          // If load fails, fall through to normal parse (checkpoint absent).
        }

        if (!resumed) {
          // ── Parse config ─────────────────────────────────────────────
          const std::string conf_text =
              read_file(vm["config"].as<std::string>());
          auto r_cfg = potfit::io::parse_config(conf_text);
          if (!r_cfg)
            return r_cfg.error();
          configs_vec = std::move(*r_cfg);

          // ── Parse startpot ───────────────────────────────────────────
          const std::string pot_text =
              read_file(vm["startpot"].as<std::string>());
          auto r_pot = potfit::io::parse_force_model(pot_text);
          if (!r_pot)
            return r_pot.error();
          model = std::move(*r_pot);
        }

        if (configs_vec.empty()) {
          std::cerr << "error: no configurations loaded\n";
          ret = 1;
          return {};
        }

        // Auxiliary grouping index over configs (composition / energy / weight).
        // Derived data, rebuilt from configs_vec; configs_vec must not be resized
        // or reordered after this point (see config_index.hpp invariant).
        const auto config_idx =
            potfit::config_index::build_config_index(configs_vec);
        (void)config_idx;
        if (std::visit([](const auto &m) { return m.param_count(); }, model) ==
            0) {
          std::cerr << "error: no potentials loaded\n";
          ret = 1;
          return {};
        }

        // ── Evaluate-only mode ───────────────────────────────────────────
        // Run a single force evaluation of the (unoptimized) start potential
        // against every configuration and dump per-config computed vs.
        // reference forces / energy / stress as JSON. This is the parity
        // artifact diffed against the original C potfit's evaluate run
        // (param `opt 0`). No optimization is performed.
        if (vm.count("evaluate")) {
          const std::string ev_path = vm["evaluate"].as<std::string>();
          const double ew = vm["eweight"].as<double>();
          const double sw = vm["stress-weight"].as<double>();

          // Demonstrates the enriched on_force_eval callback: the slot
          // reads computed forces straight off s.cfg as each config is
          // evaluated. The potential is captured (model) — the per-config
          // fire site cannot see the type-erased ForceCalculator itself.
          std::uint64_t fired = 0;
          auto ev_conn = potfit::events::on_force_eval.connect(
              [&](const potfit::events::ForceEvalStats &s) {
                (void)s;
                ++fired;
              });

          std::ofstream out(ev_path);
          if (!out) {
            std::cerr << "error: cannot open evaluate output: " << ev_path
                      << "\n";
            ret = 1;
            return {};
          }
          out << std::setprecision(17);

          // SymTens (3x3 symmetric) → potfit's 6-vector order: xx yy zz xy yz
          // zx.
          auto stress6 = [](const potfit::SymTens &s, std::ostream &o) {
            o << s(0, 0) << ", " << s(1, 1) << ", " << s(2, 2) << ", "
              << s(0, 1) << ", " << s(1, 2) << ", " << s(0, 2);
          };

          double total_sumsq = 0.0;
          out << "{\n  \"energy_weight\": " << ew
              << ",\n  \"stress_weight\": " << sw
              << ",\n  \"nconf\": " << configs_vec.size()
              << ",\n  \"configs\": [\n";

          for (std::size_t i = 0; i < configs_vec.size(); ++i) {
            const potfit::Configuration &cfg = configs_vec[i];
            const auto r = potfit::force::evaluate(model, cfg);

            double csq = 0.0; // this config's contribution to total_sumsq
            out << "    {\n      \"index\": " << i
                << ",\n      \"natoms\": " << cfg.atoms.size()
                << ",\n      \"calc_energy\": " << r.energy
                << ",\n      \"ref_energy\": " << cfg.energy
                << ",\n      \"calc_stress\": [";
            stress6(r.stress, out);
            out << "],\n"
                << "      \"ref_stress\": [";
            stress6(cfg.stress, out);
            out << "],\n"
                << "      \"atoms\": [\n";
            for (std::size_t a = 0; a < cfg.atoms.size(); ++a) {
              const auto &cf = r.forces[a];
              const auto &rf = cfg.atoms[a].force;
              for (int k = 0; k < 3; ++k) {
                const double d = cf[k] - rf[k];
                csq += d * d;
              }
              out << "        {\"calc_force\": [" << cf[0] << ", " << cf[1]
                  << ", " << cf[2] << "], \"ref_force\": [" << rf[0] << ", "
                  << rf[1] << ", " << rf[2] << "]}"
                  << (a + 1 < cfg.atoms.size() ? "," : "") << "\n";
            }
            const double de = ew * (r.energy - cfg.energy);
            csq += de * de;
            if (sw > 0.0) {
              const potfit::SymTens ds = r.stress - cfg.stress;
              const double comps[6] = {ds(0, 0), ds(1, 1), ds(2, 2),
                                       ds(0, 1), ds(1, 2), ds(0, 2)};
              for (double c : comps)
                csq += (sw * c) * (sw * c);
            }
            csq += r.limit * r.limit;
            total_sumsq += csq;

            out << "      ],\n      \"limit\": " << r.limit
                << ",\n      \"sumsq\": " << csq << "\n    }"
                << (i + 1 < configs_vec.size() ? "," : "") << "\n";
          }

          out << "  ],\n  \"total_sumsq\": " << total_sumsq << "\n}\n";
          ev_conn.disconnect();
          std::cout << "evaluated " << configs_vec.size() << " configurations ("
                    << fired << " force-eval events), total error sum = "
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
        potfit::OptimizerOptions opts;
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
        if (alg == "powell")
          opts.algorithm = potfit::Algorithm::Powell;
        else if (alg == "de")
          opts.algorithm = potfit::Algorithm::DE;
        else if (alg == "ls")
          opts.algorithm = potfit::Algorithm::LineSearch;
        else if (alg != "lm") {
          std::cerr << "unknown algorithm '" << alg
                    << "'; choose: lm | powell | de | ls\n";
          ret = 1;
          return {};
        }

        const int status = potfit::run_optimizer(configs_vec, model, opts);
        std::cout << "optimizer finished with status " << status << "\n";

        // Extract flat potentials for I/O and checkpointing (pair model only).
        std::vector<potfit::Potential> result_pots;
        std::visit(
            [&](const auto &calc) {
              using T = std::decay_t<decltype(calc)>;
              if constexpr (std::is_same_v<T, potfit::PairForceCalculator>)
                result_pots.assign(calc.pair.begin(), calc.pair.end());
            },
            model);

        // ── Save checkpoint ───────────────────────────────────────────────
        if (has_checkpoint) {
          if (auto r = potfit::CheckpointWriter(ckpt_prefix)
                           .configs(configs_vec)
                           .potentials(result_pots)
                           .write();
              !r)
            return r.error();
          std::cout << "checkpoint saved to " << ckpt_prefix << "\n";
        }

        // ── Write endpot ─────────────────────────────────────────────────
        const std::string fmt = vm["format"].as<std::string>();
        const std::string outp = vm["endpot"].as<std::string>();

        const bool wrote = std::visit(
            [&](const auto &calc) -> bool {
              using T = std::decay_t<decltype(calc)>;
              if constexpr (std::is_same_v<T, potfit::PairForceCalculator>) {
                const std::vector<potfit::Potential> pots(calc.pair.begin(),
                                                          calc.pair.end());
                if (fmt == "lammps")
                  potfit::io::write_lammps(outp, pots);
                else if (fmt == "imd")
                  potfit::io::write_imd(outp, pots);
                else
                  potfit::io::write_native(outp, pots);
                return true;
              } else if constexpr (std::is_same_v<T,
                                                  potfit::EAMForceCalculator>) {
                if (fmt != "native")
                  std::cerr
                      << "warning: '" << fmt
                      << "' output unsupported for EAM; writing native JSON\n";
                potfit::io::write_native_eam(outp, calc);
                return true;
              } else if constexpr (std::is_same_v<T,
                                                  potfit::ADPForceCalculator>) {
                if (fmt != "native")
                  std::cerr
                      << "warning: '" << fmt
                      << "' output unsupported for ADP; writing native JSON\n";
                potfit::io::write_native_adp(outp, calc);
                return true;
              } else if constexpr (std::is_same_v<
                                       T, potfit::AngularForceCalculator>) {
                if (fmt != "native")
                  std::cerr << "warning: '" << fmt
                            << "' output unsupported for angular; writing "
                               "native JSON\n";
                potfit::io::write_native_angular(outp, calc);
                return true;
              } else if constexpr (std::is_same_v<
                                       T, potfit::TersoffForceCalculator>) {
                if (fmt != "native")
                  std::cerr << "warning: '" << fmt
                            << "' output unsupported for tersoff; writing "
                               "native JSON\n";
                potfit::io::write_native_tersoff(outp, calc);
                return true;
              } else if constexpr (std::is_same_v<
                                       T, potfit::StiwebForceCalculator>) {
                if (fmt != "native")
                  std::cerr << "warning: '" << fmt
                            << "' output unsupported for stiweb; writing "
                               "native JSON\n";
                potfit::io::write_native_stiweb(outp, calc);
                return true;
              } else {
                return false; // unknown model — no writer
              }
            },
            model);

        if (wrote)
          std::cout << "wrote potential to " << outp << "\n";
        else
          std::cout << "output not yet implemented for this model; "
                       "skipping endpot write\n";
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
