#include "potfit/core/checkpoint.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/io/config_reader.hpp"
#include "potfit/io/output_writer.hpp"
#include "potfit/io/potential_reader.hpp"
#include "potfit/optimization/optimizer.hpp"
#include "potfit/events/signals.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <string>

namespace po   = boost::program_options;
namespace leaf = boost::leaf;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char* argv[]) {
    po::options_description desc("potfit — interatomic potential fitter");
    desc.add_options()
        ("help,h",                                              "show this message")
        ("config,c",      po::value<std::string>()->required(), "atomic configuration file")
        ("startpot,s",    po::value<std::string>()->required(), "initial potential file")
        ("endpot,e",      po::value<std::string>()->required(), "output potential file")
        ("format,f",      po::value<std::string>()
                              ->default_value("native"), "output format: native | lammps | imd")
        ("checkpoint,k",  po::value<std::string>(),
                              "checkpoint prefix: save after each run, resume if present")
        ("maxiter",       po::value<int>()->default_value(500),    "max optimizer iterations")
        ("eweight",       po::value<double>()->default_value(1.0), "energy residual weight")
        ("stress-weight", po::value<double>()->default_value(0.0), "stress tensor residual weight (0 = disabled)")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help") || argc == 1) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "error: " << e.what() << "\n\n" << desc << "\n";
        return 1;
    }

    // Progress logger.
    auto iter_conn = potfit::events::on_iteration.connect(
        [](const potfit::events::IterationStats& s) {
            std::cout << "iter " << s.iteration
                      << "  obj=" << s.objective << "\n";
        });

    int ret = 0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            std::vector<potfit::Configuration> configs_vec;
            potfit::ForceCalculator model = potfit::PairForceCalculator{};

            const bool has_checkpoint = vm.count("checkpoint") > 0;
            const std::string ckpt_prefix = has_checkpoint
                ? vm["checkpoint"].as<std::string>() : "";

            // ── Resume from checkpoint if it exists ──────────────────────────
            bool resumed = false;
            if (has_checkpoint) {
                std::vector<potfit::Potential> pots;
                auto r = potfit::CheckpointReader(ckpt_prefix)
                             .read(configs_vec, pots);
                if (r) {
                    model = potfit::make_pair_force_calculator(std::move(pots));
                    std::cout << "resumed from checkpoint " << ckpt_prefix << "\n";
                    resumed = true;
                }
                // If load fails, fall through to normal parse (checkpoint absent).
            }

            if (!resumed) {
                // ── Parse config ─────────────────────────────────────────────
                const std::string conf_text = read_file(vm["config"].as<std::string>());
                auto r_cfg = potfit::io::parse_config(conf_text);
                if (!r_cfg) return r_cfg.error();
                configs_vec = std::move(*r_cfg);

                // ── Parse startpot ───────────────────────────────────────────
                const std::string pot_text = read_file(vm["startpot"].as<std::string>());
                auto r_pot = potfit::io::parse_potential(pot_text);
                if (!r_pot) return r_pot.error();
                model = potfit::make_pair_force_calculator(std::move(*r_pot));
            }

            if (configs_vec.empty()) {
                std::cerr << "error: no configurations loaded\n";
                ret = 1; return {};
            }
            if (std::visit([](const auto& m){ return m.param_count(); }, model) == 0) {
                std::cerr << "error: no potentials loaded\n";
                ret = 1; return {};
            }

            // ── Optimize ─────────────────────────────────────────────────────
            potfit::OptimizerOptions opts;
            opts.max_iter      = vm["maxiter"].as<int>();
            opts.energy_weight = vm["eweight"].as<double>();
            opts.stress_weight = vm["stress-weight"].as<double>();

            const int status = potfit::run_optimizer(configs_vec, model, opts);
            std::cout << "optimizer finished with status " << status << "\n";

            // Extract flat potentials for I/O and checkpointing.
            const auto& pair_calc = std::get<potfit::PairForceCalculator>(model);
            std::vector<potfit::Potential> result_pots(
                pair_calc.pair.begin(), pair_calc.pair.end());

            // ── Save checkpoint ───────────────────────────────────────────────
            if (has_checkpoint) {
                if (auto r = potfit::CheckpointWriter(ckpt_prefix)
                                 .configs(configs_vec)
                                 .potentials(result_pots)
                                 .write(); !r)
                    return r.error();
                std::cout << "checkpoint saved to " << ckpt_prefix << "\n";
            }

            // ── Write endpot ─────────────────────────────────────────────────
            const std::string fmt  = vm["format"].as<std::string>();
            const std::string outp = vm["endpot"].as<std::string>();

            if (fmt == "lammps")
                potfit::io::write_lammps(outp, result_pots);
            else if (fmt == "imd")
                potfit::io::write_imd(outp, result_pots);
            else
                potfit::io::write_native(outp, result_pots);

            std::cout << "wrote " << fmt << " potential to " << outp << "\n";
            return {};
        },
        [&](const potfit::io::ParseError& e) {
            std::cerr << "parse error (line " << e.line << "): " << e.message << "\n";
            ret = 1;
        },
        [&](const potfit::CheckpointError& e) {
            std::cerr << "checkpoint error: " << e.message << "\n";
            ret = 1;
        },
        [&](const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            ret = 1;
        },
        [&]() {
            std::cerr << "unknown error\n";
            ret = 1;
        }
    );
    return ret;
}
