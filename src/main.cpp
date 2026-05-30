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
        ("help,h",                                         "show this message")
        ("config,c",   po::value<std::string>()->required(), "atomic configuration file")
        ("startpot,s", po::value<std::string>()->required(), "initial potential file")
        ("endpot,e",   po::value<std::string>()->required(), "output potential file")
        ("format,f",   po::value<std::string>()
                           ->default_value("native"),  "output format: native | lammps | imd")
        ("maxiter",    po::value<int>()->default_value(500), "max optimizer iterations")
        ("eweight",    po::value<double>()->default_value(1.0), "energy residual weight")
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
            // ── Step 1: parse config ─────────────────────────────────────────
            const std::string conf_text = read_file(vm["config"].as<std::string>());
            BOOST_LEAF_AUTO(configs_vec,
                            potfit::io::parse_config(conf_text));

            // ── Step 2: parse startpot ───────────────────────────────────────
            const std::string pot_text = read_file(vm["startpot"].as<std::string>());
            BOOST_LEAF_AUTO(potentials,
                            potfit::io::parse_potential(pot_text));

            if (configs_vec.empty()) {
                std::cerr << "error: configuration file contains no configurations\n";
                ret = 1;
                return {};
            }
            if (potentials.empty()) {
                std::cerr << "error: potential file contains no potentials\n";
                ret = 1;
                return {};
            }

            // ── Step 3: optimize ─────────────────────────────────────────────
            potfit::OptimizerOptions opts;
            opts.max_iter      = vm["maxiter"].as<int>();
            opts.energy_weight = vm["eweight"].as<double>();

            const int status = potfit::run_optimizer(configs_vec, potentials, opts);
            std::cout << "optimizer finished with status " << status << "\n";

            // ── Step 4: write endpot ─────────────────────────────────────────
            const std::string fmt  = vm["format"].as<std::string>();
            const std::string outp = vm["endpot"].as<std::string>();

            if (fmt == "lammps")
                potfit::io::write_lammps(outp, potentials);
            else if (fmt == "imd")
                potfit::io::write_imd(outp, potentials);
            else
                potfit::io::write_native(outp, potentials);

            std::cout << "wrote " << fmt << " potential to " << outp << "\n";
            return {};
        },
        [&](const potfit::io::ParseError& e) {
            std::cerr << "parse error (line " << e.line << "): " << e.message << "\n";
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
