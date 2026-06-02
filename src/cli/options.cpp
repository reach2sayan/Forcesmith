#include "potfit/cli/options.hpp"

#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

namespace potfit::cli {

ParseResult parse(int argc, char *argv[]) {
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
      return {ParseOutcome::ExitOk, {}};
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "error: " << e.what() << "\n\n" << desc << "\n";
    return {ParseOutcome::ExitError, {}};
  }

  CliOptions o;
  o.config = vm["config"].as<std::string>();
  o.startpot = vm["startpot"].as<std::string>();
  if (vm.count("endpot")) {
    o.endpot = vm["endpot"].as<std::string>();
  }
  if (vm.count("evaluate")) {
    o.evaluate = vm["evaluate"].as<std::string>();
  }
  o.format = vm["format"].as<std::string>();
  if (vm.count("checkpoint")) {
    o.checkpoint = vm["checkpoint"].as<std::string>();
  }
  o.max_iter = vm["maxiter"].as<int>();
  o.energy_weight = vm["eweight"].as<double>();
  o.stress_weight = vm["stress-weight"].as<double>();
  o.smooth_weight = vm["smooth-weight"].as<double>();
  o.algorithm = vm["algorithm"].as<std::string>();
  o.seed = vm["seed"].as<unsigned>();
  o.de_F = vm["de-F"].as<double>();
  o.de_CR = vm["de-CR"].as<double>();
  o.de_np = vm["de-np"].as<int>();
  o.de_gen = vm["de-gen"].as<int>();

  return {ParseOutcome::Run, std::move(o)};
}

} // namespace potfit::cli
