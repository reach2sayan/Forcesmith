#include "forcesmith/cli/options.hpp"

#include "forcesmith/cli/solver_registry.hpp"

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

namespace po = boost::program_options;

namespace forcesmith::cli {

ParseResult parse(int argc, char *argv[]) {
  CliOptions o;
  std::string endpot, evaluate, checkpoint;

  const std::string algorithm_help =
      "optimization algorithm: " + solver_help_text();

  po::options_description desc("forcesmith — interatomic potential fitter");
  desc.add_options()("help,h", "show this message")(
      "config,c", po::value(&o.config)->required(),
      "atomic configuration file")("startpot,s",
                                   po::value(&o.startpot)->required(),
                                   "initial potential file")(
      "endpot,e", po::value(&endpot),
      "output potential file (required unless --evaluate)")(
      "evaluate", po::value(&evaluate),
      "evaluate the start potential against the configs and write a "
      "per-config\n"
      "forces/energy/stress JSON report to the given file, then exit (no "
      "optimization)")("format,f",
                       po::value(&o.format)->default_value(o.format),
                       "output format: native | lammps | imd")(
      "checkpoint,k", po::value(&checkpoint),
      "checkpoint prefix: save after each run, resume if present")(
      "maxiter", po::value(&o.max_iter)->default_value(o.max_iter),
      "max optimizer iterations")(
      "eweight", po::value(&o.energy_weight)->default_value(o.energy_weight),
      "energy residual weight")(
      "stress-weight",
      po::value(&o.stress_weight)->default_value(o.stress_weight),
      "stress tensor residual weight (0 = disabled)")(
      "smooth-weight",
      po::value(&o.smooth_weight)->default_value(o.smooth_weight),
      "curvature (Tikhonov) regularization weight on free knots (0 = "
      "disabled)")(
      "algorithm,a", po::value(&o.algorithm)->default_value(o.algorithm),
      algorithm_help.c_str())("seed", po::value(&o.seed)->default_value(o.seed),
                              "RNG seed for DE (0 = random_device)")(
      "de-F", po::value(&o.de_F)->default_value(o.de_F, "0.65"),
      "DE mutation factor F ∈ (0,1)")(
      "de-CR", po::value(&o.de_CR)->default_value(o.de_CR),
      "DE crossover probability CR ∈ (0,1)")(
      "de-np", po::value(&o.de_np)->default_value(o.de_np),
      "DE population factor: NP = de-np × D")(
      "de-gen", po::value(&o.de_gen)->default_value(o.de_gen),
      "DE maximum number of generations");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || argc == 1) {
      std::cout << desc << "\n"
                << "subcommand:\n"
                << "  forcesmith init --model <type> --out <file> [...]   "
                   "scaffold a fresh startpot\n"
                << "  (run 'forcesmith init --help' for its options)\n";
      return {ParseOutcome::ExitOk, {}};
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "error: " << e.what() << "\n\n" << desc << "\n";
    return {ParseOutcome::ExitError, {}};
  }

  if (vm.count("endpot")) {
    o.endpot = endpot;
  }
  if (vm.count("evaluate")) {
    o.evaluate = evaluate;
  }
  if (vm.count("checkpoint")) {
    o.checkpoint = checkpoint;
  }

  return {ParseOutcome::Run, std::move(o)};
}

} // namespace forcesmith::cli
