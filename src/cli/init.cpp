// `potfit init` — a JSON-native makeapot. Scaffolds a fresh startpot for any of
// the nine model types so the user can immediately fit it (potfit -s <out>).
//
// Two output mechanisms, split by what the native writers round-trip:
//   • analytic classical (pair/eam/adp/angular): emitted as analytic JSON
//     directly — the pair/eam native writers TABULATE analytic potentials, so
//     io::write_model would lose the type/params/bounds a startpot needs.
//   • tersoff/stiweb/ml: built in memory and written via io::write_model, whose
//     writers serialise those models' parameters verbatim.
//
// Analytic parameter names + defaults come from the shared macro table
// (potfit/potentials/analytic_param_defs.hpp), the same single source the
// reader registry uses — so a scaffolded file always reloads.

#include "potfit/cli/init.hpp"

#include "potfit/force/force_calculator.hpp"
#include "potfit/io/write_model.hpp"
#include "potfit/potentials/acsf.hpp"
#include "potfit/potentials/analytic_param_defs.hpp"
#include "potfit/potentials/lmbtr.hpp"
#include "potfit/potentials/soap.hpp"

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace po = boost::program_options;
using json = nlohmann::json;

namespace potfit::cli::init {
namespace {

struct Args {
  std::string model;
  std::string out;
  int ntypes = 1;
  double cutoff = 6.0;
  std::string
      functions; // makeapot-style "N*name,…"; empty → per-region default
  // SOAP
  int n_max = 6;
  int l_max = 6;
  double sigma = 0.5;
  // ACSF
  int g1 = 0;
  std::vector<double> g2_eta;
  std::vector<double> g2_rs;
  // LMBTR
  int k2_n = 50;
  int k3_n = 50;
  bool drop_k2 = false;
  bool drop_k3 = false;
  // ML head
  bool bias_free = false;
  unsigned seed = 42; // RNG seed for coefficient init (reproducible startpot)
};

[[noreturn]] void die(const std::string &msg) {
  std::cerr << "error: " << msg << "\n";
  std::exit(1);
}

// Expand a makeapot function list ("3*lj,morse" → [lj,lj,lj,morse]).
std::vector<std::string> expand_functions(const std::string &spec) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < spec.size()) {
    std::size_t comma = spec.find(',', i);
    std::string tok = spec.substr(i, comma - i);
    i = (comma == std::string::npos) ? spec.size() : comma + 1;
    if (tok.empty()) {
      continue;
    }
    std::size_t star = tok.find('*');
    int count = 1;
    std::string name = tok;
    if (star != std::string::npos) {
      count = std::stoi(tok.substr(0, star));
      name = tok.substr(star + 1);
      if (count < 1) {
        die("function multiplier must be ≥ 1 in '" + tok + "'");
      }
    }
    for (int k = 0; k < count; ++k) {
      out.push_back(name);
    }
  }
  return out;
}

// One analytic potential as the JSON the reader expects: type + span + each
// named parameter as {value,min,max}. Names/defaults come from the shared
// table.
json one_analytic(const std::string &fn, double rmin, double rmax) {
  std::span<const AnalyticParamDef> defs = analytic_defaults(fn);
  if (defs.empty()) {
    die("unknown analytic function '" + fn +
        "' (no default-parameter table; see analytic_param_defs.hpp)");
  }
  json o;
  o["type"] = fn;
  o["rmin"] = rmin;
  o["rmax"] = rmax;
  for (const AnalyticParamDef &d : defs) {
    o[std::string(d.name)] = {
        {"value", d.value}, {"min", d.min}, {"max", d.max}};
  }
  return o;
}

// A region of an analytic model: how many functions, the default if -f is
// omitted, and the domain its potentials live on ([0,rcut], or [-1,1] for the
// angular g(cosθ) term).
struct Region {
  std::string key; // JSON section key; empty → inline at top level (pair model)
  std::size_t count;
  std::string default_fn;
  double rmin;
  double rmax;
};

std::vector<Region> analytic_layout(const Args &a, std::size_t paircol) {
  const auto nt = static_cast<std::size_t>(a.ntypes);
  const double rc = a.cutoff;
  if (a.model == "pair") {
    return {{"", paircol, "lj", 0.0, rc}};
  }
  if (a.model == "eam") {
    return {{"pair", paircol, "lj", 0.0, rc},
            {"density", nt, "exp_decay", 0.0, rc},
            {"embedding", nt, "sqrt", 0.0, rc}};
  }
  if (a.model == "adp") {
    return {{"pair", paircol, "lj", 0.0, rc},
            {"density", nt, "exp_decay", 0.0, rc},
            {"embedding", nt, "sqrt", 0.0, rc},
            {"dipole", paircol, "exp_decay", 0.0, rc},
            {"quadrupole", paircol, "exp_decay", 0.0, rc}};
  }
  // angular
  return {{"pair", paircol, "lj", 0.0, rc},
          {"radial", paircol, "exp_decay", 0.0, rc},
          {"angular", nt, "parabola", -1.0, 1.0}};
}

int write_json(const Args &a, const json &j) {
  std::ofstream f(a.out);
  if (!f) {
    die("cannot open output file '" + a.out + "'");
  }
  f << j.dump(2) << "\n";
  std::cout << "wrote " << a.model << " startpot to " << a.out << "\n";
  return 0;
}

int scaffold_analytic(const Args &a) {
  const auto paircol = static_cast<std::size_t>(a.ntypes) * (a.ntypes + 1) / 2;
  const std::vector<Region> regions = analytic_layout(a, paircol);

  std::size_t total = 0;
  for (const Region &r : regions) {
    total += r.count;
  }

  std::vector<std::string> flat;
  if (!a.functions.empty()) {
    flat = expand_functions(a.functions);
    if (flat.size() != total) {
      die("--functions has " + std::to_string(flat.size()) + " functions but " +
          a.model + " (ntypes=" + std::to_string(a.ntypes) + ") needs " +
          std::to_string(total));
    }
  }

  json j;
  j["model"] = a.model;
  j["ntypes"] = a.ntypes;
  std::size_t idx = 0;
  for (const Region &r : regions) {
    json pots = json::array();
    for (std::size_t s = 0; s < r.count; ++s) {
      const std::string &fn = flat.empty() ? r.default_fn : flat[idx++];
      pots.push_back(one_analytic(fn, r.rmin, r.rmax));
    }
    json section = {{"format", "analytic"}, {"potentials", std::move(pots)}};
    if (r.key.empty()) {
      j["format"] = "analytic";
      j["potentials"] = std::move(section["potentials"]);
    } else {
      j[r.key] = std::move(section);
    }
  }
  return write_json(a, j);
}

// ── ML: build the model in memory + small-Gaussian linear heads, write_model ─

// Stddev of the zero-mean Gaussian used to seed linear-head coefficients. Small
// on purpose: the SOAP descriptor is L2-normalised (Σ D² = 1), so with
// E = Σ coeffs·D this keeps the initial predicted energies sub-eV while still
// breaking the all-zero symmetry of the start point.
constexpr double kHeadInitStd = 0.01;

// Attach one linear head per element type, sized to the descriptor, with
// coefficients drawn from a small zero-mean Gaussian (kHeadInitStd). The bias
// is fixed by default (forces-first fitting leaves its Jacobian column zero);
// --bias-free frees it for energy-weighted fits. A fixed --seed makes the
// scaffolded startpot reproducible; one shared rng gives each element type a
// distinct draw.
template <typename Model> void attach_init_heads(Model &m, const Args &a) {
  const auto D = static_cast<std::size_t>(m.descriptor_size());
  std::mt19937_64 rng(static_cast<std::uint64_t>(a.seed));
  std::normal_distribution<double> nd(0.0, kHeadInitStd);
  m.heads.reserve(static_cast<std::size_t>(a.ntypes));
  for (int t = 0; t < a.ntypes; ++t) {
    LinearHead h;
    h.coeffs.reserve(D);
    for (std::size_t k = 0; k < D; ++k) {
      h.coeffs.emplace_back(Param{nd(rng), false});
    }
    h.bias = Param{0.0, /*fixed=*/!a.bias_free};
    m.heads.emplace_back(EnergyHead{std::move(h)});
  }
}

int scaffold_ml(const Args &a) {
  std::optional<ForceCalculator> model;
  if (a.model == "soap") {
    SoapModel m;
    m.ntypes = static_cast<std::size_t>(a.ntypes);
    m.n_max = a.n_max;
    m.l_max = a.l_max;
    m.rcut = a.cutoff;
    m.sigma = a.sigma;
    m.init_radial_basis();
    attach_init_heads(m, a);
    model = ForceCalculator{std::move(m)};
  } else if (a.model == "acsf") {
    ACSF m;
    m.ntypes = static_cast<std::size_t>(a.ntypes);
    m.rcut = a.cutoff;
    m.g1 = static_cast<std::size_t>(a.g1);
    for (std::size_t k = 0; k < a.g2_eta.size(); ++k) {
      const double rs = k < a.g2_rs.size() ? a.g2_rs[k] : 0.0;
      m.radial.push_back({a.g2_eta[k], rs});
    }
    if (m.descriptor_size() == 0) {
      die("acsf needs at least one channel: set --g1 and/or --g2-eta");
    }
    attach_init_heads(m, a);
    model = ForceCalculator{std::move(m)};
  } else { // lmbtr
    LMBTR m;
    m.ntypes = static_cast<std::size_t>(a.ntypes);
    m.rcut = a.cutoff;
    m.k2 = a.drop_k2 ? std::nullopt
                     : std::optional<LMBTR::Grid>{{0.0, a.cutoff, a.k2_n, 0.3}};
    m.k3 = a.drop_k3 ? std::nullopt
                     : std::optional<LMBTR::Grid>{{-1.0, 1.0, a.k3_n, 0.1}};
    if (m.descriptor_size() == 0) {
      die("lmbtr needs k2 and/or k3 with n > 0");
    }
    attach_init_heads(m, a);
    model = ForceCalculator{std::move(m)};
  }

  auto r = io::write_model(*model, a.out, "native");
  if (!r) {
    die("failed to write '" + a.out + "'");
  }
  std::cout << "wrote " << a.model << " startpot to " << a.out << "\n";
  return 0;
}

// ── bond-order: default-constructed parameter blocks + write_model ───────────

int scaffold_bond_order(const Args &a) {
  const auto paircol = static_cast<std::size_t>(a.ntypes) * (a.ntypes + 1) / 2;
  ForceCalculator model = [&] {
    if (a.model == "tersoff") {
      TersoffForceCalculator c;
      c.ntypes = static_cast<std::size_t>(a.ntypes);
      c.params.reserve(static_cast<std::size_t>(a.ntypes));
      for (std::size_t s = 0; s < paircol; ++s) {
        TersoffParams tp;
        tp.R = Param{0.8 * a.cutoff};
        tp.S = Param{a.cutoff};
        c.params.emplace_back(tp);
      }
      return ForceCalculator{std::move(c)};
    }
    // stiweb
    StiwebForceCalculator c;
    c.ntypes = static_cast<std::size_t>(a.ntypes);
    c.params.reserve(static_cast<std::size_t>(a.ntypes));
    for (std::size_t s = 0; s < paircol; ++s) {
      SWParams sp;
      sp.a1 = Param{a.cutoff};
      sp.a2 = Param{a.cutoff};
      c.params.emplace_back(sp);
    }
    c.lambda.assign(static_cast<std::size_t>(a.ntypes) * paircol, Param{2.0});
    return ForceCalculator{std::move(c)};
  }();

  auto r = io::write_model(model, a.out, "native");
  if (!r) {
    die("failed to write '" + a.out + "'");
  }
  std::cout << "wrote " << a.model << " startpot to " << a.out << "\n";
  return 0;
}

} // namespace

int run(int argc, char *argv[]) {
  Args a;
  po::options_description desc(
      "potfit init — scaffold a fresh startpot (a JSON-native makeapot)");
  desc.add_options()("help,h", "show this message")(
      "model,m", po::value(&a.model),
      "pair | eam | adp | angular | tersoff | stiweb | acsf | soap | lmbtr")(
      "out,o", po::value(&a.out), "output startpot file")(
      "ntypes,n", po::value(&a.ntypes)->default_value(a.ntypes),
      "number of atom types")("cutoff,c",
                              po::value(&a.cutoff)->default_value(a.cutoff),
                              "cutoff radius (Å)")(
      "functions,f", po::value(&a.functions),
      "analytic models: makeapot-style list, e.g. \"3*lj\" or "
      "\"lj,exp_decay,sqrt\" (omit for sensible per-region defaults)")(
      "n-max", po::value(&a.n_max)->default_value(a.n_max),
      "soap radial basis")("l-max", po::value(&a.l_max)->default_value(a.l_max),
                           "soap angular degree")(
      "sigma", po::value(&a.sigma)->default_value(a.sigma),
      "soap atomic Gaussian width")("g1", po::value(&a.g1)->default_value(a.g1),
                                    "acsf G1 channel count")(
      "g2-eta", po::value(&a.g2_eta)->multitoken(),
      "acsf G2 eta widths")("g2-rs", po::value(&a.g2_rs)->multitoken(),
                            "acsf G2 rs centres (default all 0)")(
      "k2-n", po::value(&a.k2_n)->default_value(a.k2_n),
      "lmbtr k2 grid points")("k3-n", po::value(&a.k3_n)->default_value(a.k3_n),
                              "lmbtr k3 grid points")(
      "drop-k2", po::bool_switch(&a.drop_k2), "lmbtr: disable the k2 term")(
      "drop-k3", po::bool_switch(&a.drop_k3), "lmbtr: disable the k3 term")(
      "bias-free", po::bool_switch(&a.bias_free),
      "ml: leave the head bias free (default fixed, for forces-first fits)")(
      "seed", po::value(&a.seed)->default_value(a.seed),
      "ml: RNG seed for the small-Gaussian head-coefficient init");

  po::variables_map vm;
  try {
    // argv[0] is "init"; skip it so program_options sees only init's flags.
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || argc == 1) {
      std::cout << desc << "\n";
      return vm.count("help") ? 0 : 1;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "error: " << e.what() << "\n\n" << desc << "\n";
    return 1;
  }

  if (a.model.empty()) {
    die("--model is required");
  }
  if (a.out.empty()) {
    die("--out is required");
  }
  if (a.ntypes < 1) {
    die("--ntypes must be ≥ 1");
  }
  if (a.cutoff <= 0.0) {
    die("--cutoff must be > 0");
  }

  if (a.model == "pair" || a.model == "eam" || a.model == "adp" ||
      a.model == "angular") {
    return scaffold_analytic(a);
  }
  if (a.model == "tersoff" || a.model == "stiweb") {
    return scaffold_bond_order(a);
  }
  if (a.model == "soap" || a.model == "acsf" || a.model == "lmbtr") {
    return scaffold_ml(a);
  }
  die("unknown --model '" + a.model + "'");
}

} // namespace potfit::cli::init
