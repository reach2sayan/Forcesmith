// ML-potential fitting driver for the converted UNEP DFT dataset — the ML
// parallel of tests/unep/fit_eam_api.cpp. Built as the single `ml_fit` binary.
//
// It builds, in memory through the public PotFit API, a descriptor + an energy
// head, seeds it as the force model, and runs a forces-first fit against
// data/unep/<el>_dft_unep.json (real DFT data from UNEP-v1, Zenodo 11533864).
// Descriptor and element are runtime flags (the head is a linear map):
//   --descriptor soap     (--n-max/--l-max/--sigma)   | symfunc (--g2-eta/--g2-rs)
// e.g.
//   ml_fit --element Cu --descriptor soap
//   ml_fit --element Cu --descriptor symfunc --g2-eta 0.5 1.2 0.3
//
// PERFORMANCE: this round uses finite-difference descriptor gradients, so cost
// grows quickly with config count and neighbours. Keep --max-configs small; this
// is a correctness/sanity driver, not a production trainer.

#include "potfit/api/potfit.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/io/config_reader.hpp" // io::ParseError
#include "potfit/io/logging.hpp"
#include "potfit/optimization/ipopt_solver.hpp"
#include "potfit/optimization/solver.hpp"
#include "potfit/potentials/soap.hpp"
#include "potfit/potentials/acsf.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#ifndef POTFIT_ML_ELEMENT
#define POTFIT_ML_ELEMENT "Cu"
#endif

namespace leaf = boost::leaf;
namespace po = boost::program_options;
using json = nlohmann::json;
using namespace potfit;

// BOOST_LEAF_CHECK expands to a GNU statement-expression ({ ... }); silence the
// pedantic complaint about that Boost idiom for this translation unit.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

namespace {

struct Args {
  std::string element = POTFIT_ML_ELEMENT; // compile-time default; --element overrides
  std::string descriptor = "soap"; // descriptor: "soap" | "symfunc"
  std::string head = "linear";     // energy head: "linear" (only option)
  std::vector<double> g2_eta = {0.05, 0.2, 0.5, 1.0, 2.0, 4.0}; // symfunc G2 widths
  std::vector<double> g2_rs;        // symfunc G2 centers; empty → all 0
  std::string data_dir = "data/unep";
  std::string out_dir = "tests/ml";
  int maxiter = 100;
  double eweight = 0.0;     // forces-first
  double stress_weight = 0.0;
  std::string algorithm = "lm"; // E = c·D + bias is linear in the coeffs, so
                                // Gauss–Newton (lm) converges in ~one step
  int max_configs = 20;     // small by default (FD×FD is heavy)
  int stride = 1;
  int n_max = 4;
  int l_max = 3;
  double sigma = 0.5;
  double rcut = 0.0;        // 0 → derive from dmin
};

leaf::result<std::string> read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    return leaf::new_error(io::ParseError{"cannot open file: " + path, 0});
  }
  return std::string(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
}

leaf::result<json> read_json(const std::string &path) {
  BOOST_LEAF_AUTO(text, read_file(path));
  try {
    return json::parse(text);
  } catch (const json::parse_error &e) {
    return leaf::new_error(io::ParseError{e.what(), 0});
  }
}

std::string lower(std::string s) {
  std::ranges::transform(s, s.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  return s;
}

json subsample(const json &configs, int max_configs, int stride) {
  json out = json::array();
  for (std::size_t i = 0; i < configs.size();
       i += (stride > 1 ? static_cast<std::size_t>(stride) : 1)) {
    out.push_back(configs[i]);
  }
  if (max_configs > 0 && out.size() > static_cast<std::size_t>(max_configs)) {
    out.erase(out.begin() + max_configs, out.end());
  }
  return out;
}

double nearest_neighbour_distance(const json &configs) {
  double dmin = std::numeric_limits<double>::infinity();
  for (const auto &cfg : configs) {
    const auto &atoms = cfg.at("atoms");
    const std::size_t n = atoms.size();
    if (n < 2) {
      continue;
    }
    std::vector<Vec3> pos(n);
    for (std::size_t a = 0; a < n; ++a) {
      const auto &p = atoms[a].at("position");
      pos[a] = Vec3(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
    }
    Mat3 cell;
    for (int k = 0; k < 3; ++k) {
      const auto &v = cfg.at(std::string(1, "XYZ"[k]));
      cell.row(k) << v[0].get<double>(), v[1].get<double>(), v[2].get<double>();
    }
    std::optional<Mat3> inv;
    if (std::abs(cell.determinant()) > 1e-12) {
      inv = cell.inverse();
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        Vec3 d = pos[i] - pos[j];
        if (inv) {
          Vec3 frac = inv->transpose() * d;
          frac -= frac.array().round().matrix();
          d = cell.transpose() * frac;
        }
        const double dist = d.norm();
        if (dist > 1e-6) {
          dmin = std::min(dmin, dist);
        }
      }
    }
  }
  return dmin;
}

leaf::result<void> add_configs(PotFit &s, const json &configs) {
  for (const auto &rec : configs) {
    BOOST_LEAF_AUTO(cfg, Configuration::from_text(rec.dump()));
    s.add_configuration(std::move(cfg));
  }
  return {};
}

void apply_options(PotFit &s, const Args &a) {
  OptimizerOptions &o = s.options();
  o.energy_weight = a.eweight;
  o.stress_weight = a.stress_weight;
  o.smooth_weight = 0.0; // no spline smoothness for ML models
  if (a.algorithm == "powell") {
    s.set_solver(Solver{EigenHybridSolver{a.maxiter}});
  } else if (a.algorithm == "de") {
    s.set_solver(Solver{BoostDESolver{}});
  } else if (a.algorithm == "ls") {
    s.set_solver(Solver{LineSearchSolver{a.maxiter}});
  } else if (a.algorithm == "ipopt") {
    s.set_solver(Solver{IpoptSolver{a.maxiter}});
  } else {
    // Default: Gauss–Newton/LM. The linear head makes the force residual linear
    // in the coeffs, so lm converges in ~one well-conditioned least-squares step.
    s.set_solver(Solver{EigenLMSolver{a.maxiter}});
  }
}

leaf::result<double> force_rmse(PotFit &s) {
  BOOST_LEAF_AUTO(configs, s.configurations());
  if (configs.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // The uncached predict path is expensive for SOAP (FD-over-energy), so evaluate
  // the configs in parallel. Freeze once here first: thereafter ensure_frozen is a
  // read-only no-op and concurrent s.evaluate is safe (model + configs read-only;
  // force::evaluate uses a private scratch).
  BOOST_LEAF_CHECK(s.evaluate(0));

  std::vector<std::size_t> idx(configs.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<double> partial_sq(configs.size(), 0.0);
  std::vector<std::size_t> partial_n(configs.size(), 0);
  std::atomic<bool> ok{true};

  std::for_each(std::execution::par, idx.begin(), idx.end(), [&](std::size_t i) {
    auto r = s.evaluate(i);
    if (!r) {
      ok.store(false);
      return;
    }
    double ss = 0.0;
    std::size_t nn = 0;
    for (std::size_t a = 0; a < configs[i].atoms.size(); ++a) {
      const Vec3 d = r->forces[a] - configs[i].atoms[a].ref.force;
      ss += d.squaredNorm();
      nn += 3;
    }
    partial_sq[i] = ss;
    partial_n[i] = nn;
  });
  if (!ok.load()) {
    return leaf::new_error(
        io::ParseError{"evaluate failed during force_rmse", 0});
  }

  const double sumsq =
      std::accumulate(partial_sq.begin(), partial_sq.end(), 0.0);
  const std::size_t n =
      std::accumulate(partial_n.begin(), partial_n.end(), std::size_t{0});
  return n ? std::sqrt(sumsq / static_cast<double>(n))
           : std::numeric_limits<double>::quiet_NaN();
}

// Attach a linear energy head sized to the model's descriptor.
// Descriptor-agnostic — used by both model types.
template <typename Model>
void attach_head(Model &m, const Args &a) {
  const int D = static_cast<int>(m.descriptor_size());
  m.heads.reserve(1);
  // E = c·D + bias makes the force residual linear in the coeffs, so lm
  // converges in ~one Gauss–Newton step. Start at zero; free the bias only
  // when energies are weighted (with forces-only fitting its Jacobian column
  // is identically zero and the normal equations go singular).
  LinearHead h;
  h.coeffs.assign(static_cast<std::size_t>(D), Param{0.0, false});
  h.bias = Param{0.0, /*fixed=*/a.eweight == 0.0};
  m.heads.emplace_back(EnergyHead{std::move(h)});
}

// Build the force model (single element → ntypes 1) for the requested
// descriptor ("soap" | "symfunc") and energy head.
ForceCalculator build_model(const Args &a, double rcut) {
  if (a.descriptor == "symfunc") {
    ACSF m;
    m.ntypes = 1;
    m.rcut = rcut;
    m.radial.reserve(a.g2_eta.size());
    for (std::size_t k = 0; k < a.g2_eta.size(); ++k) {
      const double rs = k < a.g2_rs.size() ? a.g2_rs[k] : 0.0;
      m.radial.push_back({a.g2_eta[k], rs}); // G2{eta, rs}
    }
    attach_head(m, a);
    return ForceCalculator{std::move(m)};
  }

  SoapModel m;
  m.ntypes = 1;
  m.n_max = a.n_max;
  m.l_max = a.l_max;
  m.rcut = rcut;
  m.sigma = a.sigma;
  m.init_radial_basis();
  attach_head(m, a);
  return ForceCalculator{std::move(m)};
}

struct Result {
  int nconf = 0;
  std::size_t natoms = 0, descriptor_size = 0, params = 0;
  double dmin = 0, rcut = 0;
  double rmse_start = NAN, rmse_final = NAN;
};

leaf::result<Result> fit_element(const Args &a) {
  const std::string ell = lower(a.element);
  const std::string data_path = a.data_dir + "/" + ell + "_dft_unep.json";

  BOOST_LEAF_AUTO(all, read_json(data_path));
  if (!all.is_array()) {
    return leaf::new_error(
        io::ParseError{data_path + ": expected a top-level array", 0});
  }
  const json configs = subsample(all, a.max_configs, a.stride);

  Result res;
  res.nconf = static_cast<int>(configs.size());
  for (const auto &c : configs) {
    res.natoms += c.at("atoms").size();
  }
  res.dmin = nearest_neighbour_distance(configs);
  if (!std::isfinite(res.dmin)) {
    return leaf::new_error(io::ParseError{"no neighbour pairs found", 0});
  }
  res.rcut = a.rcut > 0.0 ? a.rcut : std::min(6.0, 2.4 * res.dmin);

  PotFit s;
  BOOST_LEAF_CHECK(add_configs(s, configs));
  ForceCalculator model = build_model(a, res.rcut);
  res.descriptor_size = std::visit(
      [](const auto &m) -> std::size_t {
        if constexpr (requires { m.descriptor_size(); })
          return m.descriptor_size();
        else
          return 0;
      },
      model);
  res.params = std::visit([](const auto &m) { return m.param_count(); }, model);
  BOOST_LEAF_CHECK(s.seed_force_model(std::move(model)));

  using clock = std::chrono::steady_clock;
  auto ms = [](auto d) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
  };

  std::cerr << "... [phase] start_rmse begin\n" << std::flush;
  const auto t0 = clock::now();
  BOOST_LEAF_ASSIGN(res.rmse_start, force_rmse(s)); // untrained baseline
  const auto t1 = clock::now();
  std::cerr << "... [phase] start_rmse done (" << ms(t1 - t0)
            << " ms); optimize begin\n"
            << std::flush;
  apply_options(s, a);
  BOOST_LEAF_CHECK(s.optimize());
  const auto t2 = clock::now();
  std::cerr << "... [phase] optimize done (" << ms(t2 - t1)
            << " ms); final_rmse begin\n"
            << std::flush;
  BOOST_LEAF_ASSIGN(res.rmse_final, force_rmse(s));
  const auto t3 = clock::now();
  std::cerr << "... timing (ms): start_rmse=" << ms(t1 - t0)
            << " optimize=" << ms(t2 - t1) << " final_rmse=" << ms(t3 - t2)
            << "\n"
            << std::flush;

  std::filesystem::create_directories(a.out_dir + "/fits");
  BOOST_LEAF_CHECK(s.write(a.out_dir + "/fits/" + ell + "_" + a.descriptor +
                               "_" + a.head + ".json",
                           "native"));
  return res;
}

void report(const std::string &el, const Result &r) {
  auto f = [](double x) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(5) << x;
    return os.str();
  };
  std::cout << "\nelement     " << el << "\n"
            << "nconf       " << r.nconf << "\n"
            << "natoms      " << r.natoms << "\n"
            << "dmin        " << f(r.dmin) << "  (rcut " << f(r.rcut) << ")\n"
            << "descriptor  " << r.descriptor_size << "\n"
            << "params      " << r.params << "\n"
            << "force RMSE (eV/Å):\n"
            << "  start     " << f(r.rmse_start) << "\n"
            << "  final     " << f(r.rmse_final) << "\n";
}

} // namespace

int main(int argc, char *argv[]) {
  Args a;
  po::options_description desc("ML fit (API)");
  desc.add_options()("help,h", "show this message")(
      "element,e", po::value(&a.element)->default_value(a.element), "Cu | Al | …")(
      "descriptor", po::value(&a.descriptor)->default_value(a.descriptor),
      "soap | symfunc")(
      "head", po::value(&a.head)->default_value(a.head), "linear")(
      "data-dir", po::value(&a.data_dir)->default_value(a.data_dir))(
      "out-dir", po::value(&a.out_dir)->default_value(a.out_dir))(
      "maxiter", po::value(&a.maxiter)->default_value(a.maxiter))(
      "eweight", po::value(&a.eweight)->default_value(a.eweight))(
      "stress-weight",
      po::value(&a.stress_weight)->default_value(a.stress_weight))(
      "algorithm,a", po::value(&a.algorithm)->default_value(a.algorithm),
      "lm (default) | ipopt | powell | de | ls")(
      "max-configs", po::value(&a.max_configs)->default_value(a.max_configs))(
      "stride", po::value(&a.stride)->default_value(a.stride))(
      "n-max", po::value(&a.n_max)->default_value(a.n_max), "soap only")(
      "l-max", po::value(&a.l_max)->default_value(a.l_max), "soap only")(
      "sigma", po::value(&a.sigma)->default_value(a.sigma), "soap only")(
      "g2-eta", po::value(&a.g2_eta)->multitoken(), "symfunc G2 eta widths")(
      "g2-rs", po::value(&a.g2_rs)->multitoken(),
      "symfunc G2 rs centers (default all 0)")(
      "rcut", po::value(&a.rcut)->default_value(a.rcut), "0 = derive from dmin");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "error: " << e.what() << "\n\n" << desc << "\n";
    return 1;
  }

  if (a.head != "linear") {
    std::cerr << "error: --head must be 'linear' (got '" << a.head << "')\n\n"
              << desc << "\n";
    return 1;
  }
  if (a.descriptor != "soap" && a.descriptor != "symfunc") {
    std::cerr << "error: --descriptor must be 'soap' or 'symfunc' (got '"
              << a.descriptor << "')\n\n"
              << desc << "\n";
    return 1;
  }
  if (a.descriptor == "symfunc") {
    if (a.g2_eta.empty()) {
      std::cerr << "error: --descriptor symfunc needs at least one --g2-eta\n\n"
                << desc << "\n";
      return 1;
    }
    if (!a.g2_rs.empty() && a.g2_rs.size() != a.g2_eta.size()) {
      std::cerr << "error: --g2-rs (" << a.g2_rs.size() << ") must match --g2-eta ("
                << a.g2_eta.size() << ") in length\n\n"
                << desc << "\n";
      return 1;
    }
  }

  std::cerr << "... fitting " << a.element << " (" << a.descriptor << "+"
            << a.head << ")\n";

  // Per-iteration logging to console + a per-(element,descriptor,head) rotating
  // file, so parallel runs don't clobber each other's log. The guards must
  // outlive fit_element()'s optimize() call.
  potfit::log::init("ml_fit_" + a.element + "_" + a.descriptor + "_" + a.head +
                    ".log");
  auto log_sinks = potfit::log::connect_signals();

  int ret = 0;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        BOOST_LEAF_AUTO(res, fit_element(a));
        report(a.element, res);
        return {};
      },
      [&](const io::ParseError &e) {
        std::cerr << "error: " << e.message << "\n";
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

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
