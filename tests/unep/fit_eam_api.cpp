// EAM fitting driver for the converted UNEP DFT dataset — the in-process,
// programmatic-API parallel of tests/unep/fit_eam.py. Built as the single
// `unep_fit` binary.
//
// fit_eam.py shells out to the `forcesmith` CLI binary once per stage; this driver
// instead builds the fit in memory through the public Forcesmith API and runs
// the *same* two-stage, forces-first pipeline against
// data/unep/<el>_dft_unep.json (real DFT forces/energies from UNEP-v1,
// Zenodo 11533864). It is fully self-contained: cutoff derivation, the analytic
// start potential, both fits, the stage-1→stage-2 knot down-sample and the
// force-RMSE checkpoints all run here, with no Python or subprocess dependency.
//
// The element is a runtime flag (--element Cu), so there is one binary rather
// than one per metal. The flags mirror fit_eam.py's (minus --jobs, which only
// makes sense for the multi-element Python launcher):
//   unep_fit --element Cu
//
// Pipeline (matches fit_eam.py):
//   1. dmin via a minimum-image pairwise scan ⇒ re=dmin, rmin=max(1, 0.88·dmin),
//      rmax=min(6.5, 2.4·dmin).
//   2. Stage-1 analytic start (morse pair + exp_decay density + sqrt embedding).
//   3. Stage-1 fit ⇒ dense native EAM (writer's default knots).
//   4. Stage-2 start: down-sample each section of the stage-1 result to --knots
//      free knots ("analytic seeds tabulated").
//   5. Stage-2 fit with --smooth-weight regularizing the free splines.
//   6. Per-atom force RMSE at start / stage-1 / final.

#include "forcesmith/api/forcesmith.hpp"
#include "forcesmith/io/config_reader.hpp" // io::ParseError
#include "forcesmith/optimization/ipopt_solver.hpp"
#include "forcesmith/optimization/solver.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef FORCESMITH_UNEP_ELEMENT
#define FORCESMITH_UNEP_ELEMENT "Cu"
#endif

namespace leaf = boost::leaf;
namespace po = boost::program_options;
using json = nlohmann::json;
using namespace forcesmith;

// BOOST_LEAF_CHECK expands to a GNU statement-expression ({ ... }); silence the
// pedantic complaint about that Boost idiom for this translation unit.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wgnu-statement-expression-from-macro-expansion"
#endif

namespace {

struct Args {
  std::string element = FORCESMITH_UNEP_ELEMENT;
  std::string data_dir = "data/unep";
  std::string out_dir = "tests/unep";
  int maxiter = 300;
  double eweight = 0.1;
  double stress_weight = 0.0;
  double smooth_weight = 1.0;
  int knots = 15;
  std::string algorithm = "lm";
  int max_configs = 0; // 0 = all
  int stride = 1;
  int stage = 2;
};

// ── small IO helpers ─────────────────────────────────────────────────────────

leaf::result<std::string> read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    return leaf::new_error(io::ParseError{"cannot open file: " + path, 0});
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

void write_json(const std::string &path, const json &j) {
  std::ofstream(path) << j.dump(1);
}

std::string lower(std::string s) {
  std::ranges::transform(s, s.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  return s;
}

// ── data loading & geometry ──────────────────────────────────────────────────

// Deterministic subset for fast smoke runs (mirrors fit_eam.py::subsample).
json subsample(const json &configs, int max_configs, int stride) {
  json out = json::array();
  for (std::size_t i = 0; i < configs.size();
       i += (stride > 1 ? static_cast<std::size_t>(stride) : 1))
    out.push_back(configs[i]);
  if (max_configs > 0 && out.size() > static_cast<std::size_t>(max_configs))
    out.erase(out.begin() + max_configs, out.end());
  return out;
}

// Smallest interatomic distance across all configs (minimum-image). Min-image is
// exact for the *closest* pair, which is all the cutoffs need.
double nearest_neighbour_distance(const json &configs) {
  double dmin = std::numeric_limits<double>::infinity();
  for (const auto &cfg : configs) {
    const auto &atoms = cfg.at("atoms");
    const std::size_t n = atoms.size();
    if (n < 2)
      continue;
    std::vector<Vec3> pos(n);
    for (std::size_t a = 0; a < n; ++a) {
      const auto &p = atoms[a].at("position");
      pos[a] = Vec3(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
    }
    Mat3 cell; // rows = lattice vectors a, b, c
    for (int k = 0; k < 3; ++k) {
      const auto &v = cfg.at(std::string(1, "XYZ"[k]));
      cell.row(k) << v[0].get<double>(), v[1].get<double>(), v[2].get<double>();
    }
    std::optional<Mat3> inv;
    if (std::abs(cell.determinant()) > 1e-12)
      inv = cell.inverse();
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        Vec3 d = pos[i] - pos[j];
        if (inv) {
          // frac (row) = d^T · inv  ⇒  wrap to [-0.5,0.5)  ⇒  back to cartesian.
          Vec3 frac = inv->transpose() * d;
          frac -= frac.array().round().matrix();
          d = cell.transpose() * frac;
        }
        const double dist = d.norm();
        if (dist > 1e-6)
          dmin = std::min(dmin, dist);
      }
  }
  return dmin;
}

// ── potential construction ───────────────────────────────────────────────────

// Stage-1 analytic EAM start (all params free): morse pair, exp_decay density,
// sqrt embedding (B>0). Matches fit_eam.py::analytic_start.
leaf::result<void> set_analytic_start(Forcesmith &s, const std::string &el,
                                      double rmin, double rmax) {
  BOOST_LEAF_AUTO(pair, RadialPotential::from_text(
                            json{{"type", "morse"}, {"rmin", rmin},
                                 {"rmax", rmax}, {"De", 0.5}, {"a", 2.0},
                                 {"re", rmin}}
                                .dump()));
  BOOST_LEAF_AUTO(rho, RadialPotential::from_text(
                           json{{"type", "exp_decay"}, {"rmin", rmin},
                                {"rmax", rmax}, {"A", 1.0}, {"B", 1.0}}
                               .dump()));
  BOOST_LEAF_AUTO(emb, RadialPotential::from_text(
                           json{{"type", "sqrt"}, {"rmin", 1e-4}, {"rmax", 10.0},
                                {"A", -1.0}, {"B", 1.0}}
                               .dump()));
  BOOST_LEAF_CHECK(s.set_pair_potential(el, el, std::move(pair)));
  BOOST_LEAF_CHECK(s.set_density(el, std::move(rho)));
  BOOST_LEAF_CHECK(s.set_embedding(el, std::move(emb)));
  return {};
}

// Linear resample of a knot array onto `knots` evenly-spaced points over the
// section's normalized [0,1] domain (np.interp on linspace grids).
std::vector<double> resample(const std::vector<double> &y, int knots) {
  const int n = static_cast<int>(y.size());
  if (n == knots)
    return y;
  std::vector<double> out(static_cast<std::size_t>(knots));
  for (int j = 0; j < knots; ++j) {
    const double t = knots == 1 ? 0.0 : static_cast<double>(j) / (knots - 1);
    const double x = t * (n - 1);
    const int i0 = std::min(static_cast<int>(std::floor(x)), n - 2);
    const double f = x - i0;
    out[static_cast<std::size_t>(j)] = y[i0] * (1.0 - f) + y[i0 + 1] * f;
  }
  return out;
}

// Stage-2 start: down-sample each section of the dense stage-1 fit to `knots`
// free tabulated knots, then place them on the session (mirrors
// fit_eam.py::tabulated_start_from_dense). Also returns the start spec written
// out for inspection.
leaf::result<json> set_tabulated_start(Forcesmith &s, const std::string &el,
                                       const json &dense, int knots) {
  json spec{{"model", dense.value("model", "eam")},
            {"ntypes", dense.value("ntypes", 1)}};
  auto one = [&](const char *section) -> leaf::result<RadialPotential> {
    const auto &p = dense.at(section).at("potentials").at(0);
    std::vector<double> y = p.at("knots").get<std::vector<double>>();
    const double rmin = p.at("rmin").get<double>();
    const double rmax = p.at("rmax").get<double>();
    const std::vector<double> ds = resample(y, knots);
    json pot;
    pot["rmin"] = rmin;
    pot["rmax"] = rmax;
    pot["knots"] = ds;
    json sec;
    sec["format"] = "tabulated";
    sec["potentials"] = json::array({pot});
    spec[section] = sec;
    return RadialPotential::from_text(pot.dump());
  };
  BOOST_LEAF_AUTO(pair, one("pair"));
  BOOST_LEAF_AUTO(rho, one("density"));
  BOOST_LEAF_AUTO(emb, one("embedding"));
  BOOST_LEAF_CHECK(s.set_pair_potential(el, el, std::move(pair)));
  BOOST_LEAF_CHECK(s.set_density(el, std::move(rho)));
  BOOST_LEAF_CHECK(s.set_embedding(el, std::move(emb)));
  return spec;
}

// ── session assembly, optimization, evaluation ───────────────────────────────

leaf::result<void> add_configs(Forcesmith &s, const json &configs) {
  for (const auto &rec : configs) {
    BOOST_LEAF_AUTO(cfg, Configuration::from_text(rec.dump()));
    s.add_configuration(std::move(cfg));
  }
  return {};
}

void apply_options(Forcesmith &s, const Args &a, double smooth_weight) {
  OptimizerOptions &o = s.options();
  o.energy_weight = a.eweight;
  o.stress_weight = a.stress_weight;
  o.smooth_weight = smooth_weight;

  if (a.algorithm == "ipopt")
    s.set_solver(Solver{IpoptSolver{a.maxiter}});
  else if (a.algorithm == "powell")
    s.set_solver(Solver{EigenHybridSolver{a.maxiter}});
  else if (a.algorithm == "de")
    s.set_solver(Solver{BoostDESolver{}});
  else if (a.algorithm == "ls")
    s.set_solver(Solver{LineSearchSolver{a.maxiter}});
  else
    s.set_solver(Solver{EigenLMSolver{a.maxiter}});
}

// Per-atom force-component RMSE (eV/Å) of the session's current model against
// the loaded reference forces.
leaf::result<double> force_rmse(Forcesmith &s) {
  BOOST_LEAF_AUTO(configs, s.configurations());
  double sumsq = 0.0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < configs.size(); ++i) {
    BOOST_LEAF_AUTO(r, s.evaluate(i));
    for (std::size_t a = 0; a < configs[i].atoms.size(); ++a) {
      const Vec3 d = r.forces[a] - configs[i].atoms[a].ref.force;
      sumsq += d.squaredNorm();
      n += 3;
    }
  }
  return n ? std::sqrt(sumsq / static_cast<double>(n))
           : std::numeric_limits<double>::quiet_NaN();
}

// ── driver ───────────────────────────────────────────────────────────────────

struct Result {
  int nconf = 0;
  std::size_t natoms = 0;
  double dmin = 0, rmin = 0, rmax = 0;
  double rmse_start = NAN, rmse_analytic = NAN, rmse_tabular = NAN;
};

leaf::result<Result> fit_element(const Args &args) {
  const std::string el = args.element;
  const std::string ell = lower(el);
  const std::string data_path = args.data_dir + "/" + ell + "_dft_unep.json";
  const std::string work = args.out_dir + "/work/" + ell;
  const std::string fits = args.out_dir + "/fits";
  std::filesystem::create_directories(work);
  std::filesystem::create_directories(fits);

  BOOST_LEAF_AUTO(all, read_json(data_path));
  if (!all.is_array())
    return leaf::new_error(
        io::ParseError{data_path + ": expected a top-level array", 0});
  const json configs = subsample(all, args.max_configs, args.stride);

  Result res;
  res.nconf = static_cast<int>(configs.size());
  for (const auto &c : configs)
    res.natoms += c.at("atoms").size();

  res.dmin = nearest_neighbour_distance(configs);
  if (!std::isfinite(res.dmin))
    return leaf::new_error(io::ParseError{"no neighbour pairs found", 0});
  res.rmin = std::max(1.0, 0.88 * res.dmin);
  res.rmax = std::min(6.5, 2.4 * res.dmin);

  // ── Stage 1: analytic start → analytic fit (written dense) ─────────────────
  Forcesmith s1;
  BOOST_LEAF_CHECK(add_configs(s1, configs));
  BOOST_LEAF_CHECK(set_analytic_start(s1, el, res.rmin, res.rmax));
  BOOST_LEAF_ASSIGN(res.rmse_start, force_rmse(s1)); // un-fitted baseline

  const std::string analytic_fit = fits + "/" + ell + "_eam_analytic.json";
  if (args.stage >= 1) {
    apply_options(s1, args, /*smooth_weight=*/0.0);
    BOOST_LEAF_CHECK(s1.optimize());
    BOOST_LEAF_ASSIGN(res.rmse_analytic, force_rmse(s1));
    BOOST_LEAF_CHECK(s1.write(analytic_fit, "native"));
  }

  // ── Stage 2: down-sampled tabulated start → regularized tabular fit ────────
  if (args.stage >= 2) {
    BOOST_LEAF_AUTO(dense, read_json(analytic_fit));
    Forcesmith s2;
    BOOST_LEAF_CHECK(add_configs(s2, configs));
    BOOST_LEAF_AUTO(start_spec, set_tabulated_start(s2, el, dense, args.knots));
    write_json(work + "/start_tab.json", start_spec);

    apply_options(s2, args, args.smooth_weight);
    BOOST_LEAF_CHECK(s2.optimize());
    BOOST_LEAF_ASSIGN(res.rmse_tabular, force_rmse(s2));
    BOOST_LEAF_CHECK(s2.write(fits + "/" + ell + "_eam_fit.json", "native"));
  }
  return res;
}

void report(const std::string &el, const Result &r) {
  auto f = [](double x) -> std::string {
    if (std::isnan(x))
      return "—";
    std::ostringstream os;
    os << std::fixed << std::setprecision(5) << x;
    return os.str();
  };
  std::cout << "\nelement   " << el << "\n"
            << "nconf     " << r.nconf << "\n"
            << "natoms    " << r.natoms << "\n"
            << "dmin      " << f(r.dmin) << "  (rmin " << f(r.rmin) << ", rmax "
            << f(r.rmax) << ")\n"
            << "force RMSE (eV/Å):\n"
            << "  start    " << f(r.rmse_start) << "\n"
            << "  analytic " << f(r.rmse_analytic) << "\n"
            << "  tabular  " << f(r.rmse_tabular) << "\n";
}

} // namespace

int main(int argc, char *argv[]) {
  Args a;
  po::options_description desc("UNEP EAM fit (API)");
  desc.add_options()("help,h", "show this message")(
      "element,e", po::value(&a.element)->default_value(a.element), "Cu | Al | …")(
      "data-dir", po::value(&a.data_dir)->default_value(a.data_dir))(
      "out-dir", po::value(&a.out_dir)->default_value(a.out_dir))(
      "maxiter", po::value(&a.maxiter)->default_value(a.maxiter))(
      "eweight", po::value(&a.eweight)->default_value(a.eweight))(
      "stress-weight",
      po::value(&a.stress_weight)->default_value(a.stress_weight))(
      "smooth-weight",
      po::value(&a.smooth_weight)->default_value(a.smooth_weight))(
      "knots", po::value(&a.knots)->default_value(a.knots))(
      "algorithm,a", po::value(&a.algorithm)->default_value(a.algorithm),
      "lm | ipopt | powell | de | ls")(
      "max-configs", po::value(&a.max_configs)->default_value(a.max_configs),
      "cap configs (0 = all)")(
      "stride", po::value(&a.stride)->default_value(a.stride))(
      "stage", po::value(&a.stage)->default_value(a.stage), "highest stage 1|2");

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

  std::cout << std::setprecision(10);
  std::cerr << "... fitting " << a.element << "\n";

  int ret = 0;
  leaf::try_handle_all(
      [&]() -> leaf::result<void> {
        BOOST_LEAF_AUTO(res, fit_element(a));
        report(a.element, res);
        return {};
      },
      [&](const io::ParseError &e) {
        std::cerr << "error" << (e.line ? " (line " + std::to_string(e.line) +
                                              ")"
                                        : "")
                  << ": " << e.message << "\n";
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
