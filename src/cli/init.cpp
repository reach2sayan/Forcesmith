#include "forcesmith/cli/init.hpp"

#include "forcesmith/core/families.hpp"
#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/io/config_reader.hpp" // io::ParseError
#include "forcesmith/io/grammar.hpp"
#include "forcesmith/io/write_model.hpp"
#include "forcesmith/potentials/analytic_param_defs.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <boost/leaf/result.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace po = boost::program_options;
namespace leaf = boost::leaf;
namespace mp11 = boost::mp11;
using json = nlohmann::json;

namespace forcesmith::cli::init {
namespace {

struct Args {
  std::string model;
  std::string out;
  int ntypes = 1;
  double cutoff = 6.0;
  std::string
      functions; // makeapot-style "N*name,…"; empty → per-region default
  int n_max = 6;
  int l_max = 6;
  double sigma = 0.5;
  int g1 = 0;
  std::vector<double> g2_eta;
  std::vector<double> g2_rs;
  int k2_n = 50;
  int k3_n = 50;
  bool drop_k2 = false;
  bool drop_k3 = false;
  bool bias_free = false;
  unsigned seed = 42; // RNG seed for coefficient init (reproducible startpot)
};

struct InitError {
  std::string message;
};

[[nodiscard]] leaf::error_id fail(std::string msg) {
  return leaf::new_error(InitError{std::move(msg)});
}

namespace bp = boost::parser;
const auto ident = +(bp::char_('a', 'z') | bp::char_('A', 'Z') |
                     bp::char_('0', '9') | bp::char_('_'));
const auto item = (bp::uint_ >> '*' | bp::attr(1u)) >> ident;
const auto function_spec = item % ',';

[[nodiscard]] leaf::result<std::vector<std::string>>
expand_functions(std::string_view spec) {
  BOOST_LEAF_AUTO(items,
                  io::parse_or_error(function_spec, spec, "--functions"));
  for (const auto &[count, name] : items) {
    if (count < 1) {
      return fail("function multiplier must be ≥ 1 in '" +
                  std::to_string(count) + "*" + name + "'");
    }
  }
  return items | std::views::transform([](const auto &it) {
           const auto &[count, name] = it;
           return std::views::repeat(name, count);
         }) |
         std::views::join | std::ranges::to<std::vector<std::string>>();
}

enum class Count { PerPair, PerType };
enum class Domain { Radial, Cosine }; // [0, rcut] or the g(cosθ) span [-1, 1]

struct RegionSpec {
  std::string_view key;
  Count count;
  std::string_view default_fn;
  Domain domain = Domain::Radial;
};

template <class T> struct AnalyticLayout;

using R = RegionSpec;
inline constexpr Count kPer = Count::PerPair, kType = Count::PerType;

template <> struct AnalyticLayout<PairForceCalculator> {
  static constexpr std::array regions{R{"", kPer, "lj"}};
};
template <> struct AnalyticLayout<EAMForceCalculator> {
  static constexpr std::array regions{R{"pair", kPer, "lj"},
                                      R{"density", kType, "exp_decay"},
                                      R{"embedding", kType, "sqrt"}};
};
template <> struct AnalyticLayout<ADPForceCalculator> {
  static constexpr std::array regions{
      R{"pair", kPer, "lj"}, R{"density", kType, "exp_decay"},
      R{"embedding", kType, "sqrt"}, R{"dipole", kPer, "exp_decay"},
      R{"quadrupole", kPer, "exp_decay"}};
};
template <> struct AnalyticLayout<AngularForceCalculator> {
  static constexpr std::array regions{
      R{"pair", kPer, "lj"}, R{"radial", kPer, "exp_decay"},
      R{"angular", kType, "parabola", Domain::Cosine}};
};

template <class T>
concept CAnalyticFamily = requires { AnalyticLayout<T>::regions; };

template <class T>
concept CBondOrderFamily = std::same_as<T, TersoffForceCalculator> ||
                           std::same_as<T, StiwebForceCalculator>;

[[nodiscard]] leaf::result<json> one_analytic(const std::string &fn,
                                              double rmin, double rmax) {
  const std::span<const AnalyticParamDef> defs = analytic_defaults(fn);
  if (defs.empty()) {
    return fail("unknown analytic function '" + fn +
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

void report_written(const Args &a) {
  std::cout << "wrote " << a.model << " startpot to " << a.out << "\n";
}

[[nodiscard]] leaf::result<void> write_json_file(const Args &a, const json &j) {
  std::ofstream f(a.out);
  if (!f) {
    return fail("cannot open output file '" + a.out + "'");
  }
  f << j.dump(2) << "\n";
  report_written(a);
  return {};
}

[[nodiscard]] leaf::result<void> write_built_model(const Args &a,
                                                   const ForceCalculator &m) {
  BOOST_LEAF_CHECK(io::write_model(m, a.out, "native"));
  report_written(a);
  return {};
}

template <CAnalyticFamily T>
[[nodiscard]] leaf::result<void> scaffold(const Args &a) {
  const auto nt = static_cast<std::size_t>(a.ntypes);
  const std::size_t paircol = nt * (nt + 1) / 2;
  constexpr auto &regions = AnalyticLayout<T>::regions;

  const auto count_of = [&](const RegionSpec &r) {
    return r.count == Count::PerPair ? paircol : nt;
  };
  const std::size_t total = std::ranges::fold_left(
      regions | std::views::transform(count_of), std::size_t{0}, std::plus<>{});

  std::vector<std::string> flat;
  if (!a.functions.empty()) {
    BOOST_LEAF_ASSIGN(flat, expand_functions(a.functions));
    if (flat.size() != total) {
      return fail("--functions has " + std::to_string(flat.size()) +
                  " functions but " + a.model +
                  " (ntypes=" + std::to_string(a.ntypes) + ") needs " +
                  std::to_string(total));
    }
  }

  json j;
  j["model"] = a.model;
  j["ntypes"] = a.ntypes;
  std::size_t idx = 0;
  for (const RegionSpec &r : regions) {
    const double rmin = r.domain == Domain::Cosine ? -1.0 : 0.0;
    const double rmax = r.domain == Domain::Cosine ? 1.0 : a.cutoff;
    json pots = json::array();
    for (std::size_t s = 0; s < count_of(r); ++s) {
      const std::string &fn =
          flat.empty() ? std::string(r.default_fn) : flat[idx++];
      BOOST_LEAF_AUTO(p, one_analytic(fn, rmin, rmax));
      pots.push_back(std::move(p));
    }
    if (r.key.empty()) {
      j["format"] = "analytic";
      j["potentials"] = std::move(pots);
    } else {
      j[std::string(r.key)] = {{"format", "analytic"},
                               {"potentials", std::move(pots)}};
    }
  }
  return write_json_file(a, j);
}

template <CBondOrderFamily T>
[[nodiscard]] leaf::result<void> scaffold(const Args &a) {
  const auto nt = static_cast<std::size_t>(a.ntypes);
  const std::size_t paircol = nt * (nt + 1) / 2;

  T c;
  c.ntypes = nt;
  c.params.reserve(nt);
  for (std::size_t s = 0; s < paircol; ++s) {
    typename decltype(c.params)::value_type p;
    if constexpr (std::same_as<T, TersoffForceCalculator>) {
      p.R = Param{0.8 * a.cutoff};
      p.S = Param{a.cutoff};
    } else {
      p.a1 = Param{a.cutoff};
      p.a2 = Param{a.cutoff};
    }
    c.params.emplace_back(p);
  }
  if constexpr (std::same_as<T, StiwebForceCalculator>) {
    c.lambda.assign(nt * paircol, Param{2.0});
  }
  return write_built_model(a, ForceCalculator{std::move(c)});
}

constexpr double kHeadInitStd = 0.01;

template <CMLFamily Model> void attach_init_heads(Model &m, const Args &a) {
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

template <class T>
  requires is_ml_family<T>
[[nodiscard]] leaf::result<void> scaffold(const Args &a) {
  T m;
  m.ntypes = static_cast<std::size_t>(a.ntypes);
  m.rcut = a.cutoff;

  if constexpr (std::same_as<T, SoapModel>) {
    m.n_max = a.n_max;
    m.l_max = a.l_max;
    m.sigma = a.sigma;
    m.init_radial_basis();
  } else if constexpr (std::same_as<T, ACSF>) {
    m.g1 = static_cast<std::size_t>(a.g1);
    for (std::size_t k = 0; k < a.g2_eta.size(); ++k) {
      const double rs = k < a.g2_rs.size() ? a.g2_rs[k] : 0.0;
      m.radial.push_back({a.g2_eta[k], rs});
    }
    if (m.descriptor_size() == 0) {
      return fail("acsf needs at least one channel: set --g1 and/or --g2-eta");
    }
  } else {
    m.k2 = a.drop_k2 ? std::nullopt
                     : std::optional<LMBTR::Grid>{{0.0, a.cutoff, a.k2_n, 0.3}};
    m.k3 = a.drop_k3 ? std::nullopt
                     : std::optional<LMBTR::Grid>{{-1.0, 1.0, a.k3_n, 0.1}};
    if (m.descriptor_size() == 0) {
      return fail("lmbtr needs k2 and/or k3 with n > 0");
    }
  }

  attach_init_heads(m, a);
  return write_built_model(a, ForceCalculator{std::move(m)});
}

template <class T>
concept CScaffoldable =
    CAnalyticFamily<T> || CBondOrderFamily<T> || is_ml_family<T>;

[[nodiscard]] leaf::result<void> scaffold_named(const Args &a) {
  leaf::result<void> r;
  bool matched = false;
  mp11::mp_for_each<mp11::mp_transform<mp11::mp_identity, ModelFamilies>>(
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (CScaffoldable<T>) {
          if (!matched && a.model == family_name<T>) {
            matched = true;
            r = scaffold<T>(a);
          }
        }
      });
  if (!matched) {
    return fail("unknown --model '" + a.model + "'");
  }
  return r;
}

[[nodiscard]] std::string model_choices() {
  std::string s;
  mp11::mp_for_each<mp11::mp_transform<mp11::mp_identity, ModelFamilies>>(
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (CScaffoldable<T>) {
          s += (s.empty() ? "" : " | ");
          s += family_name<T>;
        }
      });
  return s;
}

[[nodiscard]] leaf::result<void> validate(const Args &a) {
  if (a.model.empty()) {
    return fail("--model is required");
  }
  if (a.out.empty()) {
    return fail("--out is required");
  }
  if (a.ntypes < 1) {
    return fail("--ntypes must be ≥ 1");
  }
  if (a.cutoff <= 0.0) {
    return fail("--cutoff must be > 0");
  }
  return {};
}

} // namespace

int run(int argc, char *argv[]) {
  Args a;
  po::options_description desc(
      "forcesmith init — scaffold a fresh startpot (a JSON-native makeapot)");
  desc.add_options()("help,h", "show this message")(
      "model,m", po::value(&a.model), model_choices().c_str())(
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

  return leaf::try_handle_all(
      [&]() -> leaf::result<int> {
        BOOST_LEAF_CHECK(validate(a));
        BOOST_LEAF_CHECK(scaffold_named(a));
        return 0;
      },
      [](const InitError &e) {
        std::cerr << "error: " << e.message << "\n";
        return 1;
      },
      [](const io::ParseError &e) {
        std::cerr << "error: " << e.message << "\n";
        return 1;
      },
      [] {
        std::cerr << "error: unknown failure\n";
        return 1;
      });
}

} // namespace forcesmith::cli::init
