#include "forcesmith/io/potential_reader.hpp"

#include "forcesmith/io/factory.hpp"
#include "forcesmith/io/json_util.hpp"
#include "forcesmith/potentials/analytic_param_defs.hpp"
#include "forcesmith/potentials/analytic_potential.hpp"
#include "forcesmith/potentials/spline.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace forcesmith::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

namespace {

using Maker = std::function<Potential(std::span<const double>, double, double)>;
struct Entry {
  int nparams;
  std::vector<std::string> param_names;
  Maker maker;
};
using Registry = std::unordered_map<std::string_view, Entry>;

template <typename... Names>
  requires(std::convertible_to<Names, std::string_view> && ...)
void add(Registry &reg, int nparams, std::vector<std::string> pnames,
         Maker maker, Names... names) {
  Entry e{nparams, std::move(pnames), std::move(maker)};
  (reg.try_emplace(std::string_view{names}, e), ...);
}

template <std::size_t N>
std::array<double, N> to_arr(std::span<const double> p) {
  std::array<double, N> a{};
  std::ranges::copy_n(p.begin(), static_cast<std::ptrdiff_t>(N), a.begin());
  return a;
}

const Registry &registry() {
  static const Registry reg = [] {
    Registry m;
    m.reserve(64);

    // clang-format off
    add(m,  2, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_lj),                                                          [](auto p, auto lo, auto hi) { return Potential(LennardJones(p[0], p[1],                                      lo, hi)); }, "pair_lj",     "lj"         );
    add(m,  3, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_morse),                                                              [](auto p, auto lo, auto hi) { return Potential(Morse(p[0], p[1], p[2],                                       lo, hi)); }, "morse"                      );
    add(m,  3, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_buckingham),                                                              [](auto p, auto lo, auto hi) { return Potential(Buckingham(p[0], p[1], p[2],                                  lo, hi)); }, "buckingham",  "buck"        );
    add(m,  5, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_born),                                                        [](auto p, auto lo, auto hi) { return Potential(Born(p[0], p[1], p[2], p[3], p[4],                           lo, hi)); }, "born"                       );
    add(m,  2, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_power_decay),                                                                    [](auto p, auto lo, auto hi) { return Potential(PowerDecay(p[0], p[1],                                        lo, hi)); }, "power_decay", "power"       );
    add(m,  2, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_exp_decay),                                                                    [](auto p, auto lo, auto hi) { return Potential(ExpDecay(p[0], p[1],                                          lo, hi)); }, "exp_decay",   "exp"         );
    add(m,  3, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_mexp_decay),                                                               [](auto p, auto lo, auto hi) { return Potential(MexpDecay(p[0], p[1], p[2],                                   lo, hi)); }, "mexp_decay",  "mexp"        );
    add(m,  2, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_harmonic),                                                                   [](auto p, auto lo, auto hi) { return Potential(Harmonic(p[0], p[1],                                          lo, hi)); }, "harmonic"                   );
    add(m,  4, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_universal),                                                           [](auto p, auto lo, auto hi) { return Potential(Universal(p[0], p[1], p[2], p[3],                             lo, hi)); }, "universal"                  );
    add(m,  6, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_eopp),                                                  [](auto p, auto lo, auto hi) { return Potential(Eopp(p[0], p[1], p[2], p[3], p[4], p[5],                     lo, hi)); }, "eopp"                       );
    add(m,  6, {"A","B","C","m","k","phi"},                                                  [](auto p, auto lo, auto hi) { return Potential(EoppExp(p[0], p[1], p[2], p[3], p[4], p[5],                  lo, hi)); }, "eopp_exp",    "eopp_exp_"   );
    add(m,  7, {"A","n","B","m","k","phi","r0"},                                             [](auto p, auto lo, auto hi) { return Potential(Meopp(p[0], p[1], p[2], p[3], p[4], p[5], p[6],              lo, hi)); }, "meopp"                      );
    add(m,  5, {"A","n","m","r0","B"},                                                       [](auto p, auto lo, auto hi) { return Potential(GenLJ(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "gen_lj",      "genlj"       );
    add(m,  7, {"D1","a1","r1","D2","a2","r2","C"},                                          [](auto p, auto lo, auto hi) { return Potential(DoubleMorse(p[0], p[1], p[2], p[3], p[4], p[5], p[6],        lo, hi)); }, "double_morse","dbl_morse"   );
    add(m,  5, {"A","B","r1","C","r2"},                                                      [](auto p, auto lo, auto hi) { return Potential(DoubleExp(p[0], p[1], p[2], p[3], p[4],                      lo, hi)); }, "double_exp",  "dbl_exp"     );
    add(m,  6, {"A","B","C","r0","n","d"},                                                   [](auto p, auto lo, auto hi) { return Potential(Mishin(p[0], p[1], p[2], p[3], p[4], p[5],                   lo, hi)); }, "mishin"                     );
    add(m,  2, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_sqrt),                                                                    [](auto p, auto lo, auto hi) { return Potential(SqrtFunc(p[0], p[1],                                          lo, hi)); }, "sqrt"                       );
    add(m,  1, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_const),                                                                        [](auto p, auto lo, auto hi) { return Potential(ConstFunc(p[0],                                               lo, hi)); }, "const"                      );
    add(m,  3, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_parabola),                                                                [](auto p, auto lo, auto hi) { return Potential(Parabola(p[0], p[1], p[2],                                   lo, hi)); }, "parabola"                   );
    add(m,  5, {"a0","a1","a2","a3","a4"},                                                   [](auto p, auto lo, auto hi) { return Potential(Poly5(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "poly5"                      );
    add(m,  6, {"A","B","p","q","delta","rc"},                                               [](auto p, auto lo, auto hi) { return Potential(StiwWeb2(p[0], p[1], p[2], p[3], p[4], p[5],                 lo, hi)); }, "stiweb_2",    "sw2"         );
    add(m,  2, {"gamma","a"},                                                                [](auto p, auto lo, auto hi) { return Potential(StiwWeb3(p[0], p[1],                                          lo, hi)); }, "stiweb_3",    "sw3"         );
    add(m, 11, {"A","B","lambda","mu","beta","n","c","d","h","R","S"},                        [](auto p, auto lo, auto hi) { return Potential(TersoffPot(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], lo, hi)); }, "tersoff", "tersoff_pot");
    add(m,  2, {"chi","omega"},                                                              [](auto p, auto lo, auto hi) { return Potential(TersoffMix(p[0], p[1],                                        lo, hi)); }, "tersoff_mix"                );
    add(m, 16, {"A","B","lambda","mu","beta","n","c","d","h","R","S","c1","c2","c3","c4","c5"}, [](auto p, auto lo, auto hi) { return Potential(TersoffModPot(to_arr<16>(p),                               lo, hi)); }, "tersoff_mod", "tmod"        );
    add(m,  9, {"A","B","C","D","E","F","G","H","I"},                                        [](auto p, auto lo, auto hi) { return Potential(Kawamura(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], lo, hi)); }, "kawamura"                 );
    add(m, 12, {"A","B","C","D","E","F","G","H","I","J","K","L"},                            [](auto p, auto lo, auto hi) { return Potential(KawamuraMix(to_arr<12>(p),                                    lo, hi)); }, "kawamura_mix"               );
    add(m,  2, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_softshell),                                                                    [](auto p, auto lo, auto hi) { return Potential(Softshell(p[0], p[1],                                         lo, hi)); }, "softshell",   "soft"        );
    add(m,  3, {"A","B","C"},                                                                [](auto p, auto lo, auto hi) { return Potential(ExpPlus(p[0], p[1], p[2],                                    lo, hi)); }, "exp_plus",    "expplus"     );
    add(m,  5, {"A","B","C","D","E"},                                                        [](auto p, auto lo, auto hi) { return Potential(Strmm(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "strmm"                      );

    // Smooth-cutoff (`_sc`) variants: SmoothCutoff decorator wraps the base and
    // multiplies by apot_cutoff(r,rmax,h); `h` (switching width) is the appended
    // last parameter. Match forcesmith's *_sc names. Adding more is a one-liner.
    add(m,  3, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_lj_sc),                                                      [](auto p, auto lo, auto hi) { return Potential(SmoothCutoff(LennardJones(p[0], p[1],            lo, hi), p[2])); }, "lj_sc",       "pair_lj_sc"  );
    add(m,  4, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_morse_sc),                                                          [](auto p, auto lo, auto hi) { return Potential(SmoothCutoff(Morse(p[0], p[1], p[2],            lo, hi), p[3])); }, "morse_sc"                   );
    add(m,  3, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_exp_decay_sc),                                                                [](auto p, auto lo, auto hi) { return Potential(SmoothCutoff(ExpDecay(p[0], p[1],               lo, hi), p[2])); }, "exp_decay_sc","exp_sc"       );
    add(m,  7, FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_eopp_sc),                                              [](auto p, auto lo, auto hi) { return Potential(SmoothCutoff(Eopp(p[0], p[1], p[2], p[3], p[4], p[5], lo, hi), p[6])); }, "eopp_sc"                    );
    // clang-format on

    return m;
  }();
  return reg;
}

template <class T> leaf::result<T> field(const json &p, const char *key) {
  if (!p.contains(key)) {
    return leaf::new_error(ParseError{std::string("missing '") + key + "'", 0});
  }
  return p.at(key).get<T>();
}

// "type" → the matching analytic Entry (or "unknown analytic function").
leaf::result<const Entry *> find_analytic(const std::string &type_name) {
  const Registry &reg = registry();
  if (auto it = reg.find(type_name); it != reg.end()) {
    return &it->second;
  }
  return leaf::new_error(
      ParseError{"unknown analytic function: " + type_name, 0});
}

// One parsed analytic parameter: its start value plus optional box constraint
// and fixed flag. min/max default to ±∞ (unbounded), fixed to false.
struct ParamSpec {
  double value = 0.0;
  double min = -std::numeric_limits<double>::infinity();
  double max = std::numeric_limits<double>::infinity();
  bool fixed = false;
};

// Each named parameter may be either a bare number (→ value only, unbounded) or
// an object {"value": x, "min": lo, "max": hi, "fixed": bool}. Bare numbers keep
// the legacy format working unchanged.
leaf::result<std::vector<ParamSpec>>
gather_param_specs(const json &p, const std::vector<std::string> &names,
                   const std::string &type_name) {
  std::vector<ParamSpec> specs;
  specs.reserve(names.size());
  for (const auto &name : names) {
    if (!p.contains(name)) {
      return leaf::new_error(ParseError{
          "missing parameter '" + name + "' for type " + type_name, 0});
    }
    const json &jv = p.at(name);
    ParamSpec s;
    if (jv.is_object()) {
      if (!jv.contains("value")) {
        return leaf::new_error(ParseError{
            "parameter '" + name + "' (" + type_name + ") missing 'value'", 0});
      }
      s.value = jv.at("value").get<double>();
      s.min = jv.value("min", s.min);
      s.max = jv.value("max", s.max);
      s.fixed = jv.value("fixed", false);
      if (s.min > s.max) {
        return leaf::new_error(ParseError{
            "parameter '" + name + "' (" + type_name + ") has min > max", 0});
      }
    } else {
      s.value = jv.get<double>();
    }
    specs.push_back(s);
  }
  return specs;
}

leaf::result<std::vector<double>> knot_values(const json &p) {
  if (!p.contains("knots") || !p.at("knots").is_array() ||
      p.at("knots").size() < 2) {
    return leaf::new_error(ParseError{
        "tabulated potential: 'knots' must be an array of >= 2 values", 0});
  }
  std::vector<double> y;
  y.reserve(p.at("knots").size());
  std::ranges::transform(
      p.at("knots"), std::back_inserter(y),
      [](const auto &knot) { return knot.template get<double>(); });
  return y;
}

// ── Per-spec creators ─────────────────────────────────────────────────────
// Each builds ONE Potential from a single self-describing JSON spec object;
// these are the concrete products the format factory hands out.

// analytic: read "type" → its registry entry → radial range → its named
// parameters (value + optional per-parameter [min,max] box and fixed flag).
leaf::result<Potential> make_analytic(const json &p) {
  BOOST_LEAF_AUTO(type_name, field<std::string>(p, "type"));
  BOOST_LEAF_AUTO(entry, find_analytic(type_name));
  BOOST_LEAF_AUTO(rmin, field<double>(p, "rmin"));
  BOOST_LEAF_AUTO(rmax, field<double>(p, "rmax"));
  BOOST_LEAF_AUTO(specs, gather_param_specs(p, entry->param_names, type_name));

  std::vector<double> values;
  values.reserve(specs.size());
  std::ranges::transform(specs, std::back_inserter(values),
                         [](const ParamSpec &s) { return s.value; });

  Potential pot = entry->maker(values, rmin, rmax);
  for (std::size_t i = 0; i < specs.size(); ++i) {
    pot.set_bounds(i, specs[i].min, specs[i].max);
    if (specs[i].fixed) {
      pot.set_fixed(i, true);
    }
  }
  return pot;
}

// tabulated: read bounds and knots → spread the knots over a uniform grid.
leaf::result<Potential> make_tabulated(const json &p) {
  BOOST_LEAF_AUTO(rmin, field<double>(p, "rmin"));
  BOOST_LEAF_AUTO(rmax, field<double>(p, "rmax"));
  BOOST_LEAF_AUTO(y, knot_values(p));

  const std::size_t n = y.size();
  const double h = (rmax - rmin) / static_cast<double>(n - 1);
  std::vector<double> x(n);
  std::size_t kcount = 0;
  std::ranges::generate(
      x, [&] { return rmin + static_cast<double>(kcount++) * h; });

  SplinePotential sp(std::move(x), std::move(y));
  if (p.value("fixed", false)) {
    for (std::size_t k = 0; k < n; ++k) {
      sp.set_fixed(k, true);
    }
  }
  return Potential(std::move(sp));
}

template <class IdentifierType, class AbstractProduct>
struct UnsupportedFormatError {
  static leaf::result<AbstractProduct> OnUnknownType(const IdentifierType &id) {
    return leaf::new_error(ParseError{
        "unsupported potential format '" + std::string(id) + "'", 0});
  }
};

// The concrete potential-format factory: format string → per-spec creator.
using PotentialFactory =
    ForcesmithFactory<Potential, std::string,
                  leaf::result<Potential> (*)(const json &),
                  UnsupportedFormatError>;

const PotentialFactory &format_factory() {
  static const PotentialFactory factory = [] {
    PotentialFactory f;
    f.Register("analytic", &make_analytic);
    f.Register("tabulated", &make_tabulated);
    return f;
  }();
  return factory;
}

// Build ONE potential from a single self-describing JSON spec, dispatching on
// the keys present rather than a top-level "format" string. Shared by
// Potential::from_text and the single-entry parse path.
leaf::result<Potential> one_potential(const json &p) {
  if (!p.is_object()) {
    return leaf::new_error(
        ParseError{"potential spec must be a JSON object", 0});
  } else if (p.contains("type")) {
    return make_analytic(p);
  } else if (p.contains("knots")) {
    return make_tabulated(p);
  }
  return leaf::new_error(ParseError{
      "potential spec has neither 'type' (analytic) nor 'knots' (tabulated)",
      0});
}

} // anonymous namespace

leaf::result<std::vector<Potential>> parse_potential(std::string_view input) {
  auto fail = [](std::string msg) -> leaf::result<std::vector<Potential>> {
    return leaf::new_error(ParseError{std::move(msg), 0});
  };

  return catch_json([&]() -> leaf::result<std::vector<Potential>> {
    const json j = json::parse(input);

    if (!j.contains("format")) {
      return fail("missing 'format' key");
    }
    const std::string fmt = j["format"].get<std::string>();

    if (!j.contains("potentials") || !j["potentials"].is_array()) {
      return fail("missing or invalid 'potentials' array");
    }
    const auto &pots_arr = j["potentials"];

    if (!format_factory().IsRegistered(fmt)) {
      return UnsupportedFormatError<std::string,
                                    std::vector<Potential>>::OnUnknownType(fmt);
    }

    std::vector<Potential> potentials;
    potentials.reserve(pots_arr.size());
    for (const auto &p : pots_arr) {
      BOOST_LEAF_AUTO(pot, format_factory().CreateObject(fmt, p));
      potentials.push_back(std::move(pot));
    }
    return potentials;
  });
}

std::optional<std::size_t> analytic_param_index(std::string_view type,
                                                std::string_view param) {
  const Registry &reg = registry();
  auto it = reg.find(type);
  if (it == reg.end()) {
    return std::nullopt;
  }
  const auto &names = it->second.param_names;
  if (auto it2 = std::ranges::find(names, param); it2 != names.end()) {
    return static_cast<std::size_t>(it2 - names.begin());
  }
  return std::nullopt;
}

} // namespace forcesmith::io

namespace forcesmith {

boost::leaf::result<Potential> Potential::from_text(std::string_view text) {
  return io::catch_json([&] {
    auto j = nlohmann::json::parse(text);
    return io::one_potential(j);
  });
}

boost::leaf::result<Potential>
Potential::from_file(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        io::ParseError{"cannot open potential file: " + path.string(), 0});
  }
  std::string text((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  return Potential::from_text(text);
}

} // namespace forcesmith
