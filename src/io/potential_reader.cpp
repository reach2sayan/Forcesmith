#include "forcesmith/io/potential_reader.hpp"

#include "forcesmith/core/json.hpp"
#include "forcesmith/io/file.hpp"
#include "forcesmith/potentials/analytic_potential.hpp"
#include "forcesmith/potentials/spline.hpp"

#include <boost/leaf/error.hpp>
#include <boost/mp11/algorithm.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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

using RadialPotentialMaker =
    std::function<RadialPotential(std::span<const double>, double, double)>;
struct Entry {
  int nparams;
  std::vector<std::string> param_names;
  RadialPotentialMaker maker;
};
using Registry = std::unordered_map<std::string_view, Entry>;

const Registry &registry() {
  static const Registry reg = [] {
    Registry m;
    m.reserve(64);
    boost::mp11::mp_for_each<
        boost::mp11::mp_transform<boost::mp11::mp_identity, AnalyticForms>>(
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Entry e{static_cast<int>(T::num_params),
                  std::vector<std::string>(T::param_names.begin(),
                                           T::param_names.end()),
                  [](std::span<const double> p, double lo, double hi) {
                    std::array<double, T::num_params> a{};
                    std::ranges::copy_n(
                        p.begin(), static_cast<std::ptrdiff_t>(T::num_params),
                        a.begin());
                    return RadialPotential(T(a, lo, hi));
                  }};
          for (std::string_view name : T::names) {
            if (!name.empty()) { // an unused alias slot
              m.try_emplace(name, e);
            }
          }
        });
    return m;
  }();
  return reg;
}

template <class T>
  requires requires(const json &j, T &v) { j.get_to(v); }
leaf::result<T> field(const json &p, const char *key) {
  if (!p.contains(key)) {
    return leaf::new_error(ParseError{std::string("missing '") + key + "'", 0});
  }
  return p.at(key).get<T>();
}

leaf::result<const Entry *> find_analytic(const std::string &type_name) {
  const Registry &reg = registry();
  if (auto it = reg.find(type_name); it != reg.end()) {
    return &it->second;
  }
  return leaf::new_error(
      ParseError{"unknown analytic function: " + type_name, 0});
}

struct ParamSpec {
  double value = 0.0;
  double min = -std::numeric_limits<double>::infinity();
  double max = std::numeric_limits<double>::infinity();
  bool fixed = false;
};

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

leaf::result<RadialPotential> make_analytic(const json &p) {
  BOOST_LEAF_AUTO(type_name, field<std::string>(p, "type"));
  BOOST_LEAF_AUTO(entry, find_analytic(type_name));
  BOOST_LEAF_AUTO(rmin, field<double>(p, "rmin"));
  BOOST_LEAF_AUTO(rmax, field<double>(p, "rmax"));
  BOOST_LEAF_AUTO(specs, gather_param_specs(p, entry->param_names, type_name));

  std::vector<double> values;
  values.reserve(specs.size());
  std::ranges::transform(specs, std::back_inserter(values),
                         [](const ParamSpec &s) { return s.value; });

  RadialPotential pot = entry->maker(values, rmin, rmax);
  for (const auto &[i, s] : specs | std::views::enumerate) {
    pot.set_bounds(static_cast<std::size_t>(i), s.min, s.max);
    if (s.fixed) {
      pot.set_fixed(static_cast<std::size_t>(i), true);
    }
  }
  return pot;
}

leaf::result<RadialPotential> make_tabulated(const json &p) {
  BOOST_LEAF_AUTO(rmin, field<double>(p, "rmin"));
  BOOST_LEAF_AUTO(rmax, field<double>(p, "rmax"));
  BOOST_LEAF_AUTO(y, knot_values(p));

  const std::size_t n = y.size();
  if (n < 2) {
    return leaf::new_error(
        ParseError{"tabulated potential needs at least 2 knots", 0});
  }
  if (!(rmax > rmin)) {
    return leaf::new_error(ParseError{
        "tabulated potential requires rmax > rmin (got rmin=" +
            std::to_string(rmin) + ", rmax=" + std::to_string(rmax) + ")",
        0});
  }
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
  return RadialPotential(std::move(sp));
}

using PotentialMaker = leaf::result<RadialPotential> (*)(const json &);

constexpr std::array<std::pair<std::string_view, PotentialMaker>, 2> kFormats{
    {{"analytic", &make_analytic}, {"tabulated", &make_tabulated}}};

leaf::result<PotentialMaker> maker_for(std::string_view format) {
  const auto it = std::ranges::find(kFormats, format,
                                    &std::pair<std::string_view,
                                               PotentialMaker>::first);
  if (it == kFormats.end()) {
    return leaf::new_error(ParseError{
        "unsupported potential format '" + std::string(format) + "'", 0});
  }
  return it->second;
}

leaf::result<RadialPotential> one_potential(const json &p) {
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

leaf::result<std::vector<RadialPotential>>
parse_potential(std::string_view input) {
  auto fail =
      [](std::string msg) -> leaf::result<std::vector<RadialPotential>> {
    return leaf::new_error(ParseError{std::move(msg), 0});
  };

  return catch_json([&]() -> leaf::result<std::vector<RadialPotential>> {
    const json j = json::parse(input);

    if (!j.contains("format")) {
      return fail("missing 'format' key");
    }
    const std::string fmt = j["format"].get<std::string>();

    if (!j.contains("potentials") || !j["potentials"].is_array()) {
      return fail("missing or invalid 'potentials' array");
    }
    const auto &pots_arr = j["potentials"];

    BOOST_LEAF_AUTO(make, maker_for(fmt));

    std::vector<RadialPotential> potentials;
    potentials.reserve(pots_arr.size());
    for (const json &p : pots_arr) {
      BOOST_LEAF_AUTO(pot, make(p));
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

boost::leaf::result<RadialPotential>
RadialPotential::from_text(std::string_view text) {
  return io::catch_json([&] {
    auto j = nlohmann::json::parse(text);
    return io::one_potential(j);
  });
}

boost::leaf::result<RadialPotential>
RadialPotential::from_file(const std::filesystem::path &path) {
  BOOST_LEAF_AUTO(text, io::read_file(path));
  return RadialPotential::from_text(text);
}

} // namespace forcesmith
