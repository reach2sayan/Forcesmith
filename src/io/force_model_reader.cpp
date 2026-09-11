#include "forcesmith/io/force_model_reader.hpp"
#include "forcesmith/core/families.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/core/json.hpp"
#include "forcesmith/io/potential_reader.hpp"
#include "forcesmith/io/schema.hpp"

#include <boost/leaf/error.hpp>
#include <boost/mp11/algorithm.hpp>
#include <nlohmann/json.hpp>

#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace forcesmith::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

namespace {

using Fail = leaf::result<ForceCalculator>;

leaf::error_id fail(std::string msg) {
  return leaf::new_error(ParseError{std::move(msg), 0});
}
leaf::error_id fail(std::string_view label, std::string msg) {
  return fail(std::string(label) + ": " + std::move(msg));
}

template <class T>
concept CPotentialTable = requires(T &t, typename T::value_type v) {
  t.reserve(std::size_t{});
  t.push_back(std::move(v));
};

template <CPotentialTable Table>
leaf::result<void> fill_table(Table &table, const json &sub,
                              std::size_t ntypes, std::string_view label) {
  constexpr bool per_pair =
      std::same_as<Table, SymmetricMatrix<typename Table::value_type>>;
  const std::size_t want = per_pair ? ntypes * (ntypes + 1) / 2 : ntypes;

  BOOST_LEAF_AUTO(pots, parse_potential(sub.dump()));
  if (pots.size() != want) {
    return fail(label, "expected " + std::to_string(want) +
                           " potentials for ntypes=" + std::to_string(ntypes) +
                           ", got " + std::to_string(pots.size()));
  }
  table.reserve(ntypes);
  std::ranges::move(pots, std::back_inserter(table));
  return {};
}

leaf::result<std::vector<GlobalParam>> build_globals(
    json &root,
    std::span<const std::pair<json *, GlobalParam::Link::LinkRegion>> regions) {
  std::vector<GlobalParam> globals;
  std::unordered_map<std::string, std::size_t> by_name;
  if (root.contains("globals")) {
    for (const auto &item : root["globals"].items()) {
      const json &spec = item.value();
      if (!spec.contains("value")) {
        return fail("globals", "global '" + item.key() + "' missing 'value'");
      }
      GlobalParam g;
      g.value =
          Param{spec.at("value").get<double>(), spec.value("fixed", false)};
      by_name.emplace(item.key(), globals.size());
      globals.push_back(std::move(g));
    }
  }

  for (auto [sub, region] : regions) {
    if (!sub->contains("potentials") || !(*sub)["potentials"].is_array())
      continue;
    json &arr = (*sub)["potentials"];
    for (std::size_t i = 0; i < arr.size(); ++i) {
      json &pot = arr[i];
      if (!pot.is_object() || !pot.contains("type")) {
        continue;
      }
      const std::string type = pot["type"].get<std::string>();
      auto refs =
          pot.items() | std::views::filter([](const auto &el) {
            return el.value().is_object() && el.value().contains("global");
          }) |
          std::views::transform([](const auto &el) { return el.key(); });

      std::vector<std::string> ref_keys;
      std::ranges::copy(refs, std::back_inserter(ref_keys));

      for (const auto &key : ref_keys) {
        const std::string gname = pot[key]["global"].get<std::string>();
        auto git = by_name.find(gname);
        if (git == by_name.end()) {
          return fail("globals",
                      "reference to undefined global '" + gname + "'");
        }
        auto slot = analytic_param_index(type, key);
        if (!slot) {
          return fail("globals", "global ref on parameter '" + key +
                                     "' not valid for analytic type '" + type +
                                     "'");
        }
        globals[git->second].links.push_back({region, i, *slot});
        pot[key] =
            globals[git->second].value.value; // replace ref → seed number
      }
    }
  }
  return globals;
}

GlobalParam::Link::LinkRegion region_of(std::string_view key) {
  using enum GlobalParam::Link::LinkRegion;
  if (key == "density") {
    return DENSITY;
  }
  if (key == "embedding") {
    return EMBEDDING;
  }
  return PAIR; // "" (the bare pair model's root) and "pair"
}

template <class T>
concept CHasGlobals = requires(T &t) { t.globals; };

template <CTabulated T>
leaf::result<ForceCalculator> build_tabulated(json &j, std::size_t ntypes) {
  T calc;
  calc.ntypes = ntypes;

  if constexpr (CHasGlobals<T>) {
    std::vector<std::pair<json *, GlobalParam::Link::LinkRegion>> regions;
    for_each_table(calc, [&](auto &, std::string_view key) {
      regions.emplace_back(key.empty() ? &j : &j.at(std::string(key)),
                           region_of(key));
    });
    BOOST_LEAF_AUTO(globals, build_globals(j, std::span(regions)));
    calc.globals = std::move(globals);
  }

  leaf::result<void> status{};
  for_each_table(calc, [&](auto &table, std::string_view key) {
    if (!status) {
      return; // a previous section already failed
    }
    const std::string label =
        std::string(family_name<T>) + (key.empty() ? "" : "." + std::string(key));
    status = fill_table(table, key.empty() ? j : j.at(std::string(key)), ntypes,
                        label);
  });
  BOOST_LEAF_CHECK(std::move(status));

  if constexpr (CHasGlobals<T>) {
    calc.finalize_globals();
  }
  return ForceCalculator{std::move(calc)};
}

template <class T, class P>
leaf::result<void> read_param_blocks(SymmetricMatrix<P> &out, const json &j,
                                     std::size_t ntypes) {
  const json &arr = j.at("potentials");
  const std::size_t paircol = ntypes * (ntypes + 1) / 2;
  if (!arr.is_array() || arr.size() != paircol) {
    return fail(family_name<T>,
                "expected " + std::to_string(paircol) +
                    " entries for ntypes=" + std::to_string(ntypes));
  }
  out.reserve(ntypes);
  for (const json &p : arr) {
    out.emplace_back(p.get<P>());
  }
  return {};
}

leaf::result<ForceCalculator> build_tersoff(json &j, std::size_t ntypes) {
  TersoffForceCalculator calc;
  calc.ntypes = ntypes;
  BOOST_LEAF_CHECK(
      (read_param_blocks<TersoffForceCalculator>(calc.params, j, ntypes)));
  const json &arr = j.at("potentials");
  std::size_t i = 0;
  for (TersoffParams &p : calc.params) {
    if (arr[i++].contains("omega")) {
      p.omega.fixed = false;
    }
  }
  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_stiweb(json &j, std::size_t ntypes) {
  StiwebForceCalculator calc;
  calc.ntypes = ntypes;
  BOOST_LEAF_CHECK(
      (read_param_blocks<StiwebForceCalculator>(calc.params, j, ntypes)));

  const json &lam = j.at("lambda");
  const std::size_t want = ntypes * (ntypes * (ntypes + 1) / 2);
  if (!lam.is_array() || lam.size() != want) {
    return fail("stiweb", "expected " + std::to_string(want) +
                              " lambda entries for ntypes=" +
                              std::to_string(ntypes) + ", got " +
                              std::to_string(lam.is_array() ? lam.size() : 0));
  }
  calc.lambda.reserve(want);
  std::ranges::transform(lam, std::back_inserter(calc.lambda),
                         [](const json &l) { return l.get<Param>(); });
  return ForceCalculator{std::move(calc)};
}

leaf::result<EnergyHead> parse_linear_head(const json &h, std::size_t S) {
  if (!h.contains("coeffs") || !h["coeffs"].is_array()) {
    return fail("ml", "linear head missing 'coeffs' array");
  }
  if (h["coeffs"].size() != S) {
    return fail("ml", "head coeffs length " +
                          std::to_string(h["coeffs"].size()) +
                          " != descriptor size " + std::to_string(S));
  }
  const bool fixed = h.value("fixed", false);
  LinearHead lh;
  lh.coeffs.reserve(S);
  for (const auto &c : h["coeffs"]) {
    lh.coeffs.push_back(Param{c.get<double>(), fixed});
  }
  lh.bias = Param{h.value("bias", 0.0), h.value("bias_fixed", true)};
  return EnergyHead{std::move(lh)};
}

leaf::result<EnergyHead> parse_head(const json &h, std::size_t S) {
  const std::string htype = h.value("type", std::string("linear"));
  if (htype == "linear") {
    return parse_linear_head(h, S);
  }
  return fail("ml", "unknown head type '" + htype + "'");
}

template <CMLFamily Model>
leaf::result<ForceCalculator> finish_ml(Model calc, json &j, std::size_t ntypes,
                                        std::size_t S) {
  if (!j.contains("heads") || !j["heads"].is_array()) {
    return fail("ml", "missing 'heads' array");
  }
  const auto &heads = j["heads"];
  if (heads.size() != ntypes) {
    return fail("ml", "expected " + std::to_string(ntypes) + " heads, got " +
                          std::to_string(heads.size()));
  }
  calc.heads.reserve(ntypes);
  for (const json &h : heads) {
    BOOST_LEAF_AUTO(head, parse_head(h, S));
    calc.heads.emplace_back(std::move(head));
  }
  if (j.contains("standardization")) {
    const auto &st = j["standardization"];
    if (!st.is_array() || st.size() != ntypes) {
      return fail("ml", "standardization must be an array of " +
                            std::to_string(ntypes) + " entries, got " +
                            std::to_string(st.is_array() ? st.size() : 0));
    }
    calc.mean_.reserve(ntypes);
    calc.inv_std_.reserve(ntypes);
    for (const auto &e : st) {
      const auto mean = e.value("mean", std::vector<double>{});
      const auto iv = e.value("inv_std", std::vector<double>{});
      if (mean.size() != iv.size() || (!mean.empty() && mean.size() != S)) {
        return fail("ml", "standardization mean/inv_std length " +
                              std::to_string(mean.size()) +
                              " != descriptor size " + std::to_string(S));
      }
      calc.mean_.emplace_back(Eigen::Map<const Eigen::VectorXd>(
          mean.data(), static_cast<Eigen::Index>(mean.size())));
      calc.inv_std_.emplace_back(Eigen::Map<const Eigen::VectorXd>(
          iv.data(), static_cast<Eigen::Index>(iv.size())));
    }
  }
  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_ml(json &j, std::size_t ntypes) {
  if (!j.contains("descriptor") || !j["descriptor"].is_object()) {
    return fail("ml", "missing 'descriptor' object");
  }
  const json &desc = j["descriptor"];
  if (!desc.contains("type") || !desc["type"].is_string()) {
    return fail("ml", "descriptor missing string 'type' "
                      "(expected acsf/soap/lmbtr)");
  }
  const std::string dtype = desc["type"].get<std::string>();

  if (dtype == "symmetry_functions" || dtype == "acsf") {
    ACSF calc;
    calc.ntypes = ntypes;
    calc.rcut = desc.value("rcut", 6.0);
    if (desc.contains("g1")) {
      const json &g1 = desc["g1"];
      calc.g1 = g1.is_number()  ? g1.get<std::size_t>()
                : g1.is_array() ? g1.size()
                                : std::size_t{0};
    }
    if (desc.contains("g2") && desc["g2"].is_array()) {
      for (const auto &g : desc["g2"]) {
        calc.radial.push_back({g.at("eta").get<double>(), g.value("rs", 0.0)});
      }
    }
    if (desc.contains("g3") && desc["g3"].is_array()) {
      for (const auto &g : desc["g3"]) {
        calc.g3.push_back({g.value("kappa", 1.0)});
      }
    }
    if (desc.contains("g4") && desc["g4"].is_array()) {
      for (const auto &g : desc["g4"]) {
        calc.g4.push_back({g.value("eta", 1.0), g.value("zeta", 1.0),
                           g.value("lambda", 1.0)});
      }
    }
    if (desc.contains("g5") && desc["g5"].is_array()) {
      for (const auto &g : desc["g5"]) {
        calc.g5.push_back({g.value("eta", 1.0), g.value("zeta", 1.0),
                           g.value("lambda", 1.0)});
      }
    }
    if (calc.descriptor_size() == 0) {
      return fail("ml", "acsf descriptor needs at least one of "
                        "g1/g2/g3/g4/g5");
    }
    const std::size_t S = calc.descriptor_size();
    return finish_ml(std::move(calc), j, ntypes, S);
  }

  if (dtype == "soap") {
    SoapModel calc;
    calc.ntypes = ntypes;
    calc.n_max = desc.value("n_max", 6);
    calc.l_max = desc.value("l_max", 6);
    calc.rcut = desc.value("rcut", 6.0);
    calc.sigma = desc.value("sigma", 0.5);
    calc.init_radial_basis();
    const std::size_t S = calc.descriptor_size();
    return finish_ml(std::move(calc), j, ntypes, S);
  }

  if (dtype == "lmbtr") {
    LMBTR calc;
    calc.ntypes = ntypes;
    calc.rcut = desc.value("rcut", 6.0);
    calc.weight_scale = desc.value("weight_scale", calc.rcut / 2.0);
    calc.normalize_l2 = desc.value("normalize", true);
    auto read_grid = [&](const char *key, LMBTR::Grid def) {
      if (desc.contains(key) && desc[key].is_object()) {
        const json &g = desc[key];
        def.min = g.value("min", def.min);
        def.max = g.value("max", def.max);
        def.n = g.value("n", def.n);
        def.sigma = g.value("sigma", def.sigma);
      }
      return def;
    };
    const LMBTR::Grid def_k2{0.0, calc.rcut, 50, 0.3};
    const LMBTR::Grid def_k3{-1.0, 1.0, 50, 0.1};
    if (desc.contains("k2") || desc.contains("k3")) {
      calc.k2 = desc.contains("k2") ? std::optional(read_grid("k2", def_k2))
                                    : std::nullopt;
      calc.k3 = desc.contains("k3") ? std::optional(read_grid("k3", def_k3))
                                    : std::nullopt;
    } else {
      calc.k2 = read_grid("k2", def_k2);
      calc.k3 = read_grid("k3", def_k3);
    }
    if (calc.descriptor_size() == 0) {
      return fail("ml", "lmbtr descriptor needs k2 and/or k3 with n > 0");
    }
    const std::size_t S = calc.descriptor_size();
    return finish_ml(std::move(calc), j, ntypes, S);
  }

  return fail("ml", "unknown descriptor type '" + dtype + "'");
}

using Builder = leaf::result<ForceCalculator> (*)(json &, std::size_t);

template <CModelFamily T> constexpr Builder builder_for() {
  if constexpr (is_ml_family<T>) {
    return &build_ml; // one builder for all three descriptors
  } else if constexpr (std::same_as<T, TersoffForceCalculator>) {
    return &build_tersoff;
  } else if constexpr (std::same_as<T, StiwebForceCalculator>) {
    return &build_stiweb;
  } else {
    return &build_tabulated<T>;
  }
}

leaf::result<Builder> builder_named(std::string_view model) {
  std::optional<Builder> found;
  boost::mp11::mp_for_each<
      boost::mp11::mp_transform<boost::mp11::mp_identity, ModelFamilies>>(
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (!found && model == (is_ml_family<T> ? "ml" : family_name<T>)) {
          found = builder_for<T>();
        }
      });
  if (!found) {
    return fail("unsupported model '" + std::string(model) + "'");
  }
  return *found;
}

} // anonymous namespace

leaf::result<ForceCalculator> parse_force_model(std::string_view input) {
  return catch_json([&]() -> leaf::result<ForceCalculator> {
    json j = json::parse(input);
    if (!j.contains("model")) {
      const std::size_t count =
          (j.contains("potentials") && j["potentials"].is_array())
              ? j["potentials"].size()
              : 0;
      std::size_t ntypes = 1;
      while (ntypes * (ntypes + 1) / 2 < count) {
        ++ntypes;
      }
      if (ntypes * (ntypes + 1) / 2 != count) {
        return fail("bare pair file: " + std::to_string(count) +
                    " potentials is not a valid pair count for any ntypes");
      }
      return build_tabulated<PairForceCalculator>(j, ntypes);
    }

    const std::size_t ntypes = j.value("ntypes", std::size_t{1});
    BOOST_LEAF_AUTO(build, builder_named(j.at("model").get<std::string>()));
    return build(j, ntypes);
  });
}

} // namespace forcesmith::io
