#include "forcesmith/io/force_model_reader.hpp"
#include "forcesmith/io/factory.hpp"
#include "forcesmith/io/json_util.hpp"
#include "forcesmith/io/potential_reader.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace forcesmith::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

namespace {

using Fail = leaf::result<ForceCalculator>;

// Build a ParseError leaf error. The returned error_id converts implicitly to
// leaf::result<T> for any T, so callers `return fail(...);` regardless of their
// result type. The two-arg form prefixes the message with `label`.
leaf::error_id fail(std::string msg) {
  return leaf::new_error(ParseError{std::move(msg), 0});
}
leaf::error_id fail(std::string_view label, std::string msg) {
  return fail(std::string(label) + ": " + std::move(msg));
}

// Parse a sub-object of shape {"format": ..., "potentials": [...]}
// into a flat vector<Potential>. Mirrors the logic in parse_potential.
leaf::result<std::vector<Potential>> parse_pot_list(const json &sub,
                                                    std::string_view label) {
  if (!sub.contains("format")) {
    return fail(label, "missing 'format' key");
  }
  if (!sub.contains("potentials") || !sub["potentials"].is_array()) {
    return fail(label, "missing or invalid 'potentials' array");
  }

  const std::string sub_str = sub.dump();
  auto r = parse_potential(sub_str);
  if (!r) {
    return r.error();
  }
  return std::move(*r);
}

// Load paircol potentials from sub into a PotentialPair
// (SymmetricMatrix<Potential>).
leaf::result<void> fill_pair(PotentialPair &mat, const json &sub,
                             std::size_t ntypes, std::string_view label) {
  auto r = parse_pot_list(sub, label);
  if (!r) {
    return r.error();
  }
  auto &pots = *r;

  const std::size_t paircol = ntypes * (ntypes + 1) / 2;
  if (pots.size() != paircol) {
    return fail(label, "expected " + std::to_string(paircol) +
                           " potentials for ntypes=" + std::to_string(ntypes) +
                           ", got " + std::to_string(pots.size()));
  }

  mat.reserve(ntypes);
  std::ranges::move(pots, std::back_inserter(mat));
  return {};
}

// Load ntypes potentials from sub into a PotentialArray (TypeArray<Potential>).
leaf::result<void> fill_arr(PotentialArray &arr, const json &sub,
                            std::size_t ntypes, std::string_view label) {
  auto r = parse_pot_list(sub, label);
  if (!r) {
    return r.error();
  }

  auto &pots = *r;
  if (pots.size() != ntypes) {
    return fail(label, "expected " + std::to_string(ntypes) +
                           " potentials, got " + std::to_string(pots.size()));
  }

  arr.reserve(ntypes);
  std::ranges::move(pots, std::back_inserter(arr));
  return {};
}

// Check that all required keys are present.
leaf::result<void> require_keys(const json &obj,
                                std::initializer_list<const char *> keys,
                                std::string_view ctx) {
  for (const char *k : keys) {
    if (!obj.contains(k))
      return leaf::new_error(
          ParseError{std::string(ctx) + ": missing section '" + k + "'", 0});
  }
  return {};
}

// Resolve shared global parameters. Reads the optional top-level
// `"globals": { name: {"value":, "min":, "max":, "fixed":} }`, then scans each
// region's `potentials` for a parameter expressed as `{"global": name}`. For
// every such reference it records a Link (region, flat index, param slot) and
// REPLACES the reference in `root` with the global's seed value, so the normal
// number-based potential parser sees a plain number. The caller assigns the
// returned vector to calc.globals and calls finalize_globals().
//
// `regions` pairs each sub-object that owns a `"potentials"` array with the
// calculator sub-table (LinkRegion) it belongs to. Mutates `root` in place.
leaf::result<std::vector<GlobalParam>> build_globals(
    json &root,
    std::initializer_list<std::pair<json *, GlobalParam::Link::LinkRegion>>
        regions) {
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
      // Collect global-ref parameter keys first, then mutate (don't modify the
      // object mid-iteration).
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

leaf::result<ForceCalculator> build_pair(json &j, std::size_t ntypes) {
  // Resolve globals in place (pair potentials live at top-level j), then
  // parse the mutated j (not the raw input string).
  auto rg = build_globals(j, {{&j, GlobalParam::Link::LinkRegion::PAIR}});
  if (!rg) {
    return rg.error();
  }
  auto r = parse_potential(j.dump());
  if (!r) {
    return r.error();
  }

  auto &pots = *r;
  const std::size_t paircol = ntypes * (ntypes + 1) / 2;
  if (pots.size() != paircol) {
    return fail("pair", "expected " + std::to_string(paircol) +
                            " potentials for ntypes=" + std::to_string(ntypes) +
                            ", got " + std::to_string(pots.size()));
  }

  PairForceCalculator calc;
  calc.ntypes = ntypes;
  calc.pair.reserve(ntypes);
  std::ranges::move(pots, std::back_inserter(calc.pair));
  calc.globals = std::move(*rg);
  calc.finalize_globals();
  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_eam(json &j, std::size_t ntypes) {
  auto rk = require_keys(j, {"pair", "density", "embedding"}, "eam");
  if (!rk) {
    return rk.error();
  }

  // Resolve shared globals BEFORE filling so the potential parser sees plain
  // numbers; mutates j["pair"/"density"/"embedding"] in place.
  using enum GlobalParam::Link::LinkRegion;
  auto rg = build_globals(j, {{&j["pair"], PAIR},
                              {&j["density"], DENSITY},
                              {&j["embedding"], EMBEDDING}});
  if (!rg) {
    return rg.error();
  }

  EAMForceCalculator calc;
  calc.ntypes = ntypes;

  auto rp = fill_pair(calc.pair, j["pair"], ntypes, "eam.pair");
  if (!rp) {
    return rp.error();
  }
  auto rd = fill_arr(calc.density, j["density"], ntypes, "eam.density");
  if (!rd) {
    return rd.error();
  }
  auto re = fill_arr(calc.embedding, j["embedding"], ntypes, "eam.embedding");
  if (!re)
    return re.error();

  calc.globals = std::move(*rg);
  calc.finalize_globals();

  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_adp(json &j, std::size_t ntypes) {
  auto rk = require_keys(
      j, {"pair", "density", "embedding", "dipole", "quadrupole"}, "adp");
  if (!rk) {
    return rk.error();
  }

  ADPForceCalculator calc;
  calc.ntypes = ntypes;

  auto rp = fill_pair(calc.pair, j["pair"], ntypes, "adp.pair");
  if (!rp) {
    return rp.error();
  }

  auto rd = fill_arr(calc.density, j["density"], ntypes, "adp.density");
  if (!rd) {
    return rd.error();
  }

  auto re = fill_arr(calc.embedding, j["embedding"], ntypes, "adp.embedding");
  if (!re) {
    return re.error();
  }

  auto rdi = fill_pair(calc.dipole, j["dipole"], ntypes, "adp.dipole");
  if (!rdi) {
    return rdi.error();
  }

  auto rq =
      fill_pair(calc.quadrupole, j["quadrupole"], ntypes, "adp.quadrupole");
  if (!rq) {
    return rq.error();
  }

  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_angular(json &j, std::size_t ntypes) {
  auto rk = require_keys(j, {"pair", "radial", "angular"}, "angular");
  if (!rk) {
    return rk.error();
  }

  AngularForceCalculator calc;
  calc.ntypes = ntypes;

  auto rp = fill_pair(calc.pair, j["pair"], ntypes, "angular.pair");
  if (!rp) {
    return rp.error();
  }
  auto rr = fill_pair(calc.radial, j["radial"], ntypes, "angular.radial");
  if (!rr) {
    return rr.error();
  }
  // angular g is indexed by the central atom type → ntypes entries.
  auto ra = fill_arr(calc.angular, j["angular"], ntypes, "angular.angular");
  if (!ra) {
    return ra.error();
  }

  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_tersoff(json &j, std::size_t ntypes) {
  if (!j.contains("potentials") || !j["potentials"].is_array()) {
    return fail("tersoff", "missing 'potentials' array");
  }

  const auto &pots_arr = j["potentials"];
  const std::size_t paircol = ntypes * (ntypes + 1) / 2;
  if (pots_arr.size() != paircol) {
    return fail("tersoff", "expected " + std::to_string(paircol) +
                               " entries for ntypes=" + std::to_string(ntypes));
  }

  TersoffForceCalculator calc;
  calc.ntypes = ntypes;
  calc.params.reserve(ntypes);

  for (const auto &p : pots_arr) {
    TersoffParams tp;
    tp.A = p.at("A").get<double>();
    tp.B = p.at("B").get<double>();
    tp.lambda = p.at("lambda").get<double>();
    tp.mu = p.at("mu").get<double>();
    tp.beta = p.at("beta").get<double>();
    tp.n = p.at("n").get<double>();
    tp.c = p.at("c").get<double>();
    tp.d = p.at("d").get<double>();
    tp.h = p.at("h").get<double>();
    tp.R = p.at("R").get<double>();
    tp.S = p.at("S").get<double>();
    // Optional bond-order mixing weight (forcesmith's omega). Absent → 1.0,
    // fixed (diagonal/same-type pairs); present → free for fitting.
    if (p.contains("omega"))
      tp.omega = Param{p.at("omega").get<double>(), false};
    calc.params.emplace_back(tp);
  }

  return ForceCalculator{std::move(calc)};
}

leaf::result<ForceCalculator> build_stiweb(json &j, std::size_t ntypes) {
  if (!j.contains("potentials") || !j["potentials"].is_array()) {
    return fail("stiweb", "missing 'potentials' array");
  }

  const auto &pots_arr = j["potentials"];
  const std::size_t paircol = ntypes * (ntypes + 1) / 2;
  if (pots_arr.size() != paircol) {
    return fail("stiweb", "expected " + std::to_string(paircol) +
                              " entries for ntypes=" + std::to_string(ntypes));
  }

  StiwebForceCalculator calc;
  calc.ntypes = ntypes;
  calc.params.reserve(ntypes);

  for (const auto &p : pots_arr) {
    SWParams sp;
    sp.A = p.at("A").get<double>();
    sp.B = p.at("B").get<double>();
    sp.p = p.at("p").get<double>();
    sp.q = p.at("q").get<double>();
    sp.delta = p.at("delta").get<double>();
    sp.a1 = p.at("a1").get<double>();
    sp.gamma = p.at("gamma").get<double>();
    sp.a2 = p.at("a2").get<double>();
    calc.params.emplace_back(sp);
  }

  // Per-triplet 3-body strength λ[i][j][k] (symmetric in j,k): a flat array
  // of ntypes·paircol entries in canonical order (i; pair_slot(j,k)).
  if (!j.contains("lambda") || !j["lambda"].is_array()) {
    return fail("stiweb", "missing 'lambda' array (ntypes·paircol entries)");
  }
  const auto &lam_arr = j["lambda"];
  const std::size_t lam_count = ntypes * paircol;
  if (lam_arr.size() != lam_count) {
    return fail("stiweb",
                "expected " + std::to_string(lam_count) +
                    " lambda entries for ntypes=" + std::to_string(ntypes) +
                    ", got " + std::to_string(lam_arr.size()));
  }
  calc.lambda.reserve(lam_count);
  std::ranges::transform(
      lam_arr, std::back_inserter(calc.lambda),
      [](const auto &l) { return Param{l.template get<double>()}; });

  return ForceCalculator{std::move(calc)};
}

// Build one EnergyHead from a JSON head spec, validated against descriptor size
// S. Supports "linear" (coeffs + bias).
leaf::result<EnergyHead> parse_head(const json &h, std::size_t S) {
  const std::string htype = h.value("type", std::string("linear"));
  if (htype == "linear") {
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
  return fail("ml", "unknown head type '" + htype + "'");
}

// Attach the per-type heads (positional, one per element slot) to an MLBase
// model and wrap it as a ForceCalculator. S is the model's descriptor size.
template <class Model>
leaf::result<ForceCalculator> finish_ml(Model calc, json &j,
                                        std::size_t ntypes, std::size_t S) {
  if (!j.contains("heads") || !j["heads"].is_array()) {
    return fail("ml", "missing 'heads' array");
  }
  const auto &heads = j["heads"];
  if (heads.size() != ntypes) {
    return fail("ml", "expected " + std::to_string(ntypes) + " heads, got " +
                          std::to_string(heads.size()));
  }
  calc.heads.reserve(ntypes);
  for (const auto &h : heads) {
    auto rh = parse_head(h, S);
    if (!rh) {
      return rh.error();
    }
    calc.heads.emplace_back(std::move(*rh));
  }
  // Optional per-feature standardization (one {mean, inv_std} per type), written
  // by output_writer's add_standardization. Absent → identity transform (older
  // startpot files still load). Each vector is empty (type had no atoms) or S long.
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

  // "symmetry_functions" is the legacy alias for "acsf".
  if (dtype == "symmetry_functions" || dtype == "acsf") {
    ACSF calc;
    calc.ntypes = ntypes;
    calc.rcut = desc.value("rcut", 6.0);
    // G1: accept either a count ("g1": 1) or an array of empty objects.
    if (desc.contains("g1")) {
      const json &g1 = desc["g1"];
      calc.g1 = g1.is_number() ? g1.get<std::size_t>()
                : g1.is_array() ? g1.size()
                                : std::size_t{0};
    }
    if (desc.contains("g2") && desc["g2"].is_array()) {
      for (const auto &g : desc["g2"]) {
        calc.radial.push_back(
            {g.at("eta").get<double>(), g.value("rs", 0.0)});
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
    // Read the descriptor size BEFORE the move — argument evaluation order is
    // unspecified, so `calc.descriptor_size()` in the call args can run after
    // calc is moved-from.
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
    // A k-term is active when its JSON object is present (default both on if
    // neither is given), so an empty descriptor never slips through.
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

// Error policy for the force-model factory: an unknown "model" string maps to
// forcesmith's existing "unsupported model" message. (The generic ForcesmithFactory
// lives in forcesmith/io/factory.hpp.)
template <class IdentifierType, class AbstractProduct>
struct UnsupportedModelError {
  static leaf::result<AbstractProduct> OnUnknownType(const IdentifierType &id) {
    return leaf::new_error(
        ParseError{"unsupported model '" + std::string(id) + "'", 0});
  }
};

// The concrete force-model factory: "model" string → per-model builder.
using ModelFactory =
    ForcesmithFactory<ForceCalculator, std::string,
                  leaf::result<ForceCalculator> (*)(json &, std::size_t),
                  UnsupportedModelError>;

const ModelFactory &model_factory() {
  static const ModelFactory factory = [] {
    ModelFactory f;
    f.Register("pair", &build_pair);
    f.Register("eam", &build_eam);
    f.Register("adp", &build_adp);
    f.Register("angular", &build_angular);
    f.Register("tersoff", &build_tersoff);
    f.Register("stiweb", &build_stiweb);
    f.Register("ml", &build_ml);
    return f;
  }();
  return factory;
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
      return build_pair(j, ntypes);
    }

    const std::string model = j["model"].get<std::string>();
    const std::size_t ntypes = j.value("ntypes", std::size_t{1});
    return model_factory().CreateObject(model, j, ntypes);
  });
}

} // namespace forcesmith::io
