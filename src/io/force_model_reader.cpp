#include "potfit/io/force_model_reader.hpp"
#include "potfit/io/factory.hpp"
#include "potfit/io/json_util.hpp"
#include "potfit/io/potential_reader.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace potfit::io {

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
    // Optional bond-order mixing weight (potfit's omega). Absent → 1.0,
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

// Error policy for the force-model factory: an unknown "model" string maps to
// potfit's existing "unsupported model" message. (The generic PotfitFactory
// lives in potfit/io/factory.hpp.)
template <class IdentifierType, class AbstractProduct>
struct UnsupportedModelError {
  static leaf::result<AbstractProduct> OnUnknownType(const IdentifierType &id) {
    return leaf::new_error(
        ParseError{"unsupported model '" + std::string(id) + "'", 0});
  }
};

// The concrete force-model factory: "model" string → per-model builder.
using ModelFactory =
    PotfitFactory<ForceCalculator, std::string,
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
    return f;
  }();
  return factory;
}

} // anonymous namespace

leaf::result<ForceCalculator> parse_force_model(std::string_view input) {
  return catch_json([&]() -> leaf::result<ForceCalculator> {
    json j = json::parse(input);
    if (!j.contains("model")) {
      // Legacy bare pair section ({format, potentials}) with no model wrapper —
      // as emitted by io::write_native. Treat it as a pair model, inferring
      // ntypes from the potential count (paircol = ntypes*(ntypes+1)/2). This
      // keeps write_native ↔ parse_force_model symmetric (checkpoints and pair
      // endpots round-trip).
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

} // namespace potfit::io
