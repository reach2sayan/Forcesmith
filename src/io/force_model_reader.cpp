#include "potfit/io/force_model_reader.hpp"
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

// Parse a sub-object of shape {"format": ..., "potentials": [...]}
// into a flat vector<Potential>. Mirrors the logic in parse_potential.
leaf::result<std::vector<Potential>> parse_pot_list(const json &sub,
                                                    std::string_view label) {
  auto fail = [&](std::string msg) -> leaf::result<std::vector<Potential>> {
    return leaf::new_error(
        ParseError{std::string(label) + ": " + std::move(msg), 0});
  };

  if (!sub.contains("format"))
    return fail("missing 'format' key");
  if (!sub.contains("potentials") || !sub["potentials"].is_array())
    return fail("missing or invalid 'potentials' array");

  // Serialise the sub-object back to string and call the existing parser,
  // which already handles both "tabulated" and "analytic" formats.
  const std::string sub_str = sub.dump();
  auto r = parse_potential(sub_str);
  if (!r)
    return r.error();
  return std::move(*r);
}

// Load paircol potentials from sub into a PotentialPair
// (SymmetricMatrix<Potential>).
leaf::result<void> fill_pair(PotentialPair &mat, const json &sub,
                             std::size_t ntypes, std::string_view label) {
  auto fail = [&](std::string msg) -> leaf::result<void> {
    return leaf::new_error(
        ParseError{std::string(label) + ": " + std::move(msg), 0});
  };

  auto r = parse_pot_list(sub, label);
  if (!r)
    return r.error();
  auto &pots = *r;

  const std::size_t paircol = ntypes * (ntypes + 1) / 2;
  if (pots.size() != paircol)
    return fail("expected " + std::to_string(paircol) +
                " potentials for ntypes=" + std::to_string(ntypes) + ", got " +
                std::to_string(pots.size()));

  mat.reserve(ntypes);
  for (auto &p : pots)
    mat.emplace_back(std::move(p));
  return {};
}

// Load ntypes potentials from sub into a PotentialArray (TypeArray<Potential>).
leaf::result<void> fill_arr(PotentialArray &arr, const json &sub,
                            std::size_t ntypes, std::string_view label) {
  auto fail = [&](std::string msg) -> leaf::result<void> {
    return leaf::new_error(
        ParseError{std::string(label) + ": " + std::move(msg), 0});
  };

  auto r = parse_pot_list(sub, label);
  if (!r)
    return r.error();
  auto &pots = *r;

  if (pots.size() != ntypes)
    return fail("expected " + std::to_string(ntypes) + " potentials, got " +
                std::to_string(pots.size()));

  arr.reserve(ntypes);
  for (auto &p : pots)
    arr.emplace_back(std::move(p));
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
// `regions` pairs each sub-object that owns a `"potentials"` array with its
// region id (0=pair, 1=density, 2=embedding). Mutates `root` in place.
leaf::result<std::vector<GlobalParam>>
build_globals(json &root,
              std::initializer_list<std::pair<json *, int>> regions) {
  auto fail = [](std::string msg) -> leaf::result<std::vector<GlobalParam>> {
    return leaf::new_error(ParseError{"globals: " + std::move(msg), 0});
  };

  std::vector<GlobalParam> globals;
  std::unordered_map<std::string, std::size_t> by_name;
  if (root.contains("globals")) {
    for (const auto &item : root["globals"].items()) {
      const json &spec = item.value();
      if (!spec.contains("value"))
        return fail("global '" + item.key() + "' missing 'value'");
      GlobalParam g;
      g.value = Param{spec.at("value").get<double>(), spec.value("fixed", false)};
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
      if (!pot.is_object() || !pot.contains("type"))
        continue;
      const std::string type = pot["type"].get<std::string>();
      // Collect global-ref parameter keys first, then mutate (don't modify the
      // object mid-iteration).
      std::vector<std::string> ref_keys;
      for (const auto &el : pot.items())
        if (el.value().is_object() && el.value().contains("global"))
          ref_keys.push_back(el.key());
      for (const auto &key : ref_keys) {
        const std::string gname = pot[key]["global"].get<std::string>();
        auto git = by_name.find(gname);
        if (git == by_name.end())
          return fail("reference to undefined global '" + gname + "'");
        auto slot = analytic_param_index(type, key);
        if (!slot)
          return fail("global ref on parameter '" + key +
                      "' not valid for analytic type '" + type + "'");
        globals[git->second].links.push_back({region, i, *slot});
        pot[key] = globals[git->second].value.value; // replace ref → seed number
      }
    }
  }
  return globals;
}

} // anonymous namespace

leaf::result<ForceCalculator> parse_force_model(std::string_view input) {
  auto fail = [](std::string msg) -> leaf::result<ForceCalculator> {
    return leaf::new_error(ParseError{std::move(msg), 0});
  };

  json j;
  try {
    j = json::parse(input);
  } catch (const json::parse_error &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }

  try {
    if (!j.contains("model"))
      return fail("missing 'model' key");
    const std::string model = j["model"].get<std::string>();
    const std::size_t ntypes = j.value("ntypes", std::size_t{1});

    // ── pair ─────────────────────────────────────────────────────────────
    if (model == "pair") {
      // Resolve globals in place (pair potentials live at top-level j), then
      // parse the mutated j (not the raw input string).
      auto rg = build_globals(j, {{&j, 0}});
      if (!rg)
        return rg.error();
      auto r = parse_potential(j.dump());
      if (!r)
        return r.error();
      auto &pots = *r;
      const std::size_t paircol = ntypes * (ntypes + 1) / 2;
      if (pots.size() != paircol)
        return fail("pair: expected " + std::to_string(paircol) +
                    " potentials for ntypes=" + std::to_string(ntypes) +
                    ", got " + std::to_string(pots.size()));
      PairForceCalculator calc;
      calc.ntypes = ntypes;
      calc.pair.reserve(ntypes);
      for (auto &p : pots)
        calc.pair.emplace_back(std::move(p));
      calc.globals = std::move(*rg);
      calc.finalize_globals();
      return ForceCalculator{std::move(calc)};
    }

    // ── eam ──────────────────────────────────────────────────────────────
    if (model == "eam") {
      auto rk = require_keys(j, {"pair", "density", "embedding"}, "eam");
      if (!rk)
        return rk.error();

      // Resolve shared globals BEFORE filling so the potential parser sees plain
      // numbers; mutates j["pair"/"density"/"embedding"] in place.
      auto rg = build_globals(
          j, {{&j["pair"], 0}, {&j["density"], 1}, {&j["embedding"], 2}});
      if (!rg)
        return rg.error();

      EAMForceCalculator calc;
      calc.ntypes = ntypes;

      auto rp = fill_pair(calc.pair, j["pair"], ntypes, "eam.pair");
      if (!rp)
        return rp.error();
      auto rd = fill_arr(calc.density, j["density"], ntypes, "eam.density");
      if (!rd)
        return rd.error();
      auto re =
          fill_arr(calc.embedding, j["embedding"], ntypes, "eam.embedding");
      if (!re)
        return re.error();

      calc.globals = std::move(*rg);
      calc.finalize_globals();

      return ForceCalculator{std::move(calc)};
    }

    // ── adp ──────────────────────────────────────────────────────────────
    if (model == "adp") {
      auto rk = require_keys(
          j, {"pair", "density", "embedding", "dipole", "quadrupole"}, "adp");
      if (!rk)
        return rk.error();

      ADPForceCalculator calc;
      calc.ntypes = ntypes;

      auto rp = fill_pair(calc.pair, j["pair"], ntypes, "adp.pair");
      if (!rp)
        return rp.error();
      auto rd = fill_arr(calc.density, j["density"], ntypes, "adp.density");
      if (!rd)
        return rd.error();
      auto re =
          fill_arr(calc.embedding, j["embedding"], ntypes, "adp.embedding");
      if (!re)
        return re.error();
      auto rdi = fill_pair(calc.dipole, j["dipole"], ntypes, "adp.dipole");
      if (!rdi)
        return rdi.error();
      auto rq =
          fill_pair(calc.quadrupole, j["quadrupole"], ntypes, "adp.quadrupole");
      if (!rq)
        return rq.error();

      return ForceCalculator{std::move(calc)};
    }

    // ── angular ───────────────────────────────────────────────────────────
    if (model == "angular") {
      auto rk = require_keys(j, {"pair", "radial", "angular"}, "angular");
      if (!rk)
        return rk.error();

      AngularForceCalculator calc;
      calc.ntypes = ntypes;

      auto rp = fill_pair(calc.pair, j["pair"], ntypes, "angular.pair");
      if (!rp)
        return rp.error();
      auto rr = fill_pair(calc.radial, j["radial"], ntypes, "angular.radial");
      if (!rr)
        return rr.error();
      // angular g is indexed by the central atom type → ntypes entries.
      auto ra = fill_arr(calc.angular, j["angular"], ntypes, "angular.angular");
      if (!ra)
        return ra.error();

      return ForceCalculator{std::move(calc)};
    }

    // ── tersoff ───────────────────────────────────────────────────────────
    if (model == "tersoff") {
      if (!j.contains("potentials") || !j["potentials"].is_array())
        return fail("tersoff: missing 'potentials' array");

      const auto &pots_arr = j["potentials"];
      const std::size_t paircol = ntypes * (ntypes + 1) / 2;
      if (pots_arr.size() != paircol)
        return fail("tersoff: expected " + std::to_string(paircol) +
                    " entries for ntypes=" + std::to_string(ntypes));

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

    // ── stiweb ────────────────────────────────────────────────────────────
    if (model == "stiweb") {
      if (!j.contains("potentials") || !j["potentials"].is_array())
        return fail("stiweb: missing 'potentials' array");

      const auto &pots_arr = j["potentials"];
      const std::size_t paircol = ntypes * (ntypes + 1) / 2;
      if (pots_arr.size() != paircol)
        return fail("stiweb: expected " + std::to_string(paircol) +
                    " entries for ntypes=" + std::to_string(ntypes));

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
      if (!j.contains("lambda") || !j["lambda"].is_array())
        return fail("stiweb: missing 'lambda' array (ntypes·paircol entries)");
      const auto &lam_arr = j["lambda"];
      const std::size_t lam_count = ntypes * paircol;
      if (lam_arr.size() != lam_count)
        return fail("stiweb: expected " + std::to_string(lam_count) +
                    " lambda entries for ntypes=" + std::to_string(ntypes) +
                    ", got " + std::to_string(lam_arr.size()));
      calc.lambda.reserve(lam_count);
      for (const auto &l : lam_arr)
        calc.lambda.emplace_back(Param{l.get<double>()});

      return ForceCalculator{std::move(calc)};
    }

    return fail("unsupported model '" + model + "'");

  } catch (const json::exception &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }
}

} // namespace potfit::io
