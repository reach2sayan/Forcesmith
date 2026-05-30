#include "potfit/io/force_model_reader.hpp"
#include "potfit/io/potential_reader.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <string>

namespace potfit::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

namespace {

using Fail = leaf::result<ForceModel>;

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
leaf::result<void> fill_pair(PotentialPair &mat, const json &sub, int ntypes,
                             std::string_view label) {
  auto fail = [&](std::string msg) -> leaf::result<void> {
    return leaf::new_error(
        ParseError{std::string(label) + ": " + std::move(msg), 0});
  };

  auto r = parse_pot_list(sub, label);
  if (!r)
    return r.error();
  auto &pots = *r;

  const int paircol = ntypes * (ntypes + 1) / 2;
  if (static_cast<int>(pots.size()) != paircol)
    return fail("expected " + std::to_string(paircol) +
                " potentials for ntypes=" + std::to_string(ntypes) + ", got " +
                std::to_string(pots.size()));

  mat.reserve(ntypes);
  for (auto &p : pots)
    mat.emplace_back(std::move(p));
  return {};
}

// Load ntypes potentials from sub into a PotentialArray (TypeArray<Potential>).
leaf::result<void> fill_arr(PotentialArray &arr, const json &sub, int ntypes,
                            std::string_view label) {
  auto fail = [&](std::string msg) -> leaf::result<void> {
    return leaf::new_error(
        ParseError{std::string(label) + ": " + std::move(msg), 0});
  };

  auto r = parse_pot_list(sub, label);
  if (!r)
    return r.error();
  auto &pots = *r;

  if (static_cast<int>(pots.size()) != ntypes)
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

} // anonymous namespace

leaf::result<ForceModel> parse_force_model(std::string_view input) {
  auto fail = [](std::string msg) -> leaf::result<ForceModel> {
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
    const int ntypes = j.value("ntypes", 1);

    // ── pair ─────────────────────────────────────────────────────────────
    if (model == "pair") {
      auto r = parse_potential(input);
      if (!r)
        return r.error();
      return ForceModel{std::move(*r)};
    }

    // ── eam ──────────────────────────────────────────────────────────────
    if (model == "eam") {
      auto rk = require_keys(j, {"pair", "density", "embedding"}, "eam");
      if (!rk)
        return rk.error();

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

      return ForceModel{std::move(calc)};
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

      return ForceModel{std::move(calc)};
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
      auto ra =
          fill_pair(calc.angular, j["angular"], ntypes, "angular.angular");
      if (!ra)
        return ra.error();

      return ForceModel{std::move(calc)};
    }

    // ── tersoff ───────────────────────────────────────────────────────────
    if (model == "tersoff") {
      if (!j.contains("potentials") || !j["potentials"].is_array())
        return fail("tersoff: missing 'potentials' array");

      const auto &pots_arr = j["potentials"];
      const int paircol = ntypes * (ntypes + 1) / 2;
      if (static_cast<int>(pots_arr.size()) != paircol)
        return fail("tersoff: expected " + std::to_string(paircol) +
                    " entries for ntypes=" + std::to_string(ntypes));

      TersoffForceCalculator calc;
      calc.ntypes = ntypes;
      calc.params.reserve(ntypes);

      for (const auto &p : pots_arr) {
        TersoffParams tp;
        tp.A      = p.at("A").get<double>();
        tp.B      = p.at("B").get<double>();
        tp.lambda = p.at("lambda").get<double>();
        tp.mu     = p.at("mu").get<double>();
        tp.beta   = p.at("beta").get<double>();
        tp.n      = p.at("n").get<double>();
        tp.c      = p.at("c").get<double>();
        tp.d      = p.at("d").get<double>();
        tp.h      = p.at("h").get<double>();
        tp.R      = p.at("R").get<double>();
        tp.S      = p.at("S").get<double>();
        calc.params.emplace_back(tp);
      }

      return ForceModel{std::move(calc)};
    }

    // ── stiweb ────────────────────────────────────────────────────────────
    if (model == "stiweb") {
      if (!j.contains("potentials") || !j["potentials"].is_array())
        return fail("stiweb: missing 'potentials' array");

      const auto &pots_arr = j["potentials"];
      const int paircol = ntypes * (ntypes + 1) / 2;
      if (static_cast<int>(pots_arr.size()) != paircol)
        return fail("stiweb: expected " + std::to_string(paircol) +
                    " entries for ntypes=" + std::to_string(ntypes));

      StiwebForceCalculator calc;
      calc.ntypes = ntypes;
      calc.params.reserve(ntypes);

      for (const auto &p : pots_arr) {
        SWParams sp;
        sp.A      = p.at("A").get<double>();
        sp.B      = p.at("B").get<double>();
        sp.p      = p.at("p").get<double>();
        sp.q      = p.at("q").get<double>();
        sp.a      = p.at("a").get<double>();
        sp.sigma  = p.at("sigma").get<double>();
        sp.lambda = p.at("lambda").get<double>();
        sp.gamma  = p.at("gamma").get<double>();
        calc.params.emplace_back(sp);
      }

      return ForceModel{std::move(calc)};
    }

    return fail("unsupported model '" + model + "'");

  } catch (const json::exception &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }
}

} // namespace potfit::io
