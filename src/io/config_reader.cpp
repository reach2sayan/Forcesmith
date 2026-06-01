#include "potfit/io/config_reader.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>

namespace potfit::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

namespace {

[[nodiscard]] leaf::error_id err(std::string msg) {
  return leaf::new_error(ParseError{std::move(msg), 0});
}

// Required key present → sub-json by pointer (avoids result<T&>).
[[nodiscard]] leaf::result<const json *>
require_key(const json &obj, const char *key, std::string_view ctx) {
  if (!obj.contains(key))
    return err(std::string(ctx) + ": missing key '" + key + "'");
  return &obj[key];
}

// Array of exactly N doubles.
template <std::size_t N>
[[nodiscard]] leaf::result<std::array<double, N>>
get_array_n(const json &arr, std::string_view ctx) {
  if (!arr.is_array() || arr.size() != N)
    return err(std::string(ctx) + ": must be an array of " + std::to_string(N) +
               " doubles");
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i)
    out[i] = arr[i].get<double>();
  return out;
}

[[nodiscard]] leaf::result<Vec3> get_vec3(const json &arr,
                                          std::string_view ctx) {
  BOOST_LEAF_AUTO(a, get_array_n<3>(arr, ctx));
  return Vec3{a[0], a[1], a[2]};
}

// Require key + parse as Vec3 in one shot.
[[nodiscard]] leaf::result<Vec3> require_vec3(const json &obj, const char *key,
                                              std::string_view ctx) {
  BOOST_LEAF_AUTO(sub, require_key(obj, key, ctx));
  return get_vec3(*sub, std::string(ctx) + " '" + key + "'");
}

[[nodiscard]] leaf::result<Mat3> parse_box(const json &obj,
                                           std::string_view ctx) {
  BOOST_LEAF_AUTO(x, require_vec3(obj, "X", ctx));
  BOOST_LEAF_AUTO(y, require_vec3(obj, "Y", ctx));
  BOOST_LEAF_AUTO(z, require_vec3(obj, "Z", ctx));
  Mat3 box;
  box.col(0) = x;
  box.col(1) = y;
  box.col(2) = z;
  return box;
}

[[nodiscard]] leaf::result<double> parse_energy(const json &obj,
                                                std::string_view ctx) {
  BOOST_LEAF_AUTO(e, require_key(obj, "E", ctx));
  return (*e).get<double>();
}

// Optional "S" = [xx, yy, zz, xy, yz, zx]; absent → Zero().
[[nodiscard]] leaf::result<SymTens> parse_stress(const json &obj,
                                                 std::string_view ctx) {
  SymTens stress = SymTens::Zero();
  if (!obj.contains("S"))
    return stress;
  BOOST_LEAF_AUTO(s, (get_array_n<6>(obj["S"], std::string(ctx) + " 'S'")));
  stress(0, 0) = s[0];
  stress(1, 1) = s[1];
  stress(2, 2) = s[2];
  stress(0, 1) = stress(1, 0) = s[3];
  stress(1, 2) = stress(2, 1) = s[4];
  stress(0, 2) = stress(2, 0) = s[5];
  return stress;
}

[[nodiscard]] leaf::result<Atom> parse_atom(const json &a_obj,
                                            const SpeciesRegistry &registry,
                                            std::string_view ctx) {
  BOOST_LEAF_AUTO(elem, require_key(a_obj, "element", ctx));
  BOOST_LEAF_AUTO(pos, require_vec3(a_obj, "position", ctx));

  Atom a;
  BOOST_LEAF_AUTO(sp, species_of(registry, (*elem).get<std::string>()));
  a.type = sp;
  a.pos = pos;

  if (a_obj.contains("force")) {
    BOOST_LEAF_AUTO(f, get_vec3(a_obj["force"], std::string(ctx) + " 'force'"));
    a.ref.force = f;
  }
  return a;
}

// Registry-free atom parse: resolve the element straight from the static
// periodic-table catalog (Species::lookup). The compact table slot
// (Species::index) is left at its default 0 — PotFit assigns the real slot
// at freeze once the full element set is known. Used by Configuration::from_*.
[[nodiscard]] leaf::result<Atom> parse_atom_catalog(const json &a_obj,
                                                    std::string_view ctx) {
  BOOST_LEAF_AUTO(elem, require_key(a_obj, "element", ctx));
  BOOST_LEAF_AUTO(pos, require_vec3(a_obj, "position", ctx));

  Atom a;
  BOOST_LEAF_AUTO(sp, Species::lookup((*elem).get<std::string>()));
  a.type = sp;
  a.pos = pos;

  if (a_obj.contains("force")) {
    BOOST_LEAF_AUTO(f, get_vec3(a_obj["force"], std::string(ctx) + " 'force'"));
    a.ref.force = f;
  }
  return a;
}

// Parse ONE configuration record using the catalog (no registry). Shared by
// Configuration::from_text and (via re-serialised records) the streaming
// loader in src/io/loaders.cpp.
[[nodiscard]] leaf::result<Configuration> config_from_json(const json &obj) {
  if (!obj.is_object())
    return err("configuration must be a JSON object");

  Configuration cfg;
  cfg.name = obj.value("name", "");
  BOOST_LEAF_AUTO(box, parse_box(obj, "configuration"));
  cfg.bc = PeriodicBC(box);

  BOOST_LEAF_AUTO(e, parse_energy(obj, "configuration"));
  cfg.ref.energy = e;

  cfg.weight = obj.value("W", 1.0);

  BOOST_LEAF_AUTO(s, parse_stress(obj, "configuration"));
  cfg.ref.stress = s;

  BOOST_LEAF_AUTO(atoms, require_key(obj, "atoms", "configuration"));
  if (!(*atoms).is_array())
    return err("configuration: 'atoms' must be an array");

  cfg.atoms.reserve((*atoms).size());
  std::size_t ai = 0;
  for (const auto &a_obj : *atoms) {
    BOOST_LEAF_AUTO(atom, parse_atom_catalog(
                              a_obj, "atom[" + std::to_string(ai++) + "]"));
    cfg.atoms.push_back(std::move(atom));
  }
  return cfg;
}

[[nodiscard]] leaf::result<Configuration>
parse_configuration(const json &obj, const SpeciesRegistry &registry,
                    std::string_view ctx) {
  if (!obj.is_object())
    return err(std::string(ctx) + ": each configuration must be a JSON object");

  Configuration cfg;
  cfg.name = obj.value("name", "");

  BOOST_LEAF_AUTO(box, parse_box(obj, ctx));
  cfg.bc = PeriodicBC(box);

  BOOST_LEAF_AUTO(e, parse_energy(obj, ctx));
  cfg.ref.energy = e;

  cfg.weight = obj.value("W", 1.0);

  BOOST_LEAF_AUTO(s, parse_stress(obj, ctx));
  cfg.ref.stress = s;

  BOOST_LEAF_AUTO(atoms, require_key(obj, "atoms", ctx));
  if (!(*atoms).is_array())
    return err(std::string(ctx) + ": 'atoms' must be an array");

  cfg.atoms.reserve((*atoms).size());
  std::size_t ai = 0;
  for (const auto &a_obj : *atoms) {
    BOOST_LEAF_AUTO(atom, parse_atom(a_obj, registry,
                                     std::string(ctx) + " atom[" +
                                         std::to_string(ai++) + "]"));
    cfg.atoms.push_back(std::move(atom));
  }
  return cfg;
}

// First pass: gather every distinct element symbol that appears in any atom, so
// the Z-sorted registry is fixed before atoms are stamped. Lenient about
// structure — full validation happens in the second pass.
[[nodiscard]] std::vector<std::string> collect_symbols(const json &j) {
  std::vector<std::string> syms;
  if (!j.is_array())
    return syms;
  for (const auto &obj : j) {
    if (!obj.is_object() || !obj.contains("atoms") || !obj["atoms"].is_array())
      continue;
    for (const auto &a_obj : obj["atoms"]) {
      if (a_obj.is_object() && a_obj.contains("element") &&
          a_obj["element"].is_string()) {
        std::string s = a_obj["element"].get<std::string>();
        if (std::ranges::find(syms, s) == syms.end())
          syms.push_back(std::move(s));
      }
    }
  }
  return syms;
}

} // namespace

leaf::result<ParsedConfig> parse_config(std::string_view input) {
  json j;
  try {
    j = json::parse(input);
  } catch (const json::parse_error &e) {
    return err(e.what());
  }

  if (!j.is_array())
    return err("top-level JSON must be an array of configurations");

  // Pass 1 — gather symbols and build the Z-sorted element↔slot registry.
  const std::vector<std::string> sym_strings = collect_symbols(j);
  std::vector<std::string_view> sym_views(sym_strings.begin(),
                                          sym_strings.end());
  BOOST_LEAF_AUTO(registry, build_species_registry(sym_views));

  // Pass 2 — parse each configuration, stamping atoms through the registry.
  std::vector<Configuration> configs;
  configs.reserve(j.size());
  try {
    std::size_t ci = 0;
    for (const auto &obj : j) {
      BOOST_LEAF_AUTO(
          cfg, parse_configuration(obj, registry,
                                   "config[" + std::to_string(ci++) + "]"));
      configs.push_back(std::move(cfg));
    }
  } catch (const json::exception &e) {
    return err(e.what());
  }

  return ParsedConfig{std::move(configs), std::move(registry)};
}

} // namespace potfit::io

namespace potfit {

boost::leaf::result<Configuration>
Configuration::from_text(std::string_view text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::parse_error &e) {
    return boost::leaf::new_error(io::ParseError{e.what(), 0});
  }
  try {
    return io::config_from_json(j);
  } catch (const nlohmann::json::exception &e) {
    return boost::leaf::new_error(io::ParseError{e.what(), 0});
  }
}

boost::leaf::result<Configuration>
Configuration::from_file(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f)
    return boost::leaf::new_error(
        io::ParseError{"cannot open config file: " + path.string(), 0});
  std::string text((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  return Configuration::from_text(text);
}

} // namespace potfit
