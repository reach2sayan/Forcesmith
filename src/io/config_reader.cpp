#include "potfit/io/config_reader.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>

namespace potfit::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

leaf::result<std::vector<Configuration>> parse_config(std::string_view input) {
  auto fail = [](std::string msg) -> leaf::result<std::vector<Configuration>> {
    return leaf::new_error(ParseError{std::move(msg), 0});
  };

  json j;
  try {
    j = json::parse(input);
  } catch (const json::parse_error &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }

  if (!j.is_array())
    return fail("top-level JSON must be an array of configurations");

  // element symbol → Atom.type index; built dynamically, first seen = 0
  std::vector<std::string> element_map;
  auto element_index = [&](const std::string &sym) -> int {
    if (auto it = std::ranges::find(element_map, sym); it != element_map.end())
      return static_cast<int>(it - element_map.begin());
    element_map.push_back(sym);
    return static_cast<int>(element_map.size()) - 1;
  };

  std::vector<Configuration> configs;
  configs.reserve(j.size());

  try {
    for (const auto &obj : j) {
      if (!obj.is_object())
        return fail("each configuration must be a JSON object");

      Configuration cfg;

      // Box vectors
      for (const char *key : {"X", "Y", "Z"}) {
        if (!obj.contains(key))
          return fail(std::string("configuration missing box vector '") + key +
                      "'");
        if (!obj[key].is_array() || obj[key].size() != 3)
          return fail(std::string("box vector '") + key +
                      "' must be array of 3 doubles");
      }
      Mat3 box;
      box.col(0) = Vec3{obj["X"][0].get<double>(), obj["X"][1].get<double>(),
                        obj["X"][2].get<double>()};
      box.col(1) = Vec3{obj["Y"][0].get<double>(), obj["Y"][1].get<double>(),
                        obj["Y"][2].get<double>()};
      box.col(2) = Vec3{obj["Z"][0].get<double>(), obj["Z"][1].get<double>(),
                        obj["Z"][2].get<double>()};
      cfg.bc = PeriodicBC(box);

      // Energy (required)
      if (!obj.contains("E"))
        return fail("configuration missing energy key 'E'");
      cfg.energy = obj["E"].get<double>();

      // Weight (optional, default 1.0)
      cfg.weight = obj.value("W", 1.0);

      // Stress (optional)
      if (obj.contains("S")) {
        const auto &s = obj["S"];
        if (!s.is_array() || s.size() != 6)
          return fail(
              "'S' must be an array of 6 doubles [xx, yy, zz, xy, yz, zx]");

        cfg.stress(0, 0) = s[0].get<double>();
        cfg.stress(1, 1) = s[1].get<double>();
        cfg.stress(2, 2) = s[2].get<double>();
        cfg.stress(0, 1) = cfg.stress(1, 0) = s[3].get<double>();
        cfg.stress(1, 2) = cfg.stress(2, 1) = s[4].get<double>();
        cfg.stress(0, 2) = cfg.stress(2, 0) = s[5].get<double>();
      }

      // Atoms
      if (!obj.contains("atoms"))
        return fail("configuration missing 'atoms' array");
      const auto &atoms_arr = obj["atoms"];
      if (!atoms_arr.is_array())
        return fail("'atoms' must be an array");

      for (const auto &a_obj : atoms_arr) {
        if (!a_obj.contains("element"))
          return fail("atom missing 'element' key");
        if (!a_obj.contains("position"))
          return fail("atom missing 'position' key");

        const auto &pos_arr = a_obj["position"];
        if (!pos_arr.is_array() || pos_arr.size() != 3)
          return fail("atom 'position' must be an array of 3 doubles");

        Atom a;
        a.type = element_index(a_obj["element"].get<std::string>());
        a.pos = Vec3{pos_arr[0].get<double>(), pos_arr[1].get<double>(),
                     pos_arr[2].get<double>()};

        if (a_obj.contains("force")) {
          const auto &f = a_obj["force"];
          if (!f.is_array() || f.size() != 3)
            return fail("atom 'force' must be an array of 3 doubles");
          a.force =
              Vec3{f[0].get<double>(), f[1].get<double>(), f[2].get<double>()};
        }

        cfg.atoms.push_back(std::move(a));
      }

      configs.push_back(std::move(cfg));
    }
  } catch (const json::exception &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }

  return configs;
}

} // namespace potfit::io
