#include "forcesmith/io/loaders.hpp"

#include "forcesmith/io/config_reader.hpp" // ParseError
#include "forcesmith/io/force_model_reader.hpp"
#include "forcesmith/io/json_util.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>
#include <string>

namespace forcesmith::io {

namespace leaf = boost::leaf;

namespace {
[[nodiscard]] leaf::result<std::string>
read_file(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    return leaf::new_error(ParseError{"cannot open file: " + path.string(), 0});
  }
  return std::string(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
}
} // namespace

leaf::result<void> load_configs(const std::filesystem::path &path,
                                Forcesmith &session) {
  BOOST_LEAF_AUTO(text, read_file(path));

  return catch_json([&]() -> leaf::result<void> {
    const nlohmann::json j = nlohmann::json::parse(text);
    if (!j.is_array()) {
      return leaf::new_error(
          ParseError{"top-level config JSON must be an array", 0});
    }

    for (const auto &rec : j) {
      BOOST_LEAF_AUTO(cfg, Configuration::from_text(rec.dump()));
      session.add_configuration(std::move(cfg));
    }
    return {};
  });
}

leaf::result<void> load_model(const std::filesystem::path &path,
                              Forcesmith &session) {
  BOOST_LEAF_AUTO(text, read_file(path));
  BOOST_LEAF_AUTO(model, parse_force_model(text));
  return session.seed_force_model(std::move(model));
}

} // namespace forcesmith::io
