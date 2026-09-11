#include "forcesmith/core/checkpoint.hpp"
#include "forcesmith/io/file.hpp"
#include "forcesmith/io/force_model_reader.hpp"
#include "forcesmith/io/write_model.hpp"

#include <concepts>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/leaf/error.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <iterator>
#include <string>

namespace forcesmith {

namespace leaf = boost::leaf;

// clang-format on

namespace {

[[nodiscard]] leaf::error_id err(std::string msg) {
  return leaf::new_error(CheckpointError{std::move(msg)});
}

[[nodiscard]] std::filesystem::path cfg_path(const std::filesystem::path &p) {
  return std::filesystem::path(p.string() + ".cfg.bin");
}
[[nodiscard]] std::filesystem::path model_path(const std::filesystem::path &p) {
  return std::filesystem::path(p.string() + ".model.json");
}

template <class Stream>
  requires std::constructible_from<Stream, const std::filesystem::path &,
                                   std::ios::openmode> &&
           std::derived_from<Stream, std::ios_base>
[[nodiscard]] leaf::result<Stream> open_file(const std::filesystem::path &path,
                                             std::ios::openmode mode,
                                             std::string_view verb) {
  Stream f(path, mode);
  if (!f) {
    return err("cannot open for " + std::string(verb) + ": " + path.string());
  }
  return f;
}

[[nodiscard]] leaf::result<void>
save_configs(const std::filesystem::path &path,
             const std::vector<Configuration> &configs) {
  BOOST_LEAF_AUTO(f,
                  open_file<std::ofstream>(path, std::ios::binary, "writing"));
  boost::archive::binary_oarchive ar(f);
  ar & configs;
  return {};
}

[[nodiscard]] leaf::result<void> save_model(const std::filesystem::path &path,
                                            const ForceCalculator &model) {
  return io::write_model(model, path, "native");
}

[[nodiscard]] leaf::result<void>
load_configs(const std::filesystem::path &path,
             std::vector<Configuration> &configs) {
  BOOST_LEAF_AUTO(f,
                  open_file<std::ifstream>(path, std::ios::binary, "reading"));
  try {
    boost::archive::binary_iarchive ar(f);
    ar & configs;
  } catch (const std::exception &e) {
    return err("failed to deserialize configs: " + std::string(e.what()));
  }
  return {};
}

[[nodiscard]] leaf::result<void> load_model(const std::filesystem::path &path,
                                            ForceCalculator &model) {
  BOOST_LEAF_AUTO(text, io::read_file(path));
  BOOST_LEAF_AUTO(m, io::parse_force_model(text));
  model = std::move(m);
  return {};
}

} // namespace

leaf::result<void> CheckpointWriter::write() const {
  if (!configs_) {
    return err("write() called without configs()");
  }
  if (!model_) {
    return err("write() called without model()");
  }

  BOOST_LEAF_CHECK(save_configs(cfg_path(prefix_), *configs_));
  BOOST_LEAF_CHECK(save_model(model_path(prefix_), *model_));
  return {};
}

leaf::result<void> CheckpointReader::read(std::vector<Configuration> &configs,
                                          ForceCalculator &model) const {
  BOOST_LEAF_CHECK(load_configs(cfg_path(prefix_), configs));
  BOOST_LEAF_CHECK(load_model(model_path(prefix_), model));
  return {};
}

} // namespace forcesmith
