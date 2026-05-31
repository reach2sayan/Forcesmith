#include "potfit/core/checkpoint.hpp"
#include "potfit/io/output_writer.hpp"
#include "potfit/io/potential_reader.hpp"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/leaf/error.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <iterator>
#include <string>

namespace potfit {

namespace leaf = boost::leaf;

// BOOST_LEAF_CHECK/AUTO expand to a GNU statement expression ({ ... }), which
// Clang flags under -Wgnu-statement-expression-from-macro-expansion. It is a
// deliberate Boost.LEAF idiom, not our code — suppress it for this TU rather
// than weakening the warning project-wide. The push is placed after the first
// declaration (the namespace alias above) so it lands in clangd's main-file body
// rather than the preamble; otherwise the push/pop pair is split across the
// preamble boundary and clangd reports a spurious "no matching push" at the pop.
// clang-format off keeps the `ignored` pragma on one physical line.
// clang-format off
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
#endif
// clang-format on

namespace {

[[nodiscard]] leaf::error_id err(std::string msg) {
  return leaf::new_error(CheckpointError{std::move(msg)});
}

[[nodiscard]] std::filesystem::path cfg_path(const std::filesystem::path &p) {
  return std::filesystem::path(p.string() + ".cfg.bin");
}
[[nodiscard]] std::filesystem::path pot_path(const std::filesystem::path &p) {
  return std::filesystem::path(p.string() + ".pot");
}

// Open a stream, mapping failure into the result channel.
template <class Stream>
[[nodiscard]] leaf::result<Stream> open_file(const std::filesystem::path &path,
                                             std::ios::openmode mode,
                                             std::string_view verb) {
  Stream f(path, mode);
  if (!f)
    return err("cannot open for " + std::string(verb) + ": " + path.string());
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

[[nodiscard]] leaf::result<void>
save_potentials(const std::filesystem::path &path,
                const std::vector<Potential> &pots) {
  try {
    io::write_native(path, pots);
  } catch (const std::exception &e) {
    return err(e.what());
  }
  return {};
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

[[nodiscard]] leaf::result<void>
load_potentials(const std::filesystem::path &path,
                std::vector<Potential> &potentials) {
  BOOST_LEAF_AUTO(f, open_file<std::ifstream>(path, std::ios::in, "reading"));
  std::string text{std::istreambuf_iterator<char>(f),
                   std::istreambuf_iterator<char>{}};
  BOOST_LEAF_AUTO(pots, io::parse_potential(text));
  potentials = std::move(pots);
  return {};
}

} // namespace

leaf::result<void> CheckpointWriter::write() const {
  if (!configs_) {
    return err("write() called without configs()");
  }
  if (!pots_) {
    return err("write() called without potentials()");
  }

  BOOST_LEAF_CHECK(save_configs(cfg_path(prefix_), *configs_));
  BOOST_LEAF_CHECK(save_potentials(pot_path(prefix_), *pots_));
  return {};
}

leaf::result<void>
CheckpointReader::read(std::vector<Configuration> &configs,
                       std::vector<Potential> &potentials) const {
  BOOST_LEAF_CHECK(load_configs(cfg_path(prefix_), configs));
  BOOST_LEAF_CHECK(load_potentials(pot_path(prefix_), potentials));
  return {};
}

} // namespace potfit

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
