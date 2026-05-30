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

static std::filesystem::path cfg_path(const std::filesystem::path &p) {
  return std::filesystem::path(p.string() + ".cfg.bin");
}
static std::filesystem::path pot_path(const std::filesystem::path &p) {
  return std::filesystem::path(p.string() + ".pot");
}

leaf::result<void> CheckpointWriter::write() const {
  if (!configs_)
    return leaf::new_error(CheckpointError{"write() called without configs()"});
  if (!pots_)
    return leaf::new_error(
        CheckpointError{"write() called without potentials()"});

  {
    std::ofstream f(cfg_path(prefix_), std::ios::binary);
    if (!f)
      return leaf::new_error(CheckpointError{"cannot open for writing: " +
                                             cfg_path(prefix_).string()});
    boost::archive::binary_oarchive ar(f);
    ar &*configs_;
  }

  try {
    io::write_native(pot_path(prefix_), *pots_);
  } catch (const std::exception &e) {
    return leaf::new_error(CheckpointError{e.what()});
  }

  return {};
}

leaf::result<void>
CheckpointReader::read(std::vector<Configuration> &configs,
                       std::vector<Potential> &potentials) const {
  {
    std::ifstream f(cfg_path(prefix_), std::ios::binary);
    if (!f)
      return leaf::new_error(CheckpointError{"cannot open for reading: " +
                                             cfg_path(prefix_).string()});
    try {
      boost::archive::binary_iarchive ar(f);
      ar & configs;
    } catch (const std::exception &e) {
      return leaf::new_error(CheckpointError{"failed to deserialize configs: " +
                                             std::string(e.what())});
    }
  }

  {
    std::ifstream f(pot_path(prefix_));
    if (!f)
      return leaf::new_error(CheckpointError{"cannot open for reading: " +
                                             pot_path(prefix_).string()});

    std::string text{std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>{}};

    auto res = io::parse_potential(text);
    if (!res)
      return res.error();
    potentials = std::move(*res);
  }

  return {};
}

} // namespace potfit
