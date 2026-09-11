#include "forcesmith/io/write_model.hpp"

#include "forcesmith/core/families.hpp"
#include "forcesmith/core/radial_potential.hpp"
#include "forcesmith/io/config_reader.hpp" // ParseError
#include "forcesmith/io/output_writer.hpp"

#include <boost/leaf/error.hpp>
#include <boost/mp11/algorithm.hpp>

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace forcesmith::io {

namespace leaf = boost::leaf;

namespace {

template <CModelFamily T>
leaf::result<void> write_one(const T &calc, const std::filesystem::path &path,
                             const std::string &fmt) {
  if constexpr (std::same_as<T, PairForceCalculator>) {
    const std::vector<RadialPotential> pots(calc.pair.begin(), calc.pair.end());
    if (fmt == "lammps") {
      return write_lammps(path, pots);
    } else if (fmt == "imd") {
      return write_imd(path, pots);
    }
    return write_native(path, pots);
  } else {
    if (fmt != "native") {
      std::cerr << "warning: '" << fmt << "' output unsupported for "
                << family_name<T> << "; writing native JSON\n";
    }
    return write_native(path, calc);
  }
}

} // namespace

boost::leaf::result<void> write_model(const ForceCalculator &model,
                                      const std::filesystem::path &path,
                                      std::string_view format) {
  const std::string fmt(format);
  leaf::result<void> done{};
  const bool known = visit_family<ModelFamilies>(
      model, [&](const auto &calc) { done = write_one(calc, path, fmt); });
  if (!known) {
    return leaf::new_error(
        ParseError{"output not implemented for this model", 0});
  }
  return done;
}

} // namespace forcesmith::io
