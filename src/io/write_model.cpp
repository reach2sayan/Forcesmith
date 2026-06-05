#include "forcesmith/io/write_model.hpp"

#include "forcesmith/core/potential_base.hpp"
#include "forcesmith/io/config_reader.hpp" // ParseError
#include "forcesmith/io/output_writer.hpp"

#include <boost/leaf/error.hpp>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace forcesmith::io {

namespace leaf = boost::leaf;

boost::leaf::result<void> write_model(const ForceCalculator &model,
                                      const std::filesystem::path &path,
                                      std::string_view format) {
  const std::string fmt(format);

  auto warn_non_native = [&](const char *which) {
    if (fmt != "native") {
      std::cerr << "warning: '" << fmt << "' output unsupported for " << which
                << "; writing native JSON\n";
    }
  };

  // Every native writer returns leaf::result<void>, reporting file-open and
  // non-finite-tabulation failures through the result channel. The visitor
  // forwards those errors so the model-output path (Forcesmith::write, checkpoints)
  // propagates them via leaf without throwing across the API.
  BOOST_LEAF_AUTO(
      wrote,
      std::visit(
          [&](const auto &calc) -> leaf::result<bool> {
            using T = std::decay_t<decltype(calc)>;
            if constexpr (std::is_same_v<T, PairForceCalculator>) {
              const std::vector<Potential> pots(calc.pair.begin(),
                                                calc.pair.end());
              if (fmt == "lammps") {
                BOOST_LEAF_CHECK(write_lammps(path, pots));
              } else if (fmt == "imd") {
                BOOST_LEAF_CHECK(write_imd(path, pots));
              } else {
                BOOST_LEAF_CHECK(write_native(path, pots));
              }
              return true;
            } else if constexpr (std::is_same_v<T, EAMForceCalculator>) {
              warn_non_native("EAM");
              BOOST_LEAF_CHECK(write_native_eam(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, ADPForceCalculator>) {
              warn_non_native("ADP");
              BOOST_LEAF_CHECK(write_native_adp(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, AngularForceCalculator>) {
              warn_non_native("angular");
              BOOST_LEAF_CHECK(write_native_angular(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, TersoffForceCalculator>) {
              warn_non_native("tersoff");
              BOOST_LEAF_CHECK(write_native_tersoff(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, StiwebForceCalculator>) {
              warn_non_native("stiweb");
              BOOST_LEAF_CHECK(write_native_stiweb(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, ACSF>) {
              warn_non_native("acsf");
              BOOST_LEAF_CHECK(write_native_acsf(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, SoapModel>) {
              warn_non_native("soap");
              BOOST_LEAF_CHECK(write_native_soap(path, calc));
              return true;
            } else if constexpr (std::is_same_v<T, LMBTR>) {
              warn_non_native("lmbtr");
              BOOST_LEAF_CHECK(write_native_lmbtr(path, calc));
              return true;
            } else {
              return false;
            }
          },
          model));

  if (!wrote) {
    return leaf::new_error(
        ParseError{"output not implemented for this model", 0});
  }
  return {};
}

} // namespace forcesmith::io
