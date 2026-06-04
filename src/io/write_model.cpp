#include "potfit/io/write_model.hpp"

#include "potfit/core/potential_base.hpp"
#include "potfit/io/config_reader.hpp" // ParseError
#include "potfit/io/output_writer.hpp"

#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace potfit::io {

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

  // The native writers throw std::runtime_error on failure (file open, or a
  // non-finite tabulation value). Funnel those into the result channel so the
  // model-output path (PotFit::write, checkpoints) never throws across the API.
  try {
    const bool wrote = std::visit(
        [&](const auto &calc) -> bool {
          using T = std::decay_t<decltype(calc)>;
          if constexpr (std::is_same_v<T, PairForceCalculator>) {
            const std::vector<Potential> pots(calc.pair.begin(),
                                              calc.pair.end());
            if (fmt == "lammps") {
              write_lammps(path, pots);
            } else if (fmt == "imd") {
              write_imd(path, pots);
            } else {
              write_native(path, pots);
            }
            return true;
          } else if constexpr (std::is_same_v<T, EAMForceCalculator>) {
            warn_non_native("EAM");
            write_native_eam(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, ADPForceCalculator>) {
            warn_non_native("ADP");
            write_native_adp(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, AngularForceCalculator>) {
            warn_non_native("angular");
            write_native_angular(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, TersoffForceCalculator>) {
            warn_non_native("tersoff");
            write_native_tersoff(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, StiwebForceCalculator>) {
            warn_non_native("stiweb");
            write_native_stiweb(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, ACSF>) {
            warn_non_native("acsf");
            write_native_acsf(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, SoapModel>) {
            warn_non_native("soap");
            write_native_soap(path, calc);
            return true;
          } else if constexpr (std::is_same_v<T, LMBTR>) {
            warn_non_native("lmbtr");
            write_native_lmbtr(path, calc);
            return true;
          } else {
            return false;
          }
        },
        model);

    if (!wrote) {
      return leaf::new_error(
          ParseError{"output not implemented for this model", 0});
    }
  } catch (const std::exception &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }
  return {};
}

} // namespace potfit::io
