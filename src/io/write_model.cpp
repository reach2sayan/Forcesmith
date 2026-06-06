#include "forcesmith/io/write_model.hpp"

#include "forcesmith/core/radial_potential.hpp"
#include "forcesmith/io/config_reader.hpp" // ParseError
#include "forcesmith/io/output_writer.hpp"

#include <boost/leaf/error.hpp>
#include <iostream>
#include <string>
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
  // non-finite-tabulation failures through the result channel; those errors
  // propagate via leaf so the model-output path (Forcesmith::write, checkpoints)
  // never throws across the API. The native writers are family-specific and
  // pull in the IO layer, so they stay out of the core concept — the concrete
  // calculator is recovered through the typed escape hatch instead.
  if (const auto *calc = model.target<PairForceCalculator>()) {
    const std::vector<RadialPotential> pots(calc->pair.begin(),
                                            calc->pair.end());
    if (fmt == "lammps") {
      return write_lammps(path, pots);
    }
    if (fmt == "imd") {
      return write_imd(path, pots);
    }
    return write_native(path, pots);
  }
  if (const auto *calc = model.target<EAMForceCalculator>()) {
    warn_non_native("EAM");
    return write_native_eam(path, *calc);
  }
  if (const auto *calc = model.target<ADPForceCalculator>()) {
    warn_non_native("ADP");
    return write_native_adp(path, *calc);
  }
  if (const auto *calc = model.target<AngularForceCalculator>()) {
    warn_non_native("angular");
    return write_native_angular(path, *calc);
  }
  if (const auto *calc = model.target<TersoffForceCalculator>()) {
    warn_non_native("tersoff");
    return write_native_tersoff(path, *calc);
  }
  if (const auto *calc = model.target<StiwebForceCalculator>()) {
    warn_non_native("stiweb");
    return write_native_stiweb(path, *calc);
  }
  if (const auto *calc = model.target<ACSF>()) {
    warn_non_native("acsf");
    return write_native_acsf(path, *calc);
  }
  if (const auto *calc = model.target<SoapModel>()) {
    warn_non_native("soap");
    return write_native_soap(path, *calc);
  }
  if (const auto *calc = model.target<LMBTR>()) {
    warn_non_native("lmbtr");
    return write_native_lmbtr(path, *calc);
  }
  return leaf::new_error(ParseError{"output not implemented for this model", 0});
}

} // namespace forcesmith::io
