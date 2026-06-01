#pragma once

// Model-aware output dispatch, shared by the CLI and PotFit::write.
// Picks the right writer for the force-model variant and the requested format
// (native | lammps | imd). lammps/imd apply only to the pair model; other
// models always emit native JSON (a warning is logged if another format was
// requested).

#include "potfit/force/force_calculator.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>
#include <string_view>

namespace potfit::io {

boost::leaf::result<void> write_model(const ForceCalculator &model,
                                      const std::filesystem::path &path,
                                      std::string_view format = "native");

} // namespace potfit::io
