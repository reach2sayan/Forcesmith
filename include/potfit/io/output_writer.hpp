#pragma once

// Step 12: LAMMPS / IMD / native output writers.

#include "potfit/core/potential_base.hpp"

#include <filesystem>
#include <vector>

namespace potfit::io {

void write_lammps(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials);

void write_imd(const std::filesystem::path &path,
               const std::vector<Potential> &potentials);

void write_native(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials);

} // namespace potfit::io
