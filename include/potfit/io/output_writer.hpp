#pragma once

// Step 12: LAMMPS / IMD / native output writers.

#include "potfit/core/potential_base.hpp"
#include "potfit/force/eam_force.hpp"

#include <filesystem>
#include <vector>

namespace potfit::io {

// Sample each potential on a uniform grid and write JSON tabulated format.
// Works correctly for both SplinePotential and analytic types (LJ, Morse,
// etc.).
static constexpr int kDefaultKnots = 500;
void write_lammps(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials);

void write_imd(const std::filesystem::path &path,
               const std::vector<Potential> &potentials);

void write_native(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials,
                  int nknots = kDefaultKnots);

// Write a fitted EAM model as structured tabulated JSON, mirroring the input
// envelope: {model:"eam", ntypes, pair, density, embedding} where each section
// is {format:"tabulated", potentials:[{rmin,rmax,knots}]}.
void write_native_eam(const std::filesystem::path &path,
                      const EAMForceCalculator &eam, int nknots = kDefaultKnots);

} // namespace potfit::io
