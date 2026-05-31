#pragma once

// Step 12: LAMMPS / IMD / native output writers.

#include "potfit/core/potential_base.hpp"
#include "potfit/force/adp_force.hpp"
#include "potfit/force/angular_force.hpp"
#include "potfit/force/eam_force.hpp"
#include "potfit/force/stiweb_force.hpp"
#include "potfit/force/tersoff_force.hpp"

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
                      const EAMForceCalculator &eam,
                      int nknots = kDefaultKnots);

// ADP/Angular are radial/tabulated like EAM: each table section is sampled on a
// uniform grid. Mirrors the reader's input envelopes.
//   ADP:     {model:"adp",     ntypes, pair, density, embedding, dipole,
//   quadrupole} Angular: {model:"angular", ntypes, pair, radial, angular}
void write_native_adp(const std::filesystem::path &path,
                      const ADPForceCalculator &adp,
                      int nknots = kDefaultKnots);
void write_native_angular(const std::filesystem::path &path,
                          const AngularForceCalculator &ang,
                          int nknots = kDefaultKnots);

// Tersoff/Stiweb are analytic: dump their fitted Param values, mirroring the
// reader's JSON schema so the output re-parses as a startpot.
//   Tersoff: {model:"tersoff", ntypes,
//   potentials:[{A,B,lambda,mu,beta,n,c,d,h,R,S,omega}]} Stiweb:
//   {model:"stiweb",  ntypes, potentials:[{A,B,p,q,delta,a1,gamma,a2}],
//   lambda:[…]}
void write_native_tersoff(const std::filesystem::path &path,
                          const TersoffForceCalculator &ters);
void write_native_stiweb(const std::filesystem::path &path,
                         const StiwebForceCalculator &sw);

} // namespace potfit::io
