#pragma once

#include "forcesmith/core/potential_base.hpp"
#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/angular_force.hpp"
#include "forcesmith/force/eam_force.hpp"
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/potentials/acsf.hpp"
#include "forcesmith/potentials/lmbtr.hpp"
#include "forcesmith/potentials/soap.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>
#include <vector>

namespace forcesmith::io {

static constexpr int kDefaultKnots = 500;
boost::leaf::result<void> write_lammps(const std::filesystem::path &path,
                                       const std::vector<Potential> &potentials);

boost::leaf::result<void> write_imd(const std::filesystem::path &path,
                                    const std::vector<Potential> &potentials);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const std::vector<Potential> &potentials,
                                       int nknots = kDefaultKnots);

// Write a fitted EAM model as structured tabulated JSON
// {model:"eam", ntypes, pair, density, embedding} where each section
// is {format:"tabulated", potentials:[{rmin,rmax,knots}]}.
boost::leaf::result<void> write_native_eam(const std::filesystem::path &path,
                                           const EAMForceCalculator &eam,
                                           int nknots = kDefaultKnots);

// ADP/Angular are radial/tabulated like EAM: each table section is sampled on a
// uniform grid. Mirrors the reader's input envelopes.
//   ADP:     {model:"adp",     ntypes, pair, density, embedding, dipole,
//   quadrupole} Angular: {model:"angular", ntypes, pair, radial, angular}
boost::leaf::result<void> write_native_adp(const std::filesystem::path &path,
                                           const ADPForceCalculator &adp,
                                           int nknots = kDefaultKnots);
boost::leaf::result<void>
write_native_angular(const std::filesystem::path &path,
                     const AngularForceCalculator &ang,
                     int nknots = kDefaultKnots);

// Tersoff/Stiweb are analytic: dump their fitted Param values, mirroring the
// reader's JSON schema so the output re-parses as a startpot.
//   Tersoff: {model:"tersoff", ntypes,
//   potentials:[{A,B,lambda,mu,beta,n,c,d,h,R,S,omega}]} Stiweb:
//   {model:"stiweb",  ntypes, potentials:[{A,B,p,q,delta,a1,gamma,a2}],
//   lambda:[…]}
boost::leaf::result<void>
write_native_tersoff(const std::filesystem::path &path,
                     const TersoffForceCalculator &ters);
boost::leaf::result<void>
write_native_stiweb(const std::filesystem::path &path,
                    const StiwebForceCalculator &sw);

// ACSF ML model: dump the fixed descriptor hyperparameters (G1/G2/G3/G4/G5)
// plus the fitted head, mirroring the reader's JSON envelope so the output
// re-parses as a startpot.
//   {model:"ml", ntypes, descriptor:{type:"acsf", rcut, g1, g2:[{eta,rs}],
//    g3:[{kappa}], g4:[{eta,zeta,lambda}], g5:[{eta,zeta,lambda}]},
//    heads:[{type, …}]}
boost::leaf::result<void> write_native_acsf(const std::filesystem::path &path,
                                            const ACSF &ml);

// SOAP ML model: descriptor hyperparameters + per-type linear heads,
// using the head's generic serialization surface so any head round-trips.
//   {model:"ml", ntypes, descriptor:{type:"soap", n_max,l_max,rcut,sigma},
//    heads:[{type, …}]}
boost::leaf::result<void> write_native_soap(const std::filesystem::path &path,
                                            const SoapModel &soap);

// LMBTR ML model: descriptor hyperparameters (k2/k3 grids, weighting) + heads.
//   {model:"ml", ntypes, descriptor:{type:"lmbtr", rcut, weight_scale,
//   normalize,
//    k2:{min,max,n,sigma}, k3:{min,max,n,sigma}}, heads:[{type, …}]}
boost::leaf::result<void> write_native_lmbtr(const std::filesystem::path &path,
                                             const LMBTR &ml);

} // namespace forcesmith::io
