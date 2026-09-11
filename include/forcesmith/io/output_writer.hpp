#pragma once

#include "forcesmith/core/radial_potential.hpp"
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
boost::leaf::result<void>
write_lammps(const std::filesystem::path &path,
             const std::vector<RadialPotential> &potentials);

boost::leaf::result<void>
write_imd(const std::filesystem::path &path,
          const std::vector<RadialPotential> &potentials);

boost::leaf::result<void>
write_native(const std::filesystem::path &path,
             const std::vector<RadialPotential> &potentials,
             int nknots = kDefaultKnots);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const EAMForceCalculator &eam,
                                       int nknots = kDefaultKnots);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const ADPForceCalculator &adp,
                                       int nknots = kDefaultKnots);
boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const AngularForceCalculator &ang,
                                       int nknots = kDefaultKnots);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const TersoffForceCalculator &ters);
boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const StiwebForceCalculator &sw);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const ACSF &ml);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const SoapModel &soap);

boost::leaf::result<void> write_native(const std::filesystem::path &path,
                                       const LMBTR &ml);

} // namespace forcesmith::io
