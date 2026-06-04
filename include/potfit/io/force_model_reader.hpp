#pragma once

// Model-aware JSON reader that returns a fully constructed force calculator.
//
// JSON format — top-level object with a "model" key:
//
//   "pair"    → PairForceCalculator   (delegates to parse_potential)
//   "eam"     → EAMForceCalculator
//   "adp"     → ADPForceCalculator
//   "angular" → AngularForceCalculator
//   "tersoff" → TersoffForceCalculator
//   "stiweb"  → StiwebForceCalculator
//   "ml"      → ACSF / SoapModel (descriptor + per-type head)

#include "potfit/force/force_calculator.hpp"
#include "potfit/io/config_reader.hpp" // ParseError, indirectly pulls in core types

#include <boost/leaf/result.hpp>
#include <string_view>

namespace potfit::io {

// Parse a model-aware JSON potential file.
// The "ntypes" key (default 1) sets the number of element types.
//
// "pair"    — requires top-level "format" and "potentials" (same as
// parse_potential).
//             Returns paircol = ntypes*(ntypes+1)/2 potentials.
//
// "eam"     — requires "pair", "density", "embedding" sub-objects,
//             each of shape {"format": ..., "potentials": [...]}.
//
// "adp"     — same as "eam" plus "dipole" and "quadrupole" sub-objects.
//
// "angular" — requires "pair", "radial", "angular" sub-objects.
//
// "tersoff" — requires "potentials" array of paircol objects with
//             fields: A, B, lambda, mu, beta, n, c, d, h, R, S.
//
// "stiweb"  — requires "potentials" array of paircol objects with
//             fields: A, B, p, q, a, sigma, lambda, gamma.
boost::leaf::result<ForceCalculator> parse_force_model(std::string_view input);

} // namespace potfit::io
