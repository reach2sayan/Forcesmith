#pragma once

// Step 8: ForceCalculator concept + PairForceCalculator implementation.

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"
#include "potfit/events/signals.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <ranges>

namespace potfit {

template <typename T>
concept ForceCalculator = requires(T calc, Configuration &cfg) {
  { calc.eval_forces(cfg) } -> std::same_as<void>;
};

// Two-body (pair) force calculator.
// Iterates each atom's neighbor list and accumulates into the configuration's
// calc_force / calc_energy / calc_stress fields.  The neighbor list must be
// built (with potentials attached) before calling eval_forces.
struct PairForceCalculator {
  std::uint64_t conf_index = 0; // forwarded to on_force_eval signal payload

  void eval_forces(Configuration &cfg) const {
    cfg.calc_energy = 0.0;
    cfg.calc_stress = SymTens::Zero();
    std::ranges::for_each(cfg.atoms,
                          [](auto &atom) { atom.calc_force = Vec3::Zero(); });

    for (auto &atom : cfg.atoms) {
      for (const auto &nb :
           atom.neighbors | std::views::filter([](const auto &nb) {
             return static_cast<bool>(nb.pot);
           })) {

        const double r = nb.dist.norm();
        const double inv_r = 1.0 / r;
        const double phi = nb.pot->eval(r);
        const double dphi = nb.pot->deriv(r);
        const Vec3 fvec = (dphi * inv_r) * nb.dist;

        atom.calc_force += fvec;
        // Factor 1/2: the full neighbor list counts each pair twice.
        cfg.calc_energy += 0.5 * phi;
        cfg.calc_stress += 0.5 * nb.dist * fvec.transpose();
      }
    }

    double rms2 = std::transform_reduce(
        cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{},
        [](const auto &atom) { return atom.calc_force.squaredNorm(); });
    const double rms =
        cfg.atoms.empty()
            ? 0.0
            : std::sqrt(rms2 / static_cast<double>(cfg.atoms.size()));

    events::on_force_eval(events::ForceEvalStats{conf_index, rms});
  }
};

static_assert(ForceCalculator<PairForceCalculator>);

} // namespace potfit
