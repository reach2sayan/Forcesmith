#include "forcesmith/force/evaluate.hpp"

namespace forcesmith::force {

EvalResult evaluate(const ForceCalculator &calc, const Configuration &cfg) {
  Configuration scratch = cfg; // non-mutating: never touch the caller's config
  calc.eval_forces(scratch);

  EvalResult out;
  out.energy = scratch.calc_energy;
  out.stress = scratch.calc_stress;
  out.limit = scratch.calc_limit;

  out.forces.reserve(scratch.atoms.size());
  std::ranges::transform(scratch.atoms, std::back_inserter(out.forces),
                         [](const auto &a) { return a.calc_force; });
  return out;
}

} // namespace forcesmith::force
