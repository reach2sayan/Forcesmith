#include "potfit/optimization/potfit_functor.hpp"

#include "potfit/events/signals.hpp"
#include "potfit/force/pair_force.hpp"

#include <cmath>
#include <ranges>

namespace potfit {

namespace {

int count_residuals(std::span<Configuration> configs, double stress_weight) {
  int n = 0;
  for (const auto& cfg : configs)
    n += static_cast<int>(3 * cfg.atoms.size()) + 1;
  if (stress_weight > 0.0)
    n += 6 * static_cast<int>(configs.size());
  return n;
}

// Build a PairForceCalculator that owns a copy of the given potentials.
// ntypes is inferred from paircol = ntypes*(ntypes+1)/2.
PairForceCalculator make_pair_calc(std::span<Potential> potentials) {
  const int n_pots = static_cast<int>(potentials.size());
  const int ntypes = static_cast<int>(std::lround((-1.0 + std::sqrt(1.0 + 8.0 * n_pots)) / 2.0));
  PairForceCalculator calc;
  calc.pair.reserve(ntypes);
  for (const auto& p : potentials) calc.pair.emplace_back(p);
  return calc;
}

} // namespace

PotfitFunctor::PotfitFunctor(std::span<Configuration> configs,
                             ForceCalculator model,
                             double energy_weight, double stress_weight)
    : configs_(configs),
      model_(std::move(model)),
      energy_weight_(energy_weight),
      stress_weight_(stress_weight),
      inputs_{std::visit([](const auto& m){ return m.param_count(); }, model_)},
      values_{count_residuals(configs, stress_weight)} {}

PotfitFunctor::PotfitFunctor(std::span<Configuration> configs,
                             std::span<Potential> potentials,
                             double energy_weight, double stress_weight)
    : PotfitFunctor(configs,
                    ForceCalculator{make_pair_calc(potentials)},
                    energy_weight, stress_weight) {}

int PotfitFunctor::operator()(const Eigen::VectorXd &x,
                              Eigen::VectorXd &fvec) const {
  std::visit([&](auto& m){ m.scatter_params(x, 0); }, model_);

  int row = 0;
  for (auto [c, cfg] : std::views::enumerate(configs_)) {
    std::visit([&](auto& m){ m.eval_forces(cfg); }, model_);

    for (const auto &atom : cfg.atoms) {
      fvec[row++] = atom.calc_force[0] - atom.force[0];
      fvec[row++] = atom.calc_force[1] - atom.force[1];
      fvec[row++] = atom.calc_force[2] - atom.force[2];
    }
    fvec[row++] = energy_weight_ * (cfg.calc_energy - cfg.energy);
    if (stress_weight_ > 0.0) {
      fvec[row++] = stress_weight_ * (cfg.calc_stress(0,0) - cfg.stress(0,0));
      fvec[row++] = stress_weight_ * (cfg.calc_stress(1,1) - cfg.stress(1,1));
      fvec[row++] = stress_weight_ * (cfg.calc_stress(2,2) - cfg.stress(2,2));
      fvec[row++] = stress_weight_ * (cfg.calc_stress(0,1) - cfg.stress(0,1));
      fvec[row++] = stress_weight_ * (cfg.calc_stress(0,2) - cfg.stress(0,2));
      fvec[row++] = stress_weight_ * (cfg.calc_stress(1,2) - cfg.stress(1,2));
    }
  }

  events::on_iteration(
      events::IterationStats{++iter_, fvec.squaredNorm(), 0.0});

  return 0;
}

int PotfitFunctor::df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
  constexpr double delta = 1e-5;
  Eigen::VectorXd fp(values_), fm(values_);
  Eigen::VectorXd xp = x;

  for (int j : std::views::iota(0, inputs_)) {
    xp[j] += delta;
    std::invoke(*this, xp, fp);
    xp[j] -= 2.0 * delta;
    std::invoke(*this, xp, fm);
    xp[j] += delta;

    fjac.col(j) = (fp - fm) / (2.0 * delta);
  }
  return 0;
}

} // namespace potfit
