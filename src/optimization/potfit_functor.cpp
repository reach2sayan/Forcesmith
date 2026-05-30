#include "potfit/optimization/potfit_functor.hpp"

#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/force_calculator.hpp"
#include "potfit/force/pair_force.hpp"

#include <algorithm>
#include <boost/hof/function.hpp>
#include <cmath>
#include <numeric>
#include <ranges>
#include <span>
namespace potfit {

BOOST_HOF_STATIC_FUNCTION(count_potential_params) =
    [](std::span<Potential> potentials_) {
      return std::transform_reduce(
          potentials_.begin(), potentials_.end(), 0, std::plus<>{},
          [](const auto &p) { return p.param_count(); });
    };

BOOST_HOF_STATIC_FUNCTION(count_configurations) =
    [](std::span<Configuration> configs_) {
      return std::transform_reduce(
          configs_.begin(), configs_.end(), 0, std::plus<>{},
          [](const auto &cfg) {
            return static_cast<int>(3 * cfg.atoms.size()) + 1;
          });
    };

PotfitFunctor::PotfitFunctor(std::span<Configuration> configs,
                             std::span<Potential> potentials,
                             double energy_weight)
    : configs_(configs), potentials_(potentials), energy_weight_(energy_weight),
      inputs_{count_potential_params(potentials)},
      values_{count_configurations(configs)} {}

int PotfitFunctor::operator()(const Eigen::VectorXd &x,
                              Eigen::VectorXd &fvec) const {
  // Scatter x into potentials.
  int off = 0;
  std::ranges::for_each(potentials_, [&](auto &p) {
    p.scatter_params(x, off);
    off += p.param_count();
  });

  // Compute rcut = max of all span().second.
  double rcut =
      std::ranges::max(potentials_ | std::views::transform([](const auto &p) {
                         return p.span().second;
                       }));

  // Build PotentialPair from the flat potentials_ span (upper-triangular order).
  // ntypes satisfies ntypes*(ntypes+1)/2 == potentials_.size().
  const int n_pots  = static_cast<int>(potentials_.size());
  const int ntypes  = static_cast<int>(std::lround((-1.0 + std::sqrt(1.0 + 8.0 * n_pots)) / 2.0));
  PotentialPair pair_pots;
  pair_pots.reserve(ntypes);
  for (const auto& p : potentials_) pair_pots.emplace_back(p);

  int row = 0;
  for (auto [c, cfg] : std::views::enumerate(configs_)) {
    build_neighbor_list(cfg, rcut, pair_pots);
    PairForceCalculator calc;
    calc.conf_index = static_cast<std::uint64_t>(c);
    calc.eval_forces(cfg);

    for (const auto &atom : cfg.atoms) {
      fvec[row++] = atom.calc_force[0] - atom.force[0];
      fvec[row++] = atom.calc_force[1] - atom.force[1];
      fvec[row++] = atom.calc_force[2] - atom.force[2];
    }
    fvec[row++] = energy_weight_ * (cfg.calc_energy - cfg.energy);
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
    xp[j] += delta; // restore

    fjac.col(j) = (fp - fm) / (2.0 * delta);
  }
  return 0;
}

} // namespace potfit
