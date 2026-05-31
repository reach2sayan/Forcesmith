#include "potfit/force/pair_force.hpp"

#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace potfit {

std::size_t PairForceCalculator::param_count() const {
  return std::transform_reduce(pair.begin(), pair.end(), std::size_t{0},
                               std::plus<>{},
                               [](const auto &p) { return p.param_count(); });
}

void PairForceCalculator::gather_params(Eigen::VectorXd &dst,
                                        std::size_t off) const {
  gather_range(pair, dst, off);
}

void PairForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                         std::size_t off) {
  scatter_range(pair, src, off);
}

double PairForceCalculator::max_cutoff() const {
  return std::transform_reduce(
      pair.begin(), pair.end(), 0.0,
      [](double a, double b) { return std::max(a, b); },
      [](const auto &p) { return p.span().second; });
}

void PairForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  std::ranges::for_each(cfg.atoms,
                        [](auto &atom) { atom.calc_force = Vec3::Zero(); });

  for (auto &atom : cfg.atoms) {
    for (const auto &nb : atom.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14)
        continue;
      const Potential &pot = pair[atom, *nb.neighbor];
      const double inv_r = 1.0 / r;
      const double phi = pot.eval(r);
      const double dphi = pot.deriv(r);
      const Vec3 fvec = (dphi * inv_r) * nb.dist;

      atom.calc_force += fvec;
      cfg.calc_energy += 0.5 * phi;
      cfg.calc_stress += 0.5 * nb.dist * fvec.transpose();
    }
  }

  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg)});
}

PairForceCalculator
make_pair_force_calculator(std::vector<Potential> potentials) {
  const std::size_t n = potentials.size();
  const std::size_t ntypes = static_cast<std::size_t>(std::lround(
      (-1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0));
  PairForceCalculator calc;
  calc.pair.reserve(ntypes);
  for (auto &p : potentials) {
    calc.pair.emplace_back(std::move(p));
  }
  return calc;
}

} // namespace potfit
