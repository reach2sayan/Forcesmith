#include "forcesmith/force/pair_force.hpp"
#include "forcesmith/force/eval_scope.hpp"
#include "forcesmith/force/kernels.hpp"
#include "forcesmith/force/param_jacobian.hpp"

#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <ranges>
#include <utility>

namespace forcesmith {

std::optional<PairBond>
PairForceCalculator::make_pair_bond(Atom &ai, const NeighborEntry &nb,
                                    const RadialPotential &pot) {
  const double r = nb.dist.norm();
  if (r < 1e-14) {
    return std::nullopt;
  }
  const auto [rmin, rmax] = pot.span();
  if (r < rmin || r >= rmax) {
    return std::nullopt;
  }
  PairBond pb{&ai, &pot, nb.dist, r, 1.0 / r};
  pb.site = nb.sites[kSitePhi]; // φ cache handle (none if not primed)
  return pb;
}

PairBond PairForceCalculator::add_pair_force(PairBond &&pb) {
  const auto [phi, dphi] = eval_deriv_cached(*pb.pot, pb.site, pb.r);
  pb.phi = phi;
  pb.force = (dphi * pb.inv_r) * pb.d;
  return std::move(pb);
}

PairBond PairForceCalculator::accumulate_pair(Configuration &cfg,
                                              PairBond &&pb) {
  force::accumulate_pair(cfg, *pb.ai, pb.d, pb.force, pb.phi);
  return std::move(pb);
}

void PairForceCalculator::eval_forces(Configuration &cfg) const {
  force::with_eval_scope(cfg, max_cutoff(), conf_index, [&] {
    for (auto &atom : cfg.atoms) {
      for (const auto &nb : atom.neighbors) {
        make_pair_bond(atom, nb, pair[atom, *nb.neighbor])
            .transform(add_pair_force)
            .transform(std::bind_front(&PairForceCalculator::accumulate_pair,
                                       std::ref(cfg)));
      }
    }
  });
}

void PairForceCalculator::prepare(std::span<Configuration> configs) const {
  build_all_neighbor_lists(configs, max_cutoff());
  // Phase 2 — single-threaded priming: prepare_site mutates shared spline
  // objects, so it stays serial.
  for (Configuration &cfg : configs) {
    for (Atom &ai : cfg.atoms) {
      for (NeighborEntry &nb : ai.neighbors) {
        const double r = nb.dist.norm();
        nb.sites[kSitePhi] = force::prime_site(pair[ai, *nb.neighbor], r);
      }
    }
  }
}

bool PairForceCalculator::has_analytic_jacobian() const {
  return force::analytic_jacobian_available(*this);
}

void PairForceCalculator::write_param_jacobian(Configuration &cfg, int row0,
                                               double energy_weight,
                                               double stress_weight,
                                               Eigen::MatrixXd &fjac) const {
  build_neighbor_list(cfg, max_cutoff());
  const auto cols = force::table_columns(*this);
  const force::JacRows rows(cfg, row0, stress_weight);
  const double inv_volume = 1.0 / bc_volume(cfg.bc);

  force::Partials phi;
  for (const auto &[i, ai] : cfg.atoms | std::views::enumerate) {
    const int atom_row = rows.force0 + 3 * static_cast<int>(i);
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      if (r < 1e-14) {
        continue;
      }
      const Atom &aj = *nb.neighbor;
      const auto &pot = pair[ai, aj];
      const auto [rmin, rmax] = pot.span();
      if (r < rmin || r >= rmax) { // exactly make_pair_bond's gate
        continue;
      }
      phi.take(pot, r);
      force::add_radial_bond(
          fjac, rows, atom_row,
          cols.of[0][pair_ordinal(ai.type, aj.type, pair.ntypes())], nb.dist,
          1.0 / r, phi, 1.0, /*with_energy=*/true, energy_weight, stress_weight,
          inv_volume);
    }
  }
}

PairForceCalculator
make_pair_force_calculator(std::vector<RadialPotential> potentials) {
  const std::size_t n = potentials.size();
  const std::size_t ntypes = static_cast<std::size_t>(std::lround(
      (-1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0));
  PairForceCalculator calc;
  calc.ntypes = ntypes;
  calc.pair.reserve(ntypes);
  std::ranges::move(potentials, std::back_inserter(calc.pair));
  return calc;
}

} // namespace forcesmith
