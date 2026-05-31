#include "potfit/force/pair_force.hpp"

#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

namespace potfit {

namespace {

// ── 2-body pipeline ─────────────────────────────────────────────────────────
// Each i–j bond is threaded through the stages; an empty std::optional (atoms
// coincident, or r outside the potential's own range) short-circuits the chain,
// mirroring adp/tersoff/stiweb.
struct PairBond {
  Atom            *ai;                  // central atom
  const Potential *pot;                 // i–j pair potential φ
  Vec3             d;                   // pos_j − pos_i
  double           r, inv_r;            // |d| and 1/|d|
  double           phi = 0.0;           // pair energy φ(r)
  Vec3             force = Vec3::Zero(); // force on i
};

// Stage 1 — geometry + cutoff gate. The neighbor list is built with the global
// max_cutoff(); gate each contribution on this potential's own range
// [rmin, rmax). Empty for coincident atoms or out-of-range separations.
std::optional<PairBond> make_pair_bond(Atom &ai, const NeighborEntry &nb,
                                       const Potential &pot) {
  const double r = nb.dist.norm();
  if (r < 1e-14) {
    return std::nullopt;
  }
  const auto [rmin, rmax] = pot.span();
  if (r < rmin || r >= rmax) {
    return std::nullopt;
  }
  return PairBond{&ai, &pot, nb.dist, r, 1.0 / r};
}

// Stage 2 — radial force φ′(r).
PairBond add_pair_force(PairBond &&pb) {
  pb.phi = pb.pot->eval(pb.r);
  const double dphi = pb.pot->deriv(pb.r);
  pb.force = (dphi * pb.inv_r) * pb.d;
  return std::move(pb);
}

// Stage 3 — commit energy / force / virial (0.5 for the full neighbor list).
auto accumulate_pair(Configuration &cfg) {
  return [&cfg](PairBond &&pb) -> PairBond {
    pb.ai->calc_force += pb.force;
    cfg.calc_energy += 0.5 * pb.phi;
    // Virial: bond ⊗ force-on-partner = dist ⊗ (−force); 0.5 for the full list.
    cfg.calc_stress -= 0.5 * pb.d * pb.force.transpose();
    return std::move(pb);
  };
}

} // anonymous namespace

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

  // Each i–j bond flows: geometry + cutoff gate → radial force → commit.
  for (auto &atom : cfg.atoms) {
    for (const auto &nb : atom.neighbors) {
      make_pair_bond(atom, nb, pair[atom, *nb.neighbor])
          .transform(add_pair_force)
          .transform(accumulate_pair(cfg));
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg)});
}

PairForceCalculator
make_pair_force_calculator(std::vector<Potential> potentials) {
  const std::size_t n = potentials.size();
  const std::size_t ntypes = static_cast<std::size_t>(std::lround(
      (-1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0));
  PairForceCalculator calc;
  calc.ntypes = ntypes;
  calc.pair.reserve(ntypes);
  std::ranges::move(potentials, std::back_inserter(calc.pair));
  return calc;
}

} // namespace potfit
