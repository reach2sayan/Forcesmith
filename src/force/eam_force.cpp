#include "potfit/force/eam_force.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>

namespace potfit {

// Count the free globals (each is exactly one optimizer slot when not fixed).
static std::size_t free_globals(const std::vector<GlobalParam> &globals) {
  return std::ranges::count_if(globals,
                               [](const auto &g) { return !g.value.fixed; });
}

std::size_t EAMForceCalculator::param_count() const {
  auto count_params = [](const auto &xs) {
    return std::transform_reduce(xs.begin(), xs.end(), std::size_t{0},
                                 std::plus<>{},
                                 [](const auto &p) { return p.param_count(); });
  };
  return count_params(pair) + count_params(density) + count_params(embedding) +
         free_globals(globals);
}

void EAMForceCalculator::gather_params(Eigen::VectorXd &dst,
                                       std::size_t off) const {
  gather_range(pair, dst, off);
  gather_range(density, dst, off);
  gather_range(embedding, dst, off);
  // Globals follow the per-potential params; their linked slots are fixed and
  // therefore already excluded above.
  std::ranges::for_each(globals, [&](const auto &g) {
    if (!g.value.fixed) {
      dst[off++] = g.value.value;
    }
  });
}

void EAMForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                        std::size_t off) {
  scatter_range(pair, src, off);
  scatter_range(density, src, off);
  scatter_range(embedding, src, off);
  std::ranges::for_each(globals, [&](auto &g) {
    if (!g.value.fixed) {
      g.value.value = src[off++];
    }
  });
  broadcast_globals();
}

void EAMForceCalculator::broadcast_globals() {
  // pair (SymmetricMatrix) and density/embedding (TypeArray) are distinct
  // types, so select the table with a templated lambda rather than a ternary.
  auto write = [](auto &tbl, std::size_t idx, std::size_t param, double v) {
    (*std::next(tbl.begin(), static_cast<std::ptrdiff_t>(idx)))
        .set_param(param, v);
  };
  for (const auto &g : globals)
    for (const auto &lk : g.links) {
      const double v = g.value.value;
      switch (lk.region) {
        using enum potfit::GlobalParam::Link::LinkRegion;
      case PAIR:
        write(pair, lk.index, lk.param, v);
        break;
      case DENSITY:
        write(density, lk.index, lk.param, v);
        break;
      case EMBEDDING:
        write(embedding, lk.index, lk.param, v);
        break;
      };
    }
}

void EAMForceCalculator::finalize_globals() {
  auto fix = [](auto &tbl, std::size_t idx, std::size_t param) {
    (*std::next(tbl.begin(), static_cast<std::ptrdiff_t>(idx)))
        .set_fixed(param, true);
  };
  for (const auto &g : globals)
    for (const auto &lk : g.links) {
      switch (lk.region) {
        using enum potfit::GlobalParam::Link::LinkRegion;
      case PAIR:
        fix(pair, lk.index, lk.param);
        break;
      case DENSITY:
        fix(density, lk.index, lk.param);
        break;
      case EMBEDDING:
        fix(embedding, lk.index, lk.param);
        break;
      };
    }
  broadcast_globals();
}

double EAMForceCalculator::max_cutoff() const {
  auto max_range = [](const auto &range) {
    return std::transform_reduce(
        range.begin(), range.end(), 0.0,
        [](double a, double b) { return std::max(a, b); },
        [](const auto &p) { return p.span().second; });
  };
  return std::max(max_range(pair), max_range(density));
}

void EAMForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  // ── Zero all scratch and output fields ──────────────────────────────────
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  cfg.calc_limit = 0.0;
  for (auto &atom : cfg.atoms) {
    atom.calc_force = Vec3::Zero();
    atom.rho = 0.0;
    atom.gradF = 0.0;
  }

  // ── Pass 1: accumulate electron density ρ_i ─────────────────────────────
  // ρ_i = Σ_{j∈neighbors(i)} g_{t(j)}(r_ij)
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const double r = nb.dist.norm();
      const auto &g = density[*nb.neighbor];
      if (in_range(g, r)) {
        ai.rho += g.eval(r);
      }
    }
  }

  // ── After pass 1: embedding energy + gradF_i = dF_i/dρ_i ────────────────
  // Out-of-range ρ is clamped to the embedding table's [begin,end] and the
  // overshoot is punished via cfg.calc_limit (matches potfit's RESCALE branch,
  // force_eam.c:334-358): F(ρ) is evaluated at the clamped ρ, never
  // extrapolated.
  for (auto &ai : cfg.atoms) {
    const auto [rho_begin, rho_end] = embedding[ai].span();
    if (ai.rho > rho_end) {
      const double d = ai.rho - rho_end;
      cfg.calc_limit += kDummyWeight * 10.0 * d * d;
      ai.rho = rho_end;
    } else if (ai.rho < rho_begin) {
      const double d = rho_begin - ai.rho;
      cfg.calc_limit += kDummyWeight * 10.0 * d * d;
      ai.rho = rho_begin;
    }
    cfg.calc_energy += embedding[ai].eval(ai.rho);
    ai.gradF = embedding[ai].deriv(ai.rho);
  }

  // ── Pass 2: pair + embedding-gradient forces ─────────────────────────────
  // Force on atom i from neighbor j:
  //   F_ij = [dφ_{ij}/dr + gradF_i × dg_{t(j)}/dr + gradF_j × dg_{t(i)}/dr] ×
  //   r̂_{ij}
  //
  // Using the full neighbor list each pair (i,j) appears twice, so:
  //   - pair energy:     add 0.5 × φ per entry
  //   - embedding force: the cross-gradient term (gradF_j × ...) is
  //     self-consistent because gradF_j was computed in pass 1
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      const auto &aj = *nb.neighbor;
      const double r = nb.dist.norm();
      if (r < 1e-14)
        continue;
      const double inv_r = 1.0 / r;

      // Gate each radial table on its own cutoff (see in_range): the neighbor
      // list spans the global max_cutoff(), so a shorter table would otherwise
      // extrapolate past its last knot here.
      const auto &phi_pot = pair[ai, aj];
      const auto &g_j = density[aj];
      const auto &g_i = density[ai];
      const double phi = in_range(phi_pot, r) ? phi_pot.eval(r) : 0.0;
      const double dphi = in_range(phi_pot, r) ? phi_pot.deriv(r) : 0.0;
      const double drho_j = in_range(g_j, r) ? g_j.deriv(r) : 0.0;
      const double drho_i = in_range(g_i, r) ? g_i.deriv(r) : 0.0;

      const double fscale =
          (dphi + ai.gradF * drho_j + aj.gradF * drho_i) * inv_r;
      const Vec3 fvec = fscale * nb.dist;

      ai.calc_force += fvec;
      cfg.calc_energy += 0.5 * phi;
      // Virial: bond ⊗ force-on-partner = dist ⊗ (−fvec); 0.5 for the full
      // list.
      cfg.calc_stress -= 0.5 * nb.dist * fvec.transpose();
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(
      events::ForceEvalStats{conf_index, force_rms(cfg), cfg});
}

} // namespace potfit
