#include "forcesmith/core/rescale.hpp"
#include "forcesmith/core/scale_potentials.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

namespace forcesmith {

std::vector<double> compute_rho_ref(EAMForceCalculator &calc,
                                    std::span<Configuration> configs) {
  const std::size_t n = calc.ntypes;
  std::for_each(configs.begin(), configs.end(),
                [&](auto &cfg) { calc.eval_forces(cfg); });

  std::vector<double> rho_sum(n, 0.0);
  std::vector<std::size_t> count(n, 0);
  std::ranges::for_each(configs, [&](const auto &cfg) {
    for (const auto &a : cfg.atoms) {
      if (a.type < n) {
        rho_sum[a.type] += a.rho;
        ++count[a.type];
      }
    }
  });

  std::vector<double> rho_ref(n, 0.0);
  for (auto [sum, cnt, ref] : std::views::zip(rho_sum, count, rho_ref)) {
    ref = cnt > 0 ? sum / cnt : 0.0;
  }
  return rho_ref;
}

double rescale_rho_axis(EAMForceCalculator &calc,
                        std::span<Configuration> configs) {
  const std::size_t n = calc.ntypes;

  // Density pass: fills each atom's a.rho via the EAM first pass.
  std::for_each(configs.begin(), configs.end(),
                [&](auto &cfg) { calc.eval_forces(cfg); });

  // Per-type min/max sampled electron density across all configs.
  constexpr double inf = std::numeric_limits<double>::infinity();
  std::vector<double> maxrho(n, -inf), minrho(n, inf);
  std::ranges::for_each(configs, [&](const auto &cfg) {
    for (const auto &a : cfg.atoms) {
      if (a.type < n) {
        maxrho[a.type] = std::max(maxrho[a.type], a.rho);
        minrho[a.type] = std::min(minrho[a.type], a.rho);
      }
    }
  });

  // Dominant type: the one with the largest |ρ| extent. `sign_pos` selects
  // whether the positive (max) or negative (min) side drives the scaling.
  std::size_t dom = 0;
  double best = -1.0;
  bool sign_pos = true;
  for (auto &&[t, pair] :
       std::views::zip(maxrho, minrho) | std::views::enumerate) {
    auto &&[max_rho, min_rho] = pair;

    if (!std::isfinite(max_rho)) {
      continue;
    }

    const double ext = std::max(std::abs(max_rho), std::abs(min_rho));
    if (ext > best) {
      best = ext;
      dom = t;
      sign_pos = (max_rho >= -min_rho);
    }
  }
  if (best < 0.0) {
    return 1.0; // no data
  }

  const auto [emb_lo, emb_hi] = calc.embedding[dom].span();
  const double pad = 0.003 * (emb_hi - emb_lo); // ≈ forcesmith's 0.3·step padding
  const double upper = sign_pos ? emb_hi : emb_lo;
  const double right = sign_pos ? maxrho[dom] + pad : minrho[dom] - pad;
  if (std::abs(right) < 1e-30) {
    return 1.0;
  }
  const double a = upper / right;

  // Domain violation: any sampled ρ falls outside its embedding span.
  const bool violation = std::ranges::any_of(
      std::views::iota(std::size_t{0}, n), [&](std::size_t t) {
        if (!std::isfinite(maxrho[t]))
          return false;

        const auto [lo, hi] = calc.embedding[t].span();
        return minrho[t] < lo || maxrho[t] > hi;
      });

  // forcesmith skip rule: only rescale when actually needed.
  if (!std::isfinite(a) || std::abs(a) < 1e-30 ||
      (!violation && std::abs(a) >= 0.95 && std::abs(a) <= 1.05)) {
    return 1.0;
  }

  // Apply one global factor: density × a, embedding argument / a.
  for (auto &&[density, embedding] :
       std::views::zip(calc.density, calc.embedding)) {
    density = Potential(ScaledOutputPotential{std::move(density), a});
    embedding = Potential(ScaledArgPotential{std::move(embedding), a});
  }
  return a;
}

void embed_shift(EAMForceCalculator &calc, std::span<const double> rho_ref) {
  const std::size_t n = calc.ntypes;

  // slope_t = F_t′(rho_ref_t)
  std::vector<double> slope(n, 0.0);
  for (auto [s, rho, emb] : std::views::zip(slope, rho_ref, calc.embedding) |
                                std::views::filter([](const auto &t) {
                                  return std::get<1>(t) > 0.0;
                                }))
    s = emb.deriv(rho);

  // Shift embedding: F_t(ρ) → F_t(ρ) − slope_t × ρ
  for (auto [t, s] : std::views::enumerate(slope)) {
    if (std::abs(s) < 1e-14) {
      continue;
    }
    auto &emb = calc.embedding[static_cast<int>(t)];
    emb = Potential(LinearAdjustedPotential{std::move(emb), s, 0.0});
  }

  // Compensate pairs: φ_{αβ}(r) → φ_{αβ}(r) + slope_α × g_β(r) + slope_β ×
  // g_α(r)
  for (auto [ti, tj] : calc.pair.indices()) {
    const double ca = slope[ti];
    const double cb = slope[tj];
    if (std::abs(ca) < 1e-14 && std::abs(cb) < 1e-14) {
      continue;
    }
    calc.pair[ti, tj] = Potential(CompensatedPairPotential{
        std::move(calc.pair[ti, tj]),
        calc.density[ti], // g_alpha: density contributed by type ti
        calc.density[tj], // g_beta:  density contributed by type tj
        ca, cb});
  }
}

void rescale_eam(EAMForceCalculator &calc, std::span<Configuration> configs) {
  // Step 0: density-axis stretch so sampled ρ fills the embedding table. Runs
  // first; compute_rho_ref below re-evaluates ρ on the stretched density.
  rescale_rho_axis(calc, configs);

  // embed_shift does not modify density functions, so rho_ref is stable across
  // both steps.
  const auto rho_ref = compute_rho_ref(calc, configs);

  // Step 1: gauge-invariant linear shift → F_t′(rho_ref) = 0
  embed_shift(calc, rho_ref);

  // Step 2: constant zero-shift → F_t(rho_ref) = 0
  for (auto [emb, rho] : std::views::zip(calc.embedding, rho_ref)) {
    if (rho <= 0.0) {
      continue;
    }
    const double F0 = emb.eval(rho);
    if (std::abs(F0) < 1e-14) {
      continue;
    }
    emb = Potential(LinearAdjustedPotential{std::move(emb), 0.0, F0});
  }
}

} // namespace forcesmith
