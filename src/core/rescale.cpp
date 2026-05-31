#include "potfit/core/rescale.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

namespace potfit {

namespace {

// Wrapper: eval(x) = base.eval(x) − slope × x − intercept.
// gather/scatter delegate to base so optimizer sees unchanged parameter layout.
struct LinearAdjustedPotential {
  Potential base;
  double slope = 0.0;
  double intercept = 0.0;

  double eval(double x) const { return base.eval(x) - slope * x - intercept; }
  double deriv(double x) const { return base.deriv(x) - slope; }
  std::pair<double, double> span() const { return base.span(); }
  std::size_t param_count() const { return base.param_count(); }
  void gather_params(Eigen::VectorXd &v, int off) const {
    base.gather_params(v, off);
  }
  void scatter_params(const Eigen::VectorXd &v, int off) {
    base.scatter_params(v, off);
  }
};

// Wrapper: scales the OUTPUT by a — density g(r) → a·g(r). Used for the
// rho-axis stretch. gather/scatter delegate to base (knot params unchanged).
struct ScaledOutputPotential {
  Potential base;
  double a = 1.0;

  double eval(double r) const { return a * base.eval(r); }
  double deriv(double r) const { return a * base.deriv(r); }
  std::pair<double, double> span() const { return base.span(); }
  std::size_t param_count() const { return base.param_count(); }
  void gather_params(Eigen::VectorXd &v, int off) const {
    base.gather_params(v, off);
  }
  void scatter_params(const Eigen::VectorXd &v, int off) {
    base.scatter_params(v, off);
  }
};

// Wrapper: scales the ARGUMENT by 1/a — embedding F(ρ) → F(ρ/a), so the stretch
// is energy-preserving (F_new(a·ρ_old) = F_old(ρ_old)). The span scales by a.
struct ScaledArgPotential {
  Potential base;
  double a = 1.0;

  double eval(double rho) const { return base.eval(rho / a); }
  double deriv(double rho) const { return base.deriv(rho / a) / a; }
  std::pair<double, double> span() const {
    auto [lo, hi] = base.span();
    return {a * lo, a * hi};
  }
  std::size_t param_count() const { return base.param_count(); }
  void gather_params(Eigen::VectorXd &v, int off) const {
    base.gather_params(v, off);
  }
  void scatter_params(const Eigen::VectorXd &v, int off) {
    base.scatter_params(v, off);
  }
};

// Wrapper: eval(r) = phi.eval(r) + coeff_alpha × g_beta.eval(r) + coeff_beta ×
// g_alpha.eval(r). Implements the EAM gauge compensation for a linear F shift.
// gather/scatter delegate to phi; density copies are fixed at rescale time.
struct CompensatedPairPotential {
  Potential phi;
  Potential g_alpha;
  Potential g_beta;
  double coeff_alpha = 0.0;
  double coeff_beta = 0.0;

  constexpr double eval(double r) const {
    return phi.eval(r) + coeff_alpha * g_beta.eval(r) +
           coeff_beta * g_alpha.eval(r);
  }
  constexpr double deriv(double r) const {
    return phi.deriv(r) + coeff_alpha * g_beta.deriv(r) +
           coeff_beta * g_alpha.deriv(r);
  }
  constexpr std::pair<double, double> span() const { return phi.span(); }
  constexpr int param_count() const { return phi.param_count(); }
  void gather_params(Eigen::VectorXd &v, int off) const {
    phi.gather_params(v, off);
  }
  void scatter_params(const Eigen::VectorXd &v, int off) {
    phi.scatter_params(v, off);
  }
};

} // anonymous namespace

std::vector<double> compute_rho_ref(EAMForceCalculator &calc,
                                    std::span<Configuration> configs) {
  const std::size_t n = calc.ntypes;
  std::for_each(configs.begin(), configs.end(),
                [&](auto &cfg) { calc.eval_forces(cfg); });

  std::vector<double> rho_sum(n, 0.0);
  std::vector<std::size_t> count(n, 0);
  for (const auto &cfg : configs) {
    for (const auto &a : cfg.atoms) {
      if (a.type < n) {
        rho_sum[a.type] += a.rho;
        ++count[a.type];
      }
    }
  }

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
  for (const auto &cfg : configs) {
    for (const auto &a : cfg.atoms) {
      if (a.type < n) {
        maxrho[a.type] = std::max(maxrho[a.type], a.rho);
        minrho[a.type] = std::min(minrho[a.type], a.rho);
      }
    }
  }

  // Dominant type: the one with the largest |ρ| extent. `sign_pos` selects
  // whether the positive (max) or negative (min) side drives the scaling.
  std::size_t dom = 0;
  double best = -1.0;
  bool sign_pos = true;
  for (std::size_t t = 0; t < n; ++t) {
    if (!std::isfinite(maxrho[t])) {
      continue; // no atoms of this type
    }
    const double ext = std::max(std::abs(maxrho[t]), std::abs(minrho[t]));
    if (ext > best) {
      best = ext;
      dom = t;
      sign_pos = (maxrho[t] >= -minrho[t]);
    }
  }
  if (best < 0.0) {
    return 1.0; // no data
  }

  const auto [emb_lo, emb_hi] = calc.embedding[dom].span();
  const double pad = 0.003 * (emb_hi - emb_lo); // ≈ potfit's 0.3·step padding
  const double upper = sign_pos ? emb_hi : emb_lo;
  const double right = sign_pos ? maxrho[dom] + pad : minrho[dom] - pad;
  if (std::abs(right) < 1e-30) {
    return 1.0;
  }
  const double a = upper / right;

  // Domain violation: any sampled ρ falls outside its embedding span.
  bool violation = false;
  for (std::size_t t = 0; t < n; ++t) {
    if (!std::isfinite(maxrho[t])) {
      continue;
    }
    const auto [lo, hi] = calc.embedding[t].span();
    if (minrho[t] < lo || maxrho[t] > hi) {
      violation = true;
    }
  }

  // potfit skip rule: only rescale when actually needed.
  if (!std::isfinite(a) || std::abs(a) < 1e-30) {
    return 1.0;
  }
  if (!violation && std::abs(a) >= 0.95 && std::abs(a) <= 1.05) {
    return 1.0;
  }

  // Apply one global factor: density × a, embedding argument / a.
  for (std::size_t t = 0; t < n; ++t) {
    calc.density[static_cast<int>(t)] = Potential(
        ScaledOutputPotential{std::move(calc.density[static_cast<int>(t)]), a});
    calc.embedding[static_cast<int>(t)] = Potential(
        ScaledArgPotential{std::move(calc.embedding[static_cast<int>(t)]), a});
  }
  return a;
}

void embed_shift(EAMForceCalculator &calc, std::span<const double> rho_ref) {
  const int n = calc.ntypes;

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
  for (int ti = 0; ti < n; ++ti) {
    for (int tj = ti; tj < n; ++tj) {
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

} // namespace potfit
