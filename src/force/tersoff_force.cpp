#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/events/signals.hpp"

#include <cmath>
#include <functional>
#include <numbers>
#include <optional>
#include <utility>

namespace forcesmith {

FORCE_INLINE std::array<Param *, 12>
TersoffForceCalculator::tersoff_fields(TersoffParams &p) {
  return std::array<Param *, 12>{&p.A,    &p.B, &p.lambda, &p.mu,
                                 &p.beta, &p.n, &p.c,      &p.d,
                                 &p.h,    &p.R, &p.S,      &p.omega};
}
FORCE_INLINE std::array<const Param *, 12>
TersoffForceCalculator::tersoff_fields(const TersoffParams &p) {
  return std::array<const Param *, 12>{&p.A,    &p.B, &p.lambda, &p.mu,
                                       &p.beta, &p.n, &p.c,      &p.d,
                                       &p.h,    &p.R, &p.S,      &p.omega};
}

FORCE_INLINE double TersoffForceCalculator::fc_val(double r, double R,
                                                   double S) noexcept {
  if (r <= R) {
    return 1.0;
  } else if (r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return 0.5 + 0.5 * std::cos(x);
}

FORCE_INLINE double TersoffForceCalculator::dfc_val(double r, double R,
                                                    double S) noexcept {
  if (r <= R || r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return -0.5 * std::numbers::pi / (S - R) * std::sin(x);
}

FORCE_INLINE double
TersoffForceCalculator::g_val(double c, const TersoffParams &p) noexcept {
  const double c2 = p.c * p.c;
  const double d2 = p.d * p.d;
  const double hc = p.h - c;
  return 1.0 + c2 / d2 - c2 / (d2 + hc * hc);
}

FORCE_INLINE double
TersoffForceCalculator::dg_val(double c, const TersoffParams &p) noexcept {
  const double c2 = p.c * p.c;
  const double d2 = p.d * p.d;
  const double hc = p.h - c;
  const double den = d2 + hc * hc;
  return -2.0 * c2 * hc / (den * den);
}

double TersoffForceCalculator::bond_order(double zeta,
                                          const TersoffParams &p) noexcept {
  if (zeta == 0.0) {
    return 1.0;
  }
  const double bz_n = std::pow(p.beta * zeta, p.n);
  return std::pow(1.0 + bz_n, -0.5 / p.n);
}

double TersoffForceCalculator::dbond_dzeta(double zeta,
                                           const TersoffParams &p) noexcept {
  if (zeta == 0.0) {
    return 0.0;
  }
  const double bz_n = std::pow(p.beta * zeta, p.n);
  const double b = std::pow(1.0 + bz_n, -0.5 / p.n);
  return -0.5 * b * bz_n / (zeta * (1.0 + bz_n));
}

std::optional<Bond> TersoffForceCalculator::make_bond(const TersoffParams &p,
                                                      const Vec3 &d1,
                                                      double r1) {
  if (r1 < 1e-14) {
    return std::nullopt;
  }
  const double fc = fc_val(r1, p.R, p.S);
  const double dfc = dfc_val(r1, p.R, p.S);
  if (fc == 0.0 && dfc == 0.0) {
    return std::nullopt;
  }
  const double VR = p.A * std::exp(-p.lambda * r1);
  const double VA = p.B * std::exp(-p.mu * r1);
  return Bond{&p, d1, r1, fc, dfc, VR, VA, -p.lambda * VR, -p.mu * VA};
}

std::optional<Bond> TersoffForceCalculator::add_zeta(const Atom &ai,
                                                     std::size_t jj,
                                                     Bond &&bond) const {
  const std::size_t nn = ai.neighbors.size();
  double zeta = 0.0;
  for (std::size_t kk = 0; kk < nn; ++kk) {
    if (kk == jj) {
      continue;
    }
    const NeighborEntry &nb_k = ai.neighbors[kk];
    const Vec3 &d2 = nb_k.dist;
    const double r2 = d2.norm();
    if (r2 < 1e-14) {
      continue;
    }
    const TersoffParams &p_ik = params[ai.type, nb_k.neighbor->type];
    const double fc_ik = fc_val(r2, p_ik.R, p_ik.S);
    if (fc_ik == 0.0) {
      continue;
    }
    const double cos_theta = bond.d1.dot(d2) / (bond.r1 * r2);
    zeta += p_ik.omega * fc_ik * g_val(cos_theta, *bond.p);
  }
  bond.zeta = zeta;
  return std::move(bond);
}

Bond TersoffForceCalculator::add_bond_order(Bond &&bond) {
  bond.b = bond_order(bond.zeta, *bond.p);
  return std::move(bond);
}

Bond TersoffForceCalculator::accumulate_pair(Atom &ai, std::size_t jj,
                                             Configuration &cfg, Bond &&bond) {
  Atom &aj = const_cast<Atom &>(*ai.neighbors[jj].neighbor);

  // Energy: (1/2) f_c [VR − b VA].
  cfg.calc_energy += 0.5 * bond.fc * (bond.VR - bond.b * bond.VA);

  // F_i += (1/2)[f_c'(VR − b VA) + f_c(VR' − b VA')] d1 / r1.
  const double pair_coeff = 0.5 *
                            (bond.dfc * (bond.VR - bond.b * bond.VA) +
                             bond.fc * (bond.VRp - bond.b * bond.VAp)) /
                            bond.r1;
  const Vec3 F_pair = pair_coeff * bond.d1;

  ai.calc_force += F_pair;
  aj.calc_force -= F_pair;
  cfg.calc_stress += 0.5 * bond.d1 * (-F_pair).transpose();
  return std::move(bond);
}

// Three-body force from ∂b_ij/∂ζ × ∂ζ/∂r_n. Empty when ζ = 0.
//   E = (1/2) f_c [VR − b VA] ⇒ ∂E/∂b = −(1/2) f_c VA, so
//   F_n = −∂E/∂b × db/dζ × ∂ζ/∂r_n = (1/2) f_c VA (db/dζ) ∂ζ/∂r_n ≡ P ∂ζ/∂r_n.
std::optional<Bond> TersoffForceCalculator::accumulate_three_body(
    Atom &ai, std::size_t jj, Configuration &cfg, Bond &&bond) const {
  if (bond.zeta == 0.0) {
    return std::nullopt;
  }
  Atom &aj = const_cast<Atom &>(*ai.neighbors[jj].neighbor);
  const double P = 0.5 * bond.fc * bond.VA * dbond_dzeta(bond.zeta, *bond.p);

  const Vec3 &d1 = bond.d1;
  const double inv_r1 = 1.0 / bond.r1;
  const std::size_t nn = ai.neighbors.size();
  for (std::size_t kk = 0; kk < nn; ++kk) {
    if (kk == jj) {
      continue;
    }
    const NeighborEntry &nb_k = ai.neighbors[kk];
    Atom &ak = const_cast<Atom &>(*nb_k.neighbor);
    const Vec3 &d2 = nb_k.dist;
    const double r2 = d2.norm();
    if (r2 < 1e-14) {
      continue;
    }
    const TersoffParams &p_ik = params[ai.type, nb_k.neighbor->type];
    const double fc_ik = fc_val(r2, p_ik.R, p_ik.S);
    const double dfc_ik = dfc_val(r2, p_ik.R, p_ik.S);
    if (fc_ik == 0.0 && dfc_ik == 0.0) {
      continue;
    }

    const double inv_r2 = 1.0 / r2;
    const double cos_theta = d1.dot(d2) * inv_r1 * inv_r2;
    const double gv = g_val(cos_theta, *bond.p);
    const double dgv = dg_val(cos_theta, *bond.p);

    // Gradient of cos θ w.r.t. each atom position:
    //   ∂c/∂r_i = Ac d1 + Bc d2   ∂c/∂r_j = d2/(r1 r2) − c d1/r1²
    //   ∂c/∂r_k = d1/(r1 r2) − c d2/r2²
    const double Ac = cos_theta * inv_r1 * inv_r1 - inv_r1 * inv_r2;
    const double Bc = cos_theta * inv_r2 * inv_r2 - inv_r1 * inv_r2;

    // Mixing weight ω for the i–k pair carries through every ζ-gradient term.
    const double w_ik = p_ik.omega;

    const Vec3 dz_dri =
        w_ik * (-dfc_ik * inv_r2 * gv * d2 + fc_ik * dgv * (Ac * d1 + Bc * d2));
    const Vec3 dz_drj =
        w_ik * fc_ik * dgv *
        (inv_r1 * inv_r2 * d2 - cos_theta * inv_r1 * inv_r1 * d1);
    const Vec3 dz_drk =
        w_ik * (dfc_ik * inv_r2 * gv * d2 +
                fc_ik * dgv *
                    (inv_r1 * inv_r2 * d1 - cos_theta * inv_r2 * inv_r2 * d2));

    const Vec3 Fi_3b = P * dz_dri;
    const Vec3 Fj_3b = P * dz_drj;
    const Vec3 Fk_3b = P * dz_drk;

    ai.calc_force += Fi_3b;
    aj.calc_force += Fj_3b;
    ak.calc_force += Fk_3b;

    // Virial: bond ⊗ force-on-partner (the 0.5 already lives in P).
    cfg.calc_stress += d1 * Fj_3b.transpose() + d2 * Fk_3b.transpose();
  }
  return std::move(bond);
}

std::size_t TersoffForceCalculator::param_count() const {
  return std::transform_reduce(params.begin(), params.end(), std::size_t{0},
                               std::plus<>{}, [](const auto &p) {
                                 return std::ranges::count_if(
                                     tersoff_fields(p),
                                     [](const Param *f) { return !f->fixed; });
                               });
}

void TersoffForceCalculator::gather_params(Eigen::VectorXd &dst,
                                           std::size_t off) const {
  for (const auto &p : params) {
    for (const Param *f : tersoff_fields(p)) {
      if (!f->fixed) {
        dst[off++] = f->value;
      }
    }
  }
}

void TersoffForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                            std::size_t off) {
  for (auto &p : params) {
    for (Param *f : tersoff_fields(p)) {
      if (!f->fixed) {
        f->value = src[off++];
      }
    }
  }
}

void TersoffForceCalculator::gather_bounds(Eigen::VectorXd &lo,
                                           Eigen::VectorXd &hi,
                                           std::size_t off) const {
  for (const auto &p : params) {
    for (const Param *f : tersoff_fields(p)) {
      if (!f->fixed) {
        lo[off] = f->min;
        hi[off] = f->max;
        ++off;
      }
    }
  }
}

double TersoffForceCalculator::max_cutoff() const {
  return std::transform_reduce(
      params.begin(), params.end(), 0.0,
      [](double a, double b) { return std::max(a, b); },
      [](const auto &p) { return p.S.value; });
}

void TersoffForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  std::for_each(cfg.atoms.begin(), cfg.atoms.end(),
                [](auto &a) { a.calc_force = Vec3::Zero(); });

  // For each i–j bond, run the pipeline:
  //   pair terms → ζ → bond order → energy + pair force → 3-body force
  // std::optional short-circuits bonds outside the cutoff (make_bond) or with
  // no angular neighbours (ζ = 0, accumulate_three_body), so each step reads as
  // one clear stage rather than a deeply nested loop body.
  std::ranges::for_each(cfg.atoms, [&](Atom &ai) {
    const std::size_t ti = ai.type;
    const std::size_t nn = ai.neighbors.size();

    for (std::size_t jj = 0; jj < nn; ++jj) {
      const NeighborEntry &nb_j = ai.neighbors[jj];
      const Vec3 &d1 = nb_j.dist;
      const TersoffParams &p = params[ti, nb_j.neighbor->type];

      make_bond(p, d1, d1.norm())
          .and_then(std::bind_front(&TersoffForceCalculator::add_zeta, this,
                                    std::cref(ai), jj))
          .transform(add_bond_order)
          .transform(std::bind_front(&TersoffForceCalculator::accumulate_pair,
                                     std::ref(ai), jj, std::ref(cfg)))
          .and_then(
              std::bind_front(&TersoffForceCalculator::accumulate_three_body,
                              this, std::ref(ai), jj, std::ref(cfg)));
    }
  });

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(
      events::ForceEvalStats{conf_index, force_rms(cfg), cfg});
}

} // namespace forcesmith
