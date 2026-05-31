#include "potfit/force/tersoff_force.hpp"
#include "potfit/events/signals.hpp"

#include <cmath>
#include <numbers>
#include <numeric>

namespace potfit {
namespace {

auto tersoff_fields(TersoffParams& p) {
  return std::array<Param*, 11>{&p.A, &p.B, &p.lambda, &p.mu, &p.beta,
                                 &p.n, &p.c, &p.d,      &p.h, &p.R, &p.S};
}
auto tersoff_fields(const TersoffParams& p) {
  return std::array<const Param*, 11>{&p.A, &p.B, &p.lambda, &p.mu, &p.beta,
                                       &p.n, &p.c, &p.d,      &p.h, &p.R, &p.S};
}

constexpr double fc_val(double r, double R, double S) noexcept {
  if (r <= R) {
    return 1.0;
  } else if (r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return 0.5 + 0.5 * std::cos(x);
}

constexpr double dfc_val(double r, double R, double S) noexcept {
  if (r <= R || r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return -0.5 * std::numbers::pi / (S - R) * std::sin(x);
}

// ── Angular function g(cos θ) = 1 + c²/d² − c²/[d² + (h − cos θ)²] ──────────

constexpr double g_val(double c, const TersoffParams &p) noexcept {
  const double c2 = p.c * p.c;
  const double d2 = p.d * p.d;
  const double hc = p.h - c;
  return 1.0 + c2 / d2 - c2 / (d2 + hc * hc);
}

// dg/d(cos θ)
constexpr double dg_val(double c, const TersoffParams &p) noexcept {
  const double c2 = p.c * p.c;
  const double d2 = p.d * p.d;
  const double hc = p.h - c;
  const double den = d2 + hc * hc;
  return -2.0 * c2 * hc / (den * den);
}

// ── Bond order b_ij = (1 + (β ζ)^n)^{−1/(2n)} ───────────────────────────────

constexpr double bond_order(double zeta, const TersoffParams &p) noexcept {
  if (zeta == 0.0) {
    return 1.0;
  }
  const double bz_n = std::pow(p.beta * zeta, p.n);
  return std::pow(1.0 + bz_n, -0.5 / p.n);
}

// db/dζ = −b × (β ζ)^n / [2 ζ (1 + (β ζ)^n)]
constexpr double dbond_dzeta(double zeta, const TersoffParams &p) noexcept {
  if (zeta == 0.0) {
    return 0.0;
  }
  const double bz_n = std::pow(p.beta * zeta, p.n);
  const double b = std::pow(1.0 + bz_n, -0.5 / p.n);
  return -0.5 * b * bz_n / (zeta * (1.0 + bz_n));
}

} // anonymous namespace

std::size_t TersoffForceCalculator::param_count() const {
  std::size_t count = 0;
  for (const auto& p : params)
    for (const Param* f : tersoff_fields(p))
      if (!f->fixed) ++count;
  return count;
}

void TersoffForceCalculator::gather_params(Eigen::VectorXd& dst, std::size_t off) const {
  for (const auto& p : params)
    for (const Param* f : tersoff_fields(p))
      if (!f->fixed) dst[off++] = f->value;
}

void TersoffForceCalculator::scatter_params(const Eigen::VectorXd& src, std::size_t off) {
  for (auto& p : params)
    for (Param* f : tersoff_fields(p))
      if (!f->fixed) f->value = src[off++];
}

double TersoffForceCalculator::max_cutoff() const {
  double rcut = 0.0;
  for (const auto& p : params)
    rcut = std::max(rcut, p.S.value);
  return rcut;
}

void TersoffForceCalculator::eval_forces(Configuration &cfg) const {
  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  std::for_each(cfg.atoms.begin(), cfg.atoms.end(),
                [](auto &a) { a.calc_force = Vec3::Zero(); });

  const std::size_t natoms = cfg.atoms.size();
  for (std::size_t ii = 0; ii < natoms; ++ii) {
    Atom &ai = cfg.atoms[ii];
    const std::size_t ti = ai.type;
    const std::size_t nn = ai.neighbors.size();

    for (std::size_t jj = 0; jj < nn; ++jj) {
      const NeighborEntry &nb_j = ai.neighbors[jj];
      Atom &aj = const_cast<Atom &>(*nb_j.neighbor);
      const Vec3 &d1 = nb_j.dist; // pos_j − pos_i
      const double r1 = d1.norm();
      if (r1 < 1e-14)
        continue;

      const std::size_t tj = aj.type;
      const TersoffParams &p = params[ti, tj];

      const double fc_ij = fc_val(r1, p.R, p.S);
      const double dfc_ij = dfc_val(r1, p.R, p.S);
      if (fc_ij == 0.0 && dfc_ij == 0.0) {
        continue;
      }

      const double VR = p.A * std::exp(-p.lambda * r1);
      const double VA = p.B * std::exp(-p.mu * r1);
      const double VRp = -p.lambda * VR; // dVR/dr
      const double VAp = -p.mu * VA;     // dVA/dr

      // ── Compute ζ_ij ─────────────────────────────────────────────────
      double zeta = 0.0;
      for (std::size_t kk = 0; kk < nn; ++kk) {
        if (kk == jj)
          continue;
        const NeighborEntry &nb_k = ai.neighbors[kk];
        const Vec3 &d2 = nb_k.dist;
        const double r2 = d2.norm();
        if (r2 < 1e-14) {
          continue;
        }

        const std::size_t tk = nb_k.neighbor->type;
        const TersoffParams &p_ik = params[ti, tk];
        const double fc_ik = fc_val(r2, p_ik.R, p_ik.S);
        if (fc_ik == 0.0) {
          continue;
        }

        const double cos_theta = d1.dot(d2) / (r1 * r2);
        zeta += fc_ik * g_val(cos_theta, p);
      }

      const double b_ij = bond_order(zeta, p);

      // ── Energy: (1/2) f_c [VR − b_ij VA] ────────────────────────────
      cfg.calc_energy += 0.5 * fc_ij * (VR - b_ij * VA);

      // ── Pair forces (b_ij treated as fixed) ──────────────────────────
      // F_i += (1/2)[f_c'(VR − b VA) + f_c(VR' − b VA')] r̂_ij
      //      = (1/2)[...] / r1 × d1
      const double pair_coeff =
          0.5 * (dfc_ij * (VR - b_ij * VA) + fc_ij * (VRp - b_ij * VAp)) / r1;
      const Vec3 F_pair = pair_coeff * d1;

      ai.calc_force += F_pair;
      aj.calc_force -= F_pair;
      cfg.calc_stress += 0.5 * d1 * (-F_pair).transpose();

      // ── 3-body forces from ∂b_ij/∂ζ × ∂ζ/∂r_n ──────────────────────
      // F_n (3b) = P × ∂ζ/∂r_n   where P = 0.5 f_c VA db/dζ
      //
      // Note: E = (1/2) f_c [VR − b VA] so ∂E/∂b = −(1/2) f_c VA
      //       F_n = −∂E/∂b × db/dζ × ∂ζ/∂r_n = (1/2) f_c VA db/dζ × ∂ζ/∂r_n
      if (zeta == 0.0) {
        continue;
      }

      const double db_dz = dbond_dzeta(zeta, p);
      const double P = 0.5 * fc_ij * VA * db_dz; // < 0 (db/dz < 0, VA > 0)

      const double inv_r1 = 1.0 / r1;
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

        const std::size_t tk = nb_k.neighbor->type;
        const TersoffParams &p_ik = params[ti, tk];
        const double fc_ik = fc_val(r2, p_ik.R, p_ik.S);
        const double dfc_ik = dfc_val(r2, p_ik.R, p_ik.S);
        if (fc_ik == 0.0 && dfc_ik == 0.0) {
          continue;
        }

        const double inv_r2 = 1.0 / r2;
        const double cos_theta = d1.dot(d2) * inv_r1 * inv_r2;
        const double gv = g_val(cos_theta, p);
        const double dgv = dg_val(cos_theta, p);

        // Gradient of cos θ w.r.t. each atom position:
        //   ∂c/∂r_i = A d1 + B d2   (A = c/r1² − 1/(r1 r2), same for B)
        //   ∂c/∂r_j = d2/(r1 r2) − c d1/r1²
        //   ∂c/∂r_k = d1/(r1 r2) − c d2/r2²
        const double Ac = cos_theta * inv_r1 * inv_r1 - inv_r1 * inv_r2;
        const double Bc = cos_theta * inv_r2 * inv_r2 - inv_r1 * inv_r2;

        // ∂ζ/∂r_i = g × dfc_ik × (−d2/r2) + fc_ik × dgv × (Ac d1 + Bc d2)
        const Vec3 dz_dri =
            -dfc_ik * inv_r2 * gv * d2 + fc_ik * dgv * (Ac * d1 + Bc * d2);

        // ∂ζ/∂r_j = fc_ik × dgv × (d2/(r1 r2) − c d1/r1²)
        const Vec3 dz_drj =
            fc_ik * dgv *
            (inv_r1 * inv_r2 * d2 - cos_theta * inv_r1 * inv_r1 * d1);

        // ∂ζ/∂r_k = g × dfc_ik × (d2/r2) + fc_ik × dgv × (d1/(r1 r2) − c
        // d2/r2²)
        const Vec3 dz_drk =
            dfc_ik * inv_r2 * gv * d2 +
            fc_ik * dgv *
                (inv_r1 * inv_r2 * d1 - cos_theta * inv_r2 * inv_r2 * d2);

        const Vec3 Fi_3b = P * dz_dri;
        const Vec3 Fj_3b = P * dz_drj;
        const Vec3 Fk_3b = P * dz_drk;

        ai.calc_force += Fi_3b;
        aj.calc_force += Fj_3b;
        ak.calc_force += Fk_3b;

        // Virial: bond-force outer products for the angular neighbors j, k
        // (factor 0.5 cancels the double counting from the full pair sum)
        cfg.calc_stress +=
            0.5 * (d1 * (-Fj_3b).transpose() + d2 * (-Fk_3b).transpose());
      }
    }
  }

  const double rms2 = std::transform_reduce(
      cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{},
      [](const auto &a) { return a.calc_force.squaredNorm(); });
  const double rms =
      cfg.atoms.empty()
          ? 0.0
          : std::sqrt(rms2 / static_cast<double>(cfg.atoms.size()));

  events::on_force_eval(events::ForceEvalStats{conf_index, rms});
}

} // namespace potfit
