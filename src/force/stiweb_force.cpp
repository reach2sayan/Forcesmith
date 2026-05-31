#include "potfit/force/stiweb_force.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/events/signals.hpp"

#include <cmath>
#include <numeric>
#include <optional>
#include <utility>

namespace potfit {
namespace {

auto sw_fields(SWParams &p) {
  return std::array<Param *, 8>{&p.A,     &p.B,  &p.p,     &p.q,
                                &p.delta, &p.a1, &p.gamma, &p.a2};
}
auto sw_fields(const SWParams &p) {
  return std::array<const Param *, 8>{&p.A,     &p.B,  &p.p,     &p.q,
                                      &p.delta, &p.a1, &p.gamma, &p.a2};
}

// ── SW 2-body: v2(r) = (A·r^{−p} − B·r^{−q}) exp(δ/(r − a1)), r < a1 ─────────
// Matches potfit stiweb_2_value. Returns {v2, dv2/dr}; both zero for r ≥ a1.
std::pair<double, double> v2_dv2(double r, const SWParams &p) noexcept {
  if (r >= p.a1) {
    return {0.0, 0.0};
  }
  const double d = r - p.a1; // d < 0
  const double e = std::exp(p.delta / d);
  if (e == 0.0) {
    return {0.0, 0.0}; // underflow guard
  }
  const double rp = std::pow(r, -p.p);
  const double rq = std::pow(r, -p.q);
  const double poly = p.A * rp - p.B * rq;
  const double dpoly = -p.A * p.p * std::pow(r, -p.p - 1.0) +
                       p.B * p.q * std::pow(r, -p.q - 1.0);
  const double v2 = poly * e;
  const double dv2 = dpoly * e + poly * e * (-p.delta / (d * d));
  return {v2, dv2};
}

// ── SW 3-body radial: h(r) = exp(γ/(r − a2)), r < a2 ────────────────────────
// Matches potfit stiweb_3_value. Returns {h, dh/dr}; both zero for r ≥ a2.
std::pair<double, double> h_dh(double r, const SWParams &p) noexcept {
  if (r >= p.a2) {
    return {0.0, 0.0};
  }
  const double d = r - p.a2; // d < 0
  const double h = std::exp(p.gamma / d);
  if (h == 0.0) {
    return {0.0, 0.0}; // underflow guard
  }
  const double dh = h * (-p.gamma / (d * d));
  return {h, dh};
}

// ── 2-body pipeline ─────────────────────────────────────────────────────────
// Each i–j bond is threaded through the stages; an empty std::optional (atoms
// coincident) short-circuits the chain, mirroring adp/tersoff.
struct SWPair {
  Atom *ai;                  // central atom
  const SWParams *p;         // i–j pair parameters
  Vec3 d;                    // pos_j − pos_i
  double r;                  // |d|
  double v2 = 0.0;           // pair energy V₂(r)
  Vec3 force = Vec3::Zero(); // force on i
};

// Stage 1 — geometry. Empty for coincident atoms.
std::optional<SWPair> make_sw_pair(Atom &ai, const NeighborEntry &nb,
                                   const SWParams &p) {
  const double r = nb.dist.norm();
  if (r < 1e-14) {
    return std::nullopt;
  }
  return SWPair{&ai, &p, nb.dist, r};
}

// Stage 2 — radial force V₂′(r).
SWPair add_pair_force(SWPair &&pf) {
  const auto [v2, dv2] = v2_dv2(pf.r, *pf.p);
  pf.v2 = v2;
  pf.force = (dv2 / pf.r) * pf.d;
  return std::move(pf);
}

// Stage 3 — commit energy / force / virial (0.5 for the full neighbor list).
auto accumulate_pair(Configuration &cfg) {
  return [&cfg](SWPair &&pf) -> SWPair {
    pf.ai->calc_force += pf.force;
    cfg.calc_energy += 0.5 * pf.v2;
    // Virial: bond ⊗ force-on-partner = dist ⊗ (−force); 0.5 for the full list.
    cfg.calc_stress -= 0.5 * pf.d * pf.force.transpose();
    return std::move(pf);
  };
}

// ── 3-body pipeline ─────────────────────────────────────────────────────────
// Per-(i,j) bond: h(r₁), h′(r₁) are built once and reused for every k. An empty
// std::optional short-circuits the chain (atoms coincident or h underflows).
struct SWJBond {
  Atom *ai;           // central atom
  const Atom *aj;     // neighbor j
  std::size_t ti, tj; // atom types
  Vec3 d1;            // pos_j − pos_i
  double r1, inv_r1;
  double h1, dh1;
};

struct SWTriplet {
  SWJBond jb;     // the i–j bond
  const Atom *ak; // neighbor k
  Vec3 d2;        // pos_k − pos_i
  double r2, inv_r2;
  double h2, dh2;
  double lambda;                     // per-triplet λ[i][j][k]
  double c = 0.0, w = 0.0, dw = 0.0; // cosθ and angular term
};

// Build the i–j bond. Empty when i,j coincide or h(r₁) underflows.
std::optional<SWJBond> make_jbond(Atom &ai, const NeighborEntry &nb_j,
                                  const SWParams &p_ij) {
  const Vec3 &d1 = nb_j.dist;
  const double r1 = d1.norm();
  if (r1 < 1e-14) {
    return std::nullopt;
  }
  const auto [h1, dh1] = h_dh(r1, p_ij);
  if (h1 == 0.0 && dh1 == 0.0) {
    return std::nullopt;
  }
  return SWJBond{
      &ai, nb_j.neighbor, ai.type, nb_j.neighbor->type, d1, r1, 1.0 / r1, h1,
      dh1};
}

// Build the (i, j, k) triplet from a bond. Empty when k coincides with i or
// h(r₂) underflows.
std::optional<SWTriplet> make_triplet(const SWJBond &jb,
                                      const NeighborEntry &nb_k,
                                      const SWParams &p_ik, double lambda) {
  const Vec3 &d2 = nb_k.dist;
  const double r2 = d2.norm();
  if (r2 < 1e-14) {
    return std::nullopt;
  }
  const auto [h2, dh2] = h_dh(r2, p_ik);
  if (h2 == 0.0 && dh2 == 0.0) {
    return std::nullopt;
  }
  return SWTriplet{jb, nb_k.neighbor, d2, r2, 1.0 / r2, h2, dh2, lambda};
}

// Stage — angular term w(c) = λ (c + 1/3)²,  w'(c) = 2λ (c + 1/3).
SWTriplet add_angular(SWTriplet &&t) {
  t.c = t.jb.d1.dot(t.d2) * t.jb.inv_r1 * t.inv_r2;
  const double cp13 = t.c + 1.0 / 3.0; // c + 1/3
  t.w = t.lambda * cp13 * cp13;
  t.dw = 2.0 * t.lambda * cp13;
  return std::move(t);
}

// Stage — commit triplet energy, the three force vectors, and virial.
//
// Force gradient derivation (d1 = r_j−r_i, d2 = r_k−r_i, c = cosθ):
//   Ac = c/r1² − 1/(r1 r2),  Bc = c/r2² − 1/(r1 r2)
//
//   F_i = (dh1/r1 · h2 · w) d1 + (h1 · dh2/r2 · w) d2 − h1·h2·w' (Ac d1 + Bc
//   d2) F_j = −(dh1/r1 · h2 · w) d1 − h1·h2·w' (d2/(r1r2) − c·d1/r1²) F_k =
//   −(h1 · dh2/r2 · w) d2 − h1·h2·w' (d1/(r1r2) − c·d2/r2²)
auto accumulate_three_body(Configuration &cfg) {
  return [&cfg](SWTriplet &&t) -> SWTriplet {
    const SWJBond &jb = t.jb;
    const Vec3 &d1 = jb.d1;
    const Vec3 &d2 = t.d2;
    const double h1 = jb.h1, dh1 = jb.dh1;
    const double h2 = t.h2, dh2 = t.dh2;
    const double inv_r1 = jb.inv_r1, inv_r2 = t.inv_r2;
    const double c = t.c, w = t.w, dw = t.dw;

    cfg.calc_energy += h1 * h2 * w;

    const double inv_r1r2 = inv_r1 * inv_r2;
    const double Ac = c * inv_r1 * inv_r1 - inv_r1r2;
    const double Bc = c * inv_r2 * inv_r2 - inv_r1r2;

    const Vec3 fi = (dh1 * inv_r1 * h2 * w) * d1 +
                    (h1 * dh2 * inv_r2 * w) * d2 -
                    (h1 * h2 * dw) * (Ac * d1 + Bc * d2);

    const Vec3 fj = -(dh1 * inv_r1 * h2 * w) * d1 -
                    (h1 * h2 * dw) * (inv_r1r2 * d2 - c * inv_r1 * inv_r1 * d1);

    const Vec3 fk = -(h1 * dh2 * inv_r2 * w) * d2 -
                    (h1 * h2 * dw) * (inv_r1r2 * d1 - c * inv_r2 * inv_r2 * d2);

    jb.ai->calc_force += fi;
    const_cast<Atom &>(*jb.aj).calc_force += fj;
    const_cast<Atom &>(*t.ak).calc_force += fk;

    // Virial: bond ⊗ force-on-partner (fj, fk are applied to atoms j, k).
    cfg.calc_stress += d1 * fj.transpose();
    cfg.calc_stress += d2 * fk.transpose();
    return std::move(t);
  };
}

} // anonymous namespace

std::size_t StiwebForceCalculator::param_count() const {
  auto count = std::transform_reduce(
      params.begin(), params.end(), std::size_t{0}, std::plus<>{},
      [](const auto &p) {
        return std::ranges::count_if(sw_fields(p),
                                     [](const Param *f) { return !f->fixed; });
      });
  count +=
      std::ranges::count_if(lambda, [](const auto &l) { return !l.fixed; });
  return count;
}

void StiwebForceCalculator::gather_params(Eigen::VectorXd &dst,
                                          std::size_t off) const {
  for (const auto &p : params) {
    for (const Param *f : sw_fields(p)) {
      if (!f->fixed) {
        dst[off++] = f->value;
      }
    }
  }
  for (const auto &l : lambda) {
    if (!l.fixed) {
      dst[off++] = l.value;
    }
  }
}

void StiwebForceCalculator::scatter_params(const Eigen::VectorXd &src,
                                           std::size_t off) {
  for (auto &p : params) {
    for (Param *f : sw_fields(p)) {
      if (!f->fixed) {
        f->value = src[off++];
      }
    }
  }
  for (auto &l : lambda) {
    if (!l.fixed) {
      l.value = src[off++];
    }
  }
}

double StiwebForceCalculator::max_cutoff() const {
  return std::transform_reduce(
      params.begin(), params.end(), 0.0,
      [](double a, double b) { return std::max(a, b); },
      [](const auto &p) { return std::max(p.a1.value, p.a2.value); });
}

void StiwebForceCalculator::eval_forces(Configuration &cfg) const {
  build_neighbor_list(cfg, max_cutoff());

  cfg.calc_energy = 0.0;
  cfg.calc_stress = SymTens::Zero();
  std::ranges::for_each(cfg.atoms,
                        [](auto &a) { a.calc_force = Vec3::Zero(); });

  // ── 2-body loop ──────────────────────────────────────────────────────────
  // Full neighbor list: each pair counted twice, factor 0.5 per entry. Each
  // i–j bond flows: geometry → radial force → commit.
  for (auto &ai : cfg.atoms) {
    for (const auto &nb : ai.neighbors) {
      make_sw_pair(ai, nb, params[ai.type, nb.neighbor->type])
          .transform(add_pair_force)
          .transform(accumulate_pair(cfg));
    }
  }

  // ── 3-body loop ──────────────────────────────────────────────────────────
  // For each central atom i, iterate ordered pairs (j < k) of its neighbors.
  // E_ijk = h(r_ij) × h(r_ik) × w(cos θ_jik). The i–j bond (h₁, h₁′) is built
  // once per j and reused for every k; an empty bond short-circuits the inner
  // loop. Each triplet then flows: angular term → commit.
  for (auto &ai : cfg.atoms) {
    const auto &nbs = ai.neighbors;
    const std::size_t nn = nbs.size();
    const std::size_t ti = ai.type;

    for (std::size_t jj = 0; jj < nn; ++jj) {
      auto jb = make_jbond(ai, nbs[jj], params[ti, nbs[jj].neighbor->type]);
      if (!jb) {
        continue;
      }

      for (std::size_t kk = jj + 1; kk < nn; ++kk) {
        const std::size_t tk = nbs[kk].neighbor->type;
        make_triplet(*jb, nbs[kk], params[ti, tk], lambda_at(ti, jb->tj, tk))
            .transform(add_angular)
            .transform(accumulate_three_body(cfg));
      }
    }
  }

  cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
  events::on_force_eval(events::ForceEvalStats{conf_index, force_rms(cfg), cfg});
}

} // namespace potfit
