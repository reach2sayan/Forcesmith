#pragma once

// Shared per-bond/per-triplet force arithmetic. Operand association matches
// the kernels these replaced, so forces stay bit-identical.

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/site_id.hpp"
#include "forcesmith/core/types.hpp"
#include "forcesmith/force/force_calculator_concept.hpp" // in_range, kDummyWeight

namespace forcesmith::force {

template <CEvaluable Pot>
[[nodiscard]] FORCE_INLINE SiteId prime_site(const Pot &p, double r) {
  return in_range(p, r) ? p.prepare_site(r) : SiteId{};
}

template <CEvaluable Emb>
FORCE_INLINE void clamp_embedding(Configuration &cfg, Atom &ai,
                                  const Emb &emb) {
  const auto [rho_begin, rho_end] = emb.span();
  if (ai.rho > rho_end) {
    const double d = ai.rho - rho_end;
    cfg.calc_limit += kDummyWeight * 10.0 * d * d;
    ai.rho = rho_end;
  } else if (ai.rho < rho_begin) {
    const double d = rho_begin - ai.rho;
    cfg.calc_limit += kDummyWeight * 10.0 * d * d;
    ai.rho = rho_begin;
  }
}

FORCE_INLINE void accumulate_pair(Configuration &cfg, Atom &ai, const Vec3 &d,
                                  const Vec3 &force, double energy) {
  ai.calc_force += force;
  cfg.calc_energy += 0.5 * energy;
  cfg.calc_stress -= 0.5 * d * force.transpose();
}

struct TripletForces {
  Vec3 fi, fj, fk;
};

[[nodiscard]] FORCE_INLINE TripletForces
three_body_forces(const Vec3 &d1, const Vec3 &d2, double inv_r1, double inv_r2,
                  double c, double f1, double df1, double f2, double df2,
                  double g, double dg) {
  const double inv_r1r2 = inv_r1 * inv_r2;
  const double A = c * inv_r1 * inv_r1 - inv_r1r2;
  const double B = c * inv_r2 * inv_r2 - inv_r1r2;

  return {.fi = (df1 * inv_r1 * f2 * g) * d1 + (f1 * df2 * inv_r2 * g) * d2 -
                (f1 * f2 * dg) * (A * d1 + B * d2),
          .fj = -(df1 * inv_r1 * f2 * g) * d1 -
                (f1 * f2 * dg) * (inv_r1r2 * d2 - c * inv_r1 * inv_r1 * d1),
          .fk = -(f1 * df2 * inv_r2 * g) * d2 -
                (f1 * f2 * dg) * (inv_r1r2 * d1 - c * inv_r2 * inv_r2 * d2)};
}

FORCE_INLINE void accumulate_triplet(Configuration &cfg, Atom &ai, Atom &aj,
                                     Atom &ak, const Vec3 &d1, const Vec3 &d2,
                                     const TripletForces &f) {
  ai.calc_force += f.fi;
  aj.calc_force += f.fj;
  ak.calc_force += f.fk;
  cfg.calc_stress += d1 * f.fj.transpose();
  cfg.calc_stress += d2 * f.fk.transpose();
}

} // namespace forcesmith::force
