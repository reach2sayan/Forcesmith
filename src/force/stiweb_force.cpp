#include "potfit/force/stiweb_force.hpp"
#include "potfit/events/signals.hpp"

#include <cmath>
#include <numeric>

namespace potfit {
namespace {

// ── SW 2-body: v2(r) = A [B(σ/r)^p − (σ/r)^q] exp(σ/(r − aσ)), r < aσ ──────
// Returns {v2, dv2/dr}; both zero for r ≥ aσ.
std::pair<double,double> v2_dv2(double r, const SWParams& p) noexcept {
    const double rcut = p.a * p.sigma;
    if (r >= rcut) return {0.0, 0.0};
    const double d    = r - rcut;               // d < 0
    const double e    = std::exp(p.sigma / d);
    if (e == 0.0) return {0.0, 0.0};           // underflow guard
    const double sr   = p.sigma / r;
    const double srp  = std::pow(sr, p.p);
    const double srq  = std::pow(sr, p.q);
    const double phi  = p.B * srp - srq;
    const double dphi = (-p.p * p.B * srp + p.q * srq) / r;
    const double v2   = p.A * phi * e;
    const double dv2  = p.A * (dphi * e + phi * e * (-p.sigma / (d * d)));
    return {v2, dv2};
}

// ── SW 3-body radial: h(r) = exp(γσ/(r − aσ)), r < aσ ───────────────────────
// Returns {h, dh/dr}; both zero for r ≥ aσ.
std::pair<double,double> h_dh(double r, const SWParams& p) noexcept {
    const double rcut = p.a * p.sigma;
    if (r >= rcut) return {0.0, 0.0};
    const double d   = r - rcut;                // d < 0
    const double h   = std::exp(p.gamma * p.sigma / d);
    if (h == 0.0) return {0.0, 0.0};           // underflow guard
    const double dh  = h * (-p.gamma * p.sigma / (d * d));
    return {h, dh};
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────

void StiwebForceCalculator::eval_forces(Configuration& cfg) const {
    cfg.calc_energy = 0.0;
    cfg.calc_stress = SymTens::Zero();
    for (auto& a : cfg.atoms) a.calc_force = Vec3::Zero();

    // ── 2-body loop ──────────────────────────────────────────────────────────
    // Full neighbor list: each pair counted twice, factor 0.5 per entry.
    for (auto& ai : cfg.atoms) {
        for (const auto& nb : ai.neighbors) {
            const Atom& aj = *nb.neighbor;
            const double r = nb.dist.norm();
            if (r < 1e-14) continue;

            const auto& p          = params(ai.type, aj.type);
            const auto [v2, dv2]   = v2_dv2(r, p);
            const Vec3 fvec        = (dv2 / r) * nb.dist;

            ai.calc_force   += fvec;
            cfg.calc_energy += 0.5 * v2;
            cfg.calc_stress += 0.5 * nb.dist * fvec.transpose();
        }
    }

    // ── 3-body loop ──────────────────────────────────────────────────────────
    // For each central atom i, iterate ordered pairs (j < k) of its neighbors.
    // E_ijk = h(r_ij) × h(r_ik) × w(cos θ_jik)
    // w(c)  = λ (c + 1/3)²,  w'(c) = 2λ (c + 1/3)
    //
    // Force gradient derivation (d1 = r_j−r_i, d2 = r_k−r_i, c = cosθ):
    //   Ac = c/r1² − 1/(r1 r2),  Bc = c/r2² − 1/(r1 r2)
    //
    //   F_i = (dh1/r1 · h2 · w) d1 + (h1 · dh2/r2 · w) d2 − h1·h2·w' (Ac d1 + Bc d2)
    //   F_j = −(dh1/r1 · h2 · w) d1 − h1·h2·w' (d2/(r1r2) − c·d1/r1²)
    //   F_k = −(h1 · dh2/r2 · w) d2 − h1·h2·w' (d1/(r1r2) − c·d2/r2²)
    for (auto& ai : cfg.atoms) {
        const auto& nbs = ai.neighbors;
        const int   nn  = static_cast<int>(nbs.size());
        const int   ti  = ai.type;

        for (int jj = 0; jj < nn; ++jj) {
            const auto& nb_j       = nbs[jj];
            const Vec3& d1         = nb_j.dist;
            const double r1        = d1.norm();
            if (r1 < 1e-14) continue;

            const int tj           = nb_j.neighbor->type;
            const auto& p_ij       = params(ti, tj);
            const auto [h1, dh1]   = h_dh(r1, p_ij);
            if (h1 == 0.0 && dh1 == 0.0) continue;
            const double inv_r1    = 1.0 / r1;

            for (int kk = jj + 1; kk < nn; ++kk) {
                const auto& nb_k       = nbs[kk];
                const Vec3& d2         = nb_k.dist;
                const double r2        = d2.norm();
                if (r2 < 1e-14) continue;

                const int tk           = nb_k.neighbor->type;
                const auto& p_ik       = params(ti, tk);
                const auto [h2, dh2]   = h_dh(r2, p_ik);
                if (h2 == 0.0 && dh2 == 0.0) continue;
                const double inv_r2    = 1.0 / r2;

                const double c         = d1.dot(d2) * inv_r1 * inv_r2;
                const double cp13      = c + 1.0 / 3.0;  // c + 1/3
                const double lambda    = p_ij.lambda;
                const double w         = lambda * cp13 * cp13;
                const double dw        = 2.0 * lambda * cp13;

                cfg.calc_energy += h1 * h2 * w;

                const double inv_r1r2  = inv_r1 * inv_r2;
                const double Ac        = c * inv_r1 * inv_r1 - inv_r1r2;
                const double Bc        = c * inv_r2 * inv_r2 - inv_r1r2;

                const Vec3 fi = (dh1 * inv_r1 * h2 * w) * d1
                              + (h1 * dh2 * inv_r2 * w) * d2
                              - (h1 * h2 * dw) * (Ac * d1 + Bc * d2);

                const Vec3 fj = -(dh1 * inv_r1 * h2 * w) * d1
                              - (h1 * h2 * dw) *
                                (inv_r1r2 * d2 - c * inv_r1 * inv_r1 * d1);

                const Vec3 fk = -(h1 * dh2 * inv_r2 * w) * d2
                              - (h1 * h2 * dw) *
                                (inv_r1r2 * d1 - c * inv_r2 * inv_r2 * d2);

                ai.calc_force                              += fi;
                const_cast<Atom&>(*nb_j.neighbor).calc_force += fj;
                const_cast<Atom&>(*nb_k.neighbor).calc_force += fk;

                cfg.calc_stress += d1 * (-fj).transpose();
                cfg.calc_stress += d2 * (-fk).transpose();
            }
        }
    }

    const double rms2 = std::transform_reduce(
        cfg.atoms.begin(), cfg.atoms.end(), 0.0, std::plus<>{},
        [](const auto& a) { return a.calc_force.squaredNorm(); });
    const double rms = cfg.atoms.empty()
        ? 0.0
        : std::sqrt(rms2 / static_cast<double>(cfg.atoms.size()));

    events::on_force_eval(events::ForceEvalStats{conf_index, rms});
}

} // namespace potfit
