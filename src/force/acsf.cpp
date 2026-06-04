#include "potfit/potentials/acsf.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace potfit {

namespace {

// Cosine cutoff and its derivative; both vanish for r >= rcut.
inline std::pair<double, double> cutoff(double r, double rcut) {
  if (r >= rcut) {
    return {0.0, 0.0};
  }
  const double x = std::numbers::pi * r / rcut;
  return {0.5 * (1.0 + std::cos(x)),
          -0.5 * std::numbers::pi / rcut * std::sin(x)};
}

// Position in the upper_triangle(S) enumeration of the unordered pair {a,b}
// (row-major, 0 <= lo <= hi < S) — matches SOAP's species-pair ordering.
inline std::size_t pair_ordinal(std::size_t a, std::size_t b, std::size_t S) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return lo * S - lo * (lo - 1) / 2 + (hi - lo);
}

} // namespace

DescriptorValue ACSF::get_descriptor(const Atom &a) const {
  const std::size_t S = ntypes;
  const std::size_t P = S * (S + 1) / 2;
  const std::size_t nG1 = g1, nG2 = radial.size(), nG3 = g3.size();
  const std::size_t nG4 = g4.size(), nG5 = g5.size();

  // Flat block offsets: [g1 per s][g2 per s][g3 per s][g4 per pair][g5 per pair]
  const std::size_t base_g1 = 0;
  const std::size_t base_g2 = base_g1 + S * nG1;
  const std::size_t base_g3 = base_g2 + S * nG2;
  const std::size_t base_g4 = base_g3 + S * nG3;
  const std::size_t base_g5 = base_g4 + P * nG4;
  const auto total = static_cast<Eigen::Index>(base_g5 + P * nG5);

  const std::size_t nn = a.neighbors.size();

  DescriptorValue out;
  out.values = Eigen::VectorXd::Zero(total);
  out.has_grad = true;
  out.grad_self = DescriptorGrad::Zero(total, 3);
  out.grad_neigh.assign(nn, DescriptorGrad::Zero(total, 3));

  // Precompute per-neighbour geometry once.
  struct NB {
    Vec3 d;       // bond vector r_j − r_i
    Vec3 rhat;    // d / r
    double r = 0; // |d|
    double fc = 0, fcp = 0;
    std::size_t s = 0; // neighbour species
    bool ok = false;
  };
  std::vector<NB> nb(nn);
  for (std::size_t j = 0; j < nn; ++j) {
    const Vec3 &d = a.neighbors[j].dist;
    const double r = d.norm();
    const auto si = static_cast<long long>(a.neighbors[j].neighbor->type.index);
    if (r < 1e-14 || r >= rcut || si < 0 ||
        static_cast<std::size_t>(si) >= S) {
      continue;
    }
    auto [fc, fcp] = cutoff(r, rcut);
    nb[j] = NB{d, d / r, r, fc, fcp, static_cast<std::size_t>(si), true};
  }

  // ---- Radial families: G1, G2, G3 (per neighbour, per species channel) ----
  for (std::size_t j = 0; j < nn; ++j) {
    if (!nb[j].ok) {
      continue;
    }
    const double r = nb[j].r;
    const double fc = nb[j].fc, fcp = nb[j].fcp;
    const Eigen::RowVector3d rhat = nb[j].rhat.transpose();
    auto &gneigh = out.grad_neigh[j];

    auto add = [&](Eigen::Index idx, double g, double dg) {
      out.values[idx] += g;
      gneigh.row(idx) += dg * rhat;        // dD/dr_j
      out.grad_self.row(idx) -= dg * rhat; // dD/dr_i
    };

    // G1: Σ f_c
    for (std::size_t t = 0; t < nG1; ++t) {
      const auto idx = static_cast<Eigen::Index>(base_g1 + nb[j].s * nG1 + t);
      add(idx, fc, fcp);
    }
    // G2: Σ exp(−η(r−Rs)²) f_c
    for (std::size_t t = 0; t < nG2; ++t) {
      const auto &p = radial[t];
      const double dr = r - p.rs;
      const double gauss = std::exp(-p.eta * dr * dr);
      const auto idx = static_cast<Eigen::Index>(base_g2 + nb[j].s * nG2 + t);
      add(idx, gauss * fc, gauss * (-2.0 * p.eta * dr * fc + fcp));
    }
    // G3: Σ cos(κ r) f_c
    for (std::size_t t = 0; t < nG3; ++t) {
      const double k = g3[t].kappa;
      const double c = std::cos(k * r), s = std::sin(k * r);
      const auto idx = static_cast<Eigen::Index>(base_g3 + nb[j].s * nG3 + t);
      add(idx, c * fc, -k * s * fc + c * fcp);
    }
  }

  // ---- Angular families: G4, G5 (per unordered neighbour pair) ----
  if (nG4 == 0 && nG5 == 0) {
    return out;
  }
  for (std::size_t j = 0; j < nn; ++j) {
    if (!nb[j].ok) {
      continue;
    }
    for (std::size_t k = j + 1; k < nn; ++k) {
      if (!nb[k].ok) {
        continue;
      }
      const double rij = nb[j].r, rik = nb[k].r;
      const Vec3 &dj = nb[j].d, &dk = nb[k].d;
      const Vec3 &rij_h = nb[j].rhat, &rik_h = nb[k].rhat;
      const double costh = std::clamp(dj.dot(dk) / (rij * rik), -1.0, 1.0);

      // r_jk geometry (G4 only).
      const Vec3 djk = dk - dj; // r_k − r_j
      const double rjk = djk.norm();
      Vec3 rjk_h = Vec3::Zero();
      if (rjk > 1e-14) {
        rjk_h = djk / rjk;
      }
      const auto [fcjk, fcpjk] = cutoff(rjk, rcut);

      const std::size_t po = pair_ordinal(nb[j].s, nb[k].s, S);

      // ∂cosθ/∂d_j and ∂cosθ/∂d_k.
      const Vec3 dcos_dj = (rik_h - costh * rij_h) / rij;
      const Vec3 dcos_dk = (rij_h - costh * rik_h) / rik;

      const double fci = nb[j].fc, fck = nb[k].fc;
      const double fcpi = nb[j].fcp, fcpk = nb[k].fcp;

      auto accumulate = [&](Eigen::Index idx, double zeta, double lambda,
                            double eta, bool with_jk) {
        const double A = 1.0 + lambda * costh;
        if (A <= 1e-12) {
          return; // (1+λcosθ)^ζ and its derivative are 0 (or undefined)
        }
        const double C = std::pow(2.0, 1.0 - zeta);
        const double ang = std::pow(A, zeta);
        const double dang_dcos = zeta * std::pow(A, zeta - 1.0) * lambda;

        const double rsum = rij * rij + rik * rik + (with_jk ? rjk * rjk : 0.0);
        const double rad = std::exp(-eta * rsum);
        const double fcs = fci * fck * (with_jk ? fcjk : 1.0);

        const double g = C * ang * rad * fcs;
        out.values[idx] += g;

        // Coefficients on the geometric gradients ∇rij, ∇rik, ∇rjk, ∇cosθ.
        const double drad_drij = rad * (-2.0 * eta * rij);
        const double drad_drik = rad * (-2.0 * eta * rik);
        const double drad_drjk = with_jk ? rad * (-2.0 * eta * rjk) : 0.0;
        const double dfcs_drij = fcpi * fck * (with_jk ? fcjk : 1.0);
        const double dfcs_drik = fci * fcpk * (with_jk ? fcjk : 1.0);
        const double dfcs_drjk = with_jk ? fci * fck * fcpjk : 0.0;

        const double cij = C * ang * (drad_drij * fcs + rad * dfcs_drij);
        const double cik = C * ang * (drad_drik * fcs + rad * dfcs_drik);
        const double cjk = C * ang * (drad_drjk * fcs + rad * dfcs_drjk);
        const double ccos = C * rad * fcs * dang_dcos;

        // ∂g/∂d_j and ∂g/∂d_k; ∂g/∂d_i = −(∂g/∂d_j + ∂g/∂d_k).
        const Vec3 gj = cij * rij_h - cjk * rjk_h + ccos * dcos_dj;
        const Vec3 gk = cik * rik_h + cjk * rjk_h + ccos * dcos_dk;

        out.grad_neigh[j].row(idx) += gj.transpose();
        out.grad_neigh[k].row(idx) += gk.transpose();
        out.grad_self.row(idx) -= (gj + gk).transpose();
      };

      for (std::size_t t = 0; t < nG4; ++t) {
        const auto idx = static_cast<Eigen::Index>(base_g4 + po * nG4 + t);
        accumulate(idx, g4[t].zeta, g4[t].lambda, g4[t].eta, /*with_jk=*/true);
      }
      for (std::size_t t = 0; t < nG5; ++t) {
        const auto idx = static_cast<Eigen::Index>(base_g5 + po * nG5 + t);
        accumulate(idx, g5[t].zeta, g5[t].lambda, g5[t].eta, /*with_jk=*/false);
      }
    }
  }

  return out;
}

} // namespace potfit
