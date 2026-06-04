#include "potfit/potentials/acsf.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

namespace potfit {

namespace {

// Cosine cutoff and its derivative; both vanish for r >= rcut.
FORCE_INLINE std::pair<double, double> cutoff(double r, double rcut) {
  if (r >= rcut) {
    return {0.0, 0.0};
  }
  const double x = std::numbers::pi * r / rcut;
  return {0.5 * (1.0 + std::cos(x)),
          -0.5 * std::numbers::pi / rcut * std::sin(x)};
}

// Scatter one radial-family contribution at descriptor index `idx`: value g and
// radial derivative dg, for neighbour `orig` with unit bond direction rhat.
//   dD/dr_j = dg·rhat, dD/dr_i = −dg·rhat.
void scatter_radial(DescriptorValue &out, std::size_t orig, Eigen::Index idx,
                    const Eigen::RowVector3d &rhat, double g, double dg) {
  out.values[idx] += g;
  out.grad_neigh[orig].row(idx) += dg * rhat;
  out.grad_self.row(idx) -= dg * rhat;
}

// Per-pair geometry (j<k), computed once and shared by every G4/G5 term.
struct AngularPair {
  std::size_t oj, ok; // original neighbour indices (into atom.neighbors)
  double rij, rik, rjk;
  Vec3 rij_h, rik_h, rjk_h;
  double costh;
  Vec3 dcos_dj, dcos_dk;       // ∂cosθ/∂d_j, ∂cosθ/∂d_k
  double fci, fck, fcpi, fcpk; // r_ij / r_ik cutoffs + derivatives
  double fcjk, fcpjk;          // r_jk cutoff + derivative (G4 only)
};

// Scatter one angular-family term at descriptor index `idx`. G4 includes the
// r_jk factor (with_jk=true); G5 omits it (with_jk=false).
void scatter_angular(DescriptorValue &out, const AngularPair &p,
                     Eigen::Index idx, double zeta, double lambda, double eta,
                     bool with_jk) {
  const double A = 1.0 + lambda * p.costh;
  if (A <= 1e-12) {
    return; // (1+λcosθ)^ζ and its derivative are 0 (or undefined)
  }
  const double C = std::pow(2.0, 1.0 - zeta);
  const double ang = std::pow(A, zeta);
  const double dang_dcos = zeta * std::pow(A, zeta - 1.0) * lambda;

  const double rsum =
      p.rij * p.rij + p.rik * p.rik + (with_jk ? p.rjk * p.rjk : 0.0);
  const double rad = std::exp(-eta * rsum);
  const double fcs = p.fci * p.fck * (with_jk ? p.fcjk : 1.0);

  const double g = C * ang * rad * fcs;
  out.values[idx] += g;

  // Coefficients on the geometric gradients ∇rij, ∇rik, ∇rjk, ∇cosθ.
  const double drad_drij = rad * (-2.0 * eta * p.rij);
  const double drad_drik = rad * (-2.0 * eta * p.rik);
  const double drad_drjk = with_jk ? rad * (-2.0 * eta * p.rjk) : 0.0;
  const double dfcs_drij = p.fcpi * p.fck * (with_jk ? p.fcjk : 1.0);
  const double dfcs_drik = p.fci * p.fcpk * (with_jk ? p.fcjk : 1.0);
  const double dfcs_drjk = with_jk ? p.fci * p.fck * p.fcpjk : 0.0;

  const double cij = C * ang * (drad_drij * fcs + rad * dfcs_drij);
  const double cik = C * ang * (drad_drik * fcs + rad * dfcs_drik);
  const double cjk = C * ang * (drad_drjk * fcs + rad * dfcs_drjk);
  const double ccos = C * rad * fcs * dang_dcos;

  // ∂g/∂d_j and ∂g/∂d_k; ∂g/∂d_i = −(∂g/∂d_j + ∂g/∂d_k).
  const Vec3 gj = cij * p.rij_h - cjk * p.rjk_h + ccos * p.dcos_dj;
  const Vec3 gk = cik * p.rik_h + cjk * p.rjk_h + ccos * p.dcos_dk;

  out.grad_neigh[p.oj].row(idx) += gj.transpose();
  out.grad_neigh[p.ok].row(idx) += gk.transpose();
  out.grad_self.row(idx) -= (gj + gk).transpose();
}

} // namespace

// ---- step: per-neighbour distance/species filter + cutoff precompute ----
std::vector<ACSF::Neighbor> ACSF::collect_neighbors(const Atom &a) const {
  const std::size_t S = ntypes;
  std::vector<Neighbor> nb;
  nb.reserve(a.neighbors.size());
  for (auto [j, neigh] : a.neighbors | std::views::enumerate) {
    const Vec3 &d = neigh.dist;
    const double r = d.norm();
    const auto si = static_cast<long long>(neigh.neighbor->type.index);
    if (r < 1e-14 || r >= rcut || si < 0 || static_cast<std::size_t>(si) >= S) {
      continue;
    }
    auto [fc, fcp] = cutoff(r, rcut);
    nb.push_back(Neighbor{static_cast<std::size_t>(j), d, d / r, r, fc, fcp,
                          static_cast<std::size_t>(si)});
  }
  return nb;
}

// ---- step: radial families G1/G2/G3 (per neighbour, per species channel) ----
void ACSF::accumulate_radial(DescriptorValue &out,
                             const std::vector<Neighbor> &nb,
                             const AcsfLayout &L) const {
  for (const Neighbor &j : nb) {
    const double r = j.r, fc = j.fc, fcp = j.fcp;
    const Eigen::RowVector3d rhat = j.rhat.transpose();

    // G1: Σ f_c
    for (std::size_t t : std::views::iota(std::size_t{0}, g1)) {
      scatter_radial(out, j.orig, L.radial(SymmetryFunctionFamily::G1, j.s, t),
                     rhat, fc, fcp);
    }
    // G2: Σ exp(−η(r−Rs)²) f_c
    for (auto [ti, p] : radial | std::views::enumerate) {
      const auto t = static_cast<std::size_t>(ti);
      const double dr = r - p.rs;
      const double gauss = std::exp(-p.eta * dr * dr);
      scatter_radial(out, j.orig, L.radial(SymmetryFunctionFamily::G2, j.s, t),
                     rhat, gauss * fc, gauss * (-2.0 * p.eta * dr * fc + fcp));
    }
    // G3: Σ cos(κ r) f_c
    for (auto [ti, gp] : g3 | std::views::enumerate) {
      const auto t = static_cast<std::size_t>(ti);
      const double k = gp.kappa;
      const double c = std::cos(k * r), s = std::sin(k * r);
      scatter_radial(out, j.orig, L.radial(SymmetryFunctionFamily::G3, j.s, t),
                     rhat, c * fc, -k * s * fc + c * fcp);
    }
  }
}

// ---- step: angular families G4/G5 (per unordered neighbour pair) ----
void ACSF::accumulate_angular(DescriptorValue &out,
                              const std::vector<Neighbor> &nb,
                              const AcsfLayout &L) const {
  if (g4.empty() && g5.empty()) {
    return;
  }
  const std::size_t S = ntypes;
  for (const auto &[nbj, nbk] :
       strict_upper_triangle(nb.size()) |
           std::views::transform([&nb](auto pair) {
             return std::pair<const Neighbor &, const Neighbor &>(
                 nb[pair.first], nb[pair.second]);
           })) {
    AngularPair p;
    p.oj = nbj.orig;
    p.ok = nbk.orig;
    p.rij = nbj.r;
    p.rik = nbk.r;
    p.rij_h = nbj.rhat;
    p.rik_h = nbk.rhat;
    p.costh = std::clamp(nbj.d.dot(nbk.d) / (p.rij * p.rik), -1.0, 1.0);

    // r_jk geometry (G4 only).
    const Vec3 djk = nbk.d - nbj.d; // r_k − r_j
    p.rjk = djk.norm();
    p.rjk_h = p.rjk > 1e-14 ? Vec3(djk / p.rjk) : Vec3::Zero();
    std::tie(p.fcjk, p.fcpjk) = cutoff(p.rjk, rcut);

    // ∂cosθ/∂d_j and ∂cosθ/∂d_k.
    p.dcos_dj = (p.rik_h - p.costh * p.rij_h) / p.rij;
    p.dcos_dk = (p.rij_h - p.costh * p.rik_h) / p.rik;

    p.fci = nbj.fc;
    p.fck = nbk.fc;
    p.fcpi = nbj.fcp;
    p.fcpk = nbk.fcp;

    const std::size_t po = pair_ordinal(nbj.s, nbk.s, S);

    for (auto [ti, gp] : g4 | std::views::enumerate) {
      const auto t = static_cast<std::size_t>(ti);
      scatter_angular(out, p, L.angular(SymmetryFunctionFamily::G4, po, t),
                      gp.zeta, gp.lambda, gp.eta, /*with_jk=*/true);
    }
    for (auto [ti, gp] : g5 | std::views::enumerate) {
      const auto t = static_cast<std::size_t>(ti);
      scatter_angular(out, p, L.angular(SymmetryFunctionFamily::G5, po, t),
                      gp.zeta, gp.lambda, gp.eta, /*with_jk=*/false);
    }
  }
}

DescriptorValue ACSF::get_descriptor(const Atom &a) const {
  // The descriptor's flat layout (block bases + per-component indices) lives in
  // one place; no offset arithmetic in the steps below.
  const AcsfLayout L{ntypes,    g1,        radial.size(),
                     g3.size(), g4.size(), g5.size()};
  const std::vector<Neighbor> nb = collect_neighbors(a);

  DescriptorValue out;
  out.values = Eigen::VectorXd::Zero(L.size());
  out.has_grad = true;
  out.grad_self = DescriptorGrad::Zero(L.size(), 3);
  // grad_neigh stays full-length and parallel to atom.neighbors; the steps
  // scatter into row `orig` of the matching neighbour.
  out.grad_neigh.assign(a.neighbors.size(), DescriptorGrad::Zero(L.size(), 3));

  accumulate_radial(out, nb, L);
  accumulate_angular(out, nb, L);
  return out;
}

} // namespace potfit
