#include "forcesmith/potentials/lmbtr.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <vector>

namespace forcesmith {

namespace {

// Per-grid Gaussian broadening constants. These depend only on the grid, so we
// build them once per descriptor (in accumulate_k2/k3) and reuse across every
// neighbour/pair, instead of recomputing them — and re-calling exp() for the
// recurrence constant — on every broaden() call.
struct GaussKernel {
  double inv;   // 1 / (sigma * sqrt(2*pi)) — peak normalisation
  double s2;    // 2 * sigma^2
  double step;  // uniform grid spacing
  double c;     // exp(-2*step^2/s2) — constant ratio of the recurrence
  double cut;   // truncation radius (geometry units): Gaussian treated as 0
                // beyond |grid - geom| > cut
};

GaussKernel make_kernel(const LMBTR::Grid &g) {
  GaussKernel k{};
  k.inv = 1.0 / (g.sigma * std::sqrt(2.0 * std::numbers::pi));
  k.s2 = 2.0 * g.sigma * g.sigma;
  k.step = (g.n > 1) ? (g.max - g.min) / (g.n - 1) : 0.0;
  k.c = std::exp(-2.0 * k.step * k.step / k.s2);
  // Beyond 5 sigma the unit-area Gaussian's remaining tail is ~3e-7, well below
  // the descriptor's fitting tolerance; truncating there skips the bulk of grid
  // points (and their mul-adds) for each broaden.
  k.cut = 5.0 * g.sigma;
  return k;
}

// Accumulate a unit-area Gaussian centred at `geom`, scaled by `w`, onto the
// grid points of `block`. Only points within `k.cut` of `geom` are touched, and
// the Gaussian over that uniform window is evaluated with an exact 3-term
// recurrence (f(b+1) = f(b)*u(b), u(b+1) = u(b)*c), so just two exp() calls are
// made regardless of window width. Result matches the per-point exp() form up to
// floating-point rounding of the multiply chain plus the truncated tail.
void broaden(Eigen::Ref<Eigen::VectorXd> block, const LMBTR::Grid &g,
             const GaussKernel &k, double geom, double w) {
  const double a = w * k.inv;
  if (g.n <= 1) {
    const double dx = g.min - geom;
    block[0] += a * std::exp(-dx * dx / k.s2);
    return;
  }
  // Index range [b0, b1] of grid points within the truncation window, where
  // grid(b) = g.min + b*step, so the fractional index of `geom` is below.
  const double center = (geom - g.min) / k.step;
  const double rad = k.cut / k.step;
  const int b0 = std::max(0, static_cast<int>(std::ceil(center - rad)));
  const int b1 = std::min(g.n - 1, static_cast<int>(std::floor(center + rad)));
  if (b0 > b1) {
    return; // entire Gaussian falls outside the grid
  }
  // Seed the recurrence at b0: dx(b) = (g.min + b*step) - geom.
  const double dx0 = g.min + static_cast<double>(b0) * k.step - geom;
  double f = std::exp(-dx0 * dx0 / k.s2);                              // f(b0)
  double u =
      std::exp(-(2.0 * dx0 * k.step + k.step * k.step) / k.s2);        // ratio
  for (int b = b0; b <= b1; ++b) {
    block[b] += a * f;
    f *= u;
    u *= k.c;
  }
}

// Guarded L2 normalization (no-op on a (near-)zero vector).
FORCE_INLINE void normalize_l2_inplace(Eigen::VectorXd &v) {
  const double nrm = v.norm();
  if (nrm > 1e-300) {
    v /= nrm;
  }
}

} // namespace

// ---- step: per-neighbour distance/species filter ----
std::vector<LMBTR::Neighbor> LMBTR::collect_neighbors(const Atom &a) const {
  const std::size_t S = ntypes;
  std::vector<Neighbor> nb;
  nb.reserve(a.neighbors.size());
  for (const auto &neigh : a.neighbors) {
    const Vec3 &d = neigh.dist;
    const double r = d.norm();
    const auto si = static_cast<long long>(neigh.neighbor->type.index);
    if (r < 1e-14 || r >= rcut || si < 0 || static_cast<std::size_t>(si) >= S) {
      continue;
    }
    nb.emplace_back(r, d, static_cast<std::size_t>(si));
  }
  return nb;
}

// ---- step: k2 block (distance, per neighbour species) ----
void LMBTR::accumulate_k2(Eigen::VectorXd &values,
                          const std::vector<Neighbor> &nb, const Grid &g,
                          const LmbtrLayout &L) const {
  const auto n = static_cast<Eigen::Index>(g.n);
  const GaussKernel k = make_kernel(g);
  for (const Neighbor &j : nb) {
    const double w = std::exp(-j.r / weight_scale);
    broaden(values.segment(L.k2(j.s), n), g, k, j.r, w);
  }
}

// ---- step: k3 block (cos angle, per unordered species pair) ----
void LMBTR::accumulate_k3(Eigen::VectorXd &values,
                          const std::vector<Neighbor> &nb, const Grid &g,
                          const LmbtrLayout &L) const {
  const std::size_t S = ntypes;
  const auto n = static_cast<Eigen::Index>(g.n);
  const GaussKernel kern = make_kernel(g);
  for (auto [j, k] : strict_upper_triangle(nb.size())) {
    const double rij = nb[j].r, rik = nb[k].r;
    const double costh =
        std::clamp(nb[j].d.dot(nb[k].d) / (rij * rik), -1.0, 1.0);
    const double rjk = (nb[k].d - nb[j].d).norm();
    const double w = std::exp(-(rij + rik + rjk) / weight_scale);
    const std::size_t po = pair_ordinal(nb[j].s, nb[k].s, S);
    broaden(values.segment(L.k3(po), n), g, kern, costh, w);
  }
}

DescriptorValue LMBTR::get_descriptor(const Atom &a) const {
  // The descriptor's flat layout (block bases + per-channel offsets) lives in
  // one place; the steps broaden into the shared values vector below.
  const LmbtrLayout L = layout();
  const std::vector<Neighbor> nb = collect_neighbors(a);

  DescriptorValue out;
  out.values = Eigen::VectorXd::Zero(L.size());
  out.has_grad = false; // forces via the framework's finite-difference fallback
  if (k2) {
    accumulate_k2(out.values, nb, *k2, L);
  }
  if (k3) {
    accumulate_k3(out.values, nb, *k3, L);
  }
  if (normalize_l2) {
    normalize_l2_inplace(out.values);
  }
  return out;
}

std::vector<std::optional<Eigen::Index>>
LMBTR::descriptor_index_map(const SpeciesRegistry &old_reg,
                            const SpeciesRegistry &new_reg) const {
  const std::size_t S_old = forcesmith::ntypes(old_reg);
  const std::size_t S_new = forcesmith::ntypes(new_reg);
  const auto grid_n = [](const Grid &g) { return g.n; };
  const LmbtrLayout old_L{S_old, k2.transform(grid_n), k3.transform(grid_n)};
  const LmbtrLayout new_L{S_new, k2.transform(grid_n), k3.transform(grid_n)};
  return remap_layout(old_L.d_, new_L.d_, old_slot_of_new(old_reg, new_reg),
                      S_old, S_new);
}

} // namespace forcesmith
