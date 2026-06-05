#include "forcesmith/potentials/lmbtr.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <vector>

namespace forcesmith {

namespace {

FORCE_INLINE double grid_point(const LMBTR::Grid &g, int b) {
  if (g.n <= 1) {
    return g.min;
  }
  return g.min + (g.max - g.min) * static_cast<double>(b) / (g.n - 1);
}

// Accumulate a unit-area Gaussian centred at `geom`, scaled by `w`, onto the
// `g.n` grid points of `block`.
void broaden(Eigen::Ref<Eigen::VectorXd> block, const LMBTR::Grid &g,
             double geom, double w) {
  const double inv = 1.0 / (g.sigma * std::sqrt(2.0 * std::numbers::pi));
  const double s2 = 2.0 * g.sigma * g.sigma;
  for (int b : std::views::iota(0, g.n)) {
    const double dx = grid_point(g, b) - geom;
    block[b] += w * inv * std::exp(-dx * dx / s2);
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
  for (const Neighbor &j : nb) {
    const double w = std::exp(-j.r / weight_scale);
    broaden(values.segment(L.k2(j.s), n), g, j.r, w);
  }
}

// ---- step: k3 block (cos angle, per unordered species pair) ----
void LMBTR::accumulate_k3(Eigen::VectorXd &values,
                          const std::vector<Neighbor> &nb, const Grid &g,
                          const LmbtrLayout &L) const {
  const std::size_t S = ntypes;
  const auto n = static_cast<Eigen::Index>(g.n);
  for (auto [j, k] : strict_upper_triangle(nb.size())) {
    const double rij = nb[j].r, rik = nb[k].r;
    const double costh =
        std::clamp(nb[j].d.dot(nb[k].d) / (rij * rik), -1.0, 1.0);
    const double rjk = (nb[k].d - nb[j].d).norm();
    const double w = std::exp(-(rij + rik + rjk) / weight_scale);
    const std::size_t po = pair_ordinal(nb[j].s, nb[k].s, S);
    broaden(values.segment(L.k3(po), n), g, costh, w);
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
