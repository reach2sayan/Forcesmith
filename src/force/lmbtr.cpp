#include "forcesmith/potentials/lmbtr.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <vector>

namespace forcesmith {

namespace {

struct GaussKernel {
  double inv;  // 1 / (sigma * sqrt(2*pi)) — peak normalisation
  double s2;   // 2 * sigma^2
  double step; // uniform grid spacing
  double c;    // exp(-2*step^2/s2) — constant ratio of the recurrence
  double cut;  // truncation radius (geometry units): Gaussian treated as 0
};

GaussKernel make_kernel(const LMBTR::Grid &g) {
  GaussKernel k{};
  k.inv = 1.0 / (g.sigma * std::sqrt(2.0 * std::numbers::pi));
  k.s2 = 2.0 * g.sigma * g.sigma;
  k.step = (g.n > 1) ? (g.max - g.min) / (g.n - 1) : 0.0;
  k.c = std::exp(-2.0 * k.step * k.step / k.s2);
  k.cut = 5.0 * g.sigma;
  return k;
}

void broaden(Eigen::Ref<Eigen::VectorXd> block, const LMBTR::Grid &g,
             const GaussKernel &k, double geom, double w) {
  const double a = w * k.inv;
  if (g.n <= 1) {
    const double dx = g.min - geom;
    block[0] += a * std::exp(-dx * dx / k.s2);
    return;
  }
  const double center = (geom - g.min) / k.step;
  const double rad = k.cut / k.step;
  const int b0 = std::max(0, static_cast<int>(std::ceil(center - rad)));
  const int b1 = std::min(g.n - 1, static_cast<int>(std::floor(center + rad)));
  if (b0 > b1) {
    return; // entire Gaussian falls outside the grid
  }
  const double dx0 = g.min + static_cast<double>(b0) * k.step - geom;
  double f = std::exp(-dx0 * dx0 / k.s2);                              // f(b0)
  double u = std::exp(-(2.0 * dx0 * k.step + k.step * k.step) / k.s2); // ratio
  for (int b = b0; b <= b1; ++b) {
    block[b] += a * f;
    f *= u;
    u *= k.c;
  }
}

FORCE_INLINE void normalize_l2_inplace(Eigen::VectorXd &v) {
  const double nrm = v.norm();
  if (nrm > 1e-300) {
    v /= nrm;
  }
}

} // namespace

std::vector<LMBTR::Neighbor> LMBTR::collect_neighbors(const Atom &a) const {
  const std::size_t S = ntypes;
  std::vector<Neighbor> nb;
  nb.reserve(a.neighbors.size());
  for (const auto &neigh : a.neighbors) {
    const auto hit = descriptor_neighbor(neigh, rcut, S, 1e-14);
    if (!hit) {
      continue;
    }
    nb.emplace_back(hit->r, neigh.dist, hit->slot);
  }
  return nb;
}

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

} // namespace forcesmith
