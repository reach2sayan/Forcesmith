#include "potfit/potentials/lmbtr.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace potfit {

namespace {

// Position in the upper_triangle(S) enumeration of the unordered pair {a,b}.
inline std::size_t pair_ordinal(std::size_t a, std::size_t b, std::size_t S) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return lo * S - lo * (lo - 1) / 2 + (hi - lo);
}

inline double grid_point(const LMBTR::Grid &g, int b) {
  if (g.n <= 1) {
    return g.min;
  }
  return g.min + (g.max - g.min) * static_cast<double>(b) / (g.n - 1);
}

} // namespace

DescriptorValue LMBTR::get_descriptor(const Atom &a) const {
  const std::size_t S = ntypes;
  const std::size_t P = S * (S + 1) / 2;
  const std::size_t n2 = use_k2 ? static_cast<std::size_t>(k2.n) : 0;
  const std::size_t n3 = use_k3 ? static_cast<std::size_t>(k3.n) : 0;

  const std::size_t size_k2 = use_k2 ? S * n2 : 0;
  const std::size_t base_k3 = size_k2;
  const auto total =
      static_cast<Eigen::Index>(size_k2 + (use_k3 ? P * n3 : 0));

  const std::size_t nn = a.neighbors.size();

  DescriptorValue out;
  out.values = Eigen::VectorXd::Zero(total);
  out.has_grad = false; // forces via the framework's finite-difference fallback

  // Precompute per-neighbour distance/species.
  struct NB {
    double r = 0;
    Vec3 d;
    std::size_t s = 0;
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
    nb[j] = NB{r, d, static_cast<std::size_t>(si), true};
  }

  // Unit-area Gaussian broadening onto a grid block [base, base+n).
  auto broaden = [&](std::size_t base, const Grid &g, std::size_t n,
                     double geom, double w) {
    const double inv = 1.0 / (g.sigma * std::sqrt(2.0 * std::numbers::pi));
    const double s2 = 2.0 * g.sigma * g.sigma;
    for (std::size_t b = 0; b < n; ++b) {
      const double x = grid_point(g, static_cast<int>(b));
      const double dx = x - geom;
      out.values[static_cast<Eigen::Index>(base + b)] +=
          w * inv * std::exp(-dx * dx / s2);
    }
  };

  // ---- k2: distance, per neighbour species ----
  if (use_k2) {
    for (std::size_t j = 0; j < nn; ++j) {
      if (!nb[j].ok) {
        continue;
      }
      const double w = std::exp(-nb[j].r / weight_scale);
      broaden(nb[j].s * n2, k2, n2, nb[j].r, w);
    }
  }

  // ---- k3: cos(angle), per unordered species pair ----
  if (use_k3) {
    for (std::size_t j = 0; j < nn; ++j) {
      if (!nb[j].ok) {
        continue;
      }
      for (std::size_t k = j + 1; k < nn; ++k) {
        if (!nb[k].ok) {
          continue;
        }
        const double rij = nb[j].r, rik = nb[k].r;
        const double costh =
            std::clamp(nb[j].d.dot(nb[k].d) / (rij * rik), -1.0, 1.0);
        const double rjk = (nb[k].d - nb[j].d).norm();
        const double w = std::exp(-(rij + rik + rjk) / weight_scale);
        const std::size_t po = pair_ordinal(nb[j].s, nb[k].s, S);
        broaden(base_k3 + po * n3, k3, n3, costh, w);
      }
    }
  }

  // ---- optional L2 normalization ----
  if (normalize_l2) {
    const double nrm = out.values.norm();
    if (nrm > 1e-300) {
      out.values /= nrm;
    }
  }

  return out;
}

} // namespace potfit
