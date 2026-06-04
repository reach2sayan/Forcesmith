#pragma once

// Local Many-Body Tensor Representation (LMBTR) — a local per-atom ML
// descriptor (DScribe's LMBTR). Around each central atom it builds:
//   k2 (distance):  for every neighbour j, the geometry value r_ij is broadened
//                   by a unit-area Gaussian onto a fixed distance grid, weighted
//                   by w = exp(−r_ij / weight_scale), channelled per neighbour
//                   species s∈[0,ntypes).
//   k3 (cosine):    for every neighbour pair (j,k), the geometry value cosθ_ijk
//                   is broadened onto a fixed cosine grid, weighted by
//                   w = exp(−(r_ij+r_ik+r_jk) / weight_scale), channelled per
//                   unordered species pair (upper_triangle(ntypes), as SOAP).
// Each k-term is independently optional: a term is active iff its grid is set
// (std::optional<Grid>), and the descriptor is the concatenation of the active
// blocks (k2 first, then k3). The full vector is optionally L2-normalized. All
// hyperparameters are fixed; only the head coefficients are fitted. Gradients
// use the framework's finite-difference fallback (analytic_grads() == false).

#include "potfit/core/atom.hpp"
#include "potfit/force/descriptor_layout.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/ml_force.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace potfit {

// Flat layout of the LMBTR descriptor vector — the concatenation of the active
// blocks [k2 per species][k3 per pair] — built once from the present grid sizes
// + ntypes. A thin wrapper over the shared DescriptorLayout that keeps the
// k2/k3 vocabulary; each term is optional, so its block is added (and its base
// recorded) only when its grid is set. The accessors return the channel's flat
// offset (t = 0), the first grid point of that channel's segment.
struct LmbtrLayout {
  DescriptorLayout d_;
  std::optional<std::size_t> k2b, k3b;

  LmbtrLayout(std::size_t S, std::optional<int> n2, std::optional<int> n3) {
    const std::size_t P = S * (S + 1) / 2;
    if (n2) {
      k2b = d_.add(static_cast<std::size_t>(*n2), S);
    }
    if (n3) {
      k3b = d_.add(static_cast<std::size_t>(*n3), P);
    }
  }

  [[nodiscard]] Eigen::Index size() const { return d_.size(); }
  [[nodiscard]] Eigen::Index k2(std::size_t s) const {
    return d_.index(*k2b, s, 0);
  }
  [[nodiscard]] Eigen::Index k3(std::size_t po) const {
    return d_.index(*k3b, po, 0);
  }
};

struct LMBTR : MLBase<LMBTR> {
  struct Grid {
    double min = 0.0;
    double max = 6.0;
    int n = 50;
    double sigma = 0.3;
  };

  std::optional<Grid> k2 = Grid{0.0, 6.0, 50, 0.3};  // geometry = distance r_ij
  std::optional<Grid> k3 = Grid{-1.0, 1.0, 50, 0.1}; // geometry = cos θ_ijk
  double rcut = 6.0;
  double weight_scale = 3.0; // exp weighting decay length
  bool normalize_l2 = true;

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] constexpr double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] constexpr bool analytic_grads() const { return false; }
  [[nodiscard]] std::size_t descriptor_size() const {
    return static_cast<std::size_t>(layout().size());
  }

private:
  // A valid neighbour after distance/species filtering (collect_neighbors).
  struct Neighbor {
    double r;
    Vec3 d;
    std::size_t s;
  };

  // The descriptor layout for the currently-active terms.
  [[nodiscard]] LmbtrLayout layout() const {
    const auto grid_n = [](const Grid &g) { return g.n; };
    return LmbtrLayout{ntypes, k2.transform(grid_n), k3.transform(grid_n)};
  }

  // Pipeline steps (defined in lmbtr.cpp). Member functions because they read
  // rcut/ntypes/weight_scale; they broaden in place into the shared values
  // vector at the layout-assigned block offsets.
  [[nodiscard]] std::vector<Neighbor> collect_neighbors(const Atom &a) const;
  void accumulate_k2(Eigen::VectorXd &values, const std::vector<Neighbor> &nb,
                     const Grid &g, const LmbtrLayout &L) const;
  void accumulate_k3(Eigen::VectorXd &values, const std::vector<Neighbor> &nb,
                     const Grid &g, const LmbtrLayout &L) const;
};

static_assert(ForceCalculatorModel<LMBTR>);

} // namespace potfit
