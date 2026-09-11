#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/descriptor_layout.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/ml_force.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace forcesmith {

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

  [[nodiscard]] LmbtrLayout layout_for(std::size_t S) const {
    const auto grid_n = [](const Grid &g) { return g.n; };
    return LmbtrLayout{S, k2.transform(grid_n), k3.transform(grid_n)};
  }

private:
  struct Neighbor {
    double r;
    Vec3 d;
    std::size_t s;
  };

  [[nodiscard]] LmbtrLayout layout() const {
    const auto grid_n = [](const Grid &g) { return g.n; };
    return LmbtrLayout{ntypes, k2.transform(grid_n), k3.transform(grid_n)};
  }

  [[nodiscard]] std::vector<Neighbor> collect_neighbors(const Atom &a) const;
  void accumulate_k2(Eigen::VectorXd &values, const std::vector<Neighbor> &nb,
                     const Grid &g, const LmbtrLayout &L) const;
  void accumulate_k3(Eigen::VectorXd &values, const std::vector<Neighbor> &nb,
                     const Grid &g, const LmbtrLayout &L) const;
};

static_assert(CForceCalculator<LMBTR>);

static_assert(CDescriptorModel<LMBTR>);

} // namespace forcesmith
