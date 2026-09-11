#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/ml_force.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace forcesmith {

struct SoapRadialTable;

struct SoapModel : MLBase<SoapModel> {
  using MLBase<SoapModel>::ntypes;
  int n_max = 6;      // radial basis size
  int l_max = 6;      // max spherical-harmonic degree
  double rcut = 6.0;  // environment cutoff (Å)
  double sigma = 0.5; // atomic Gaussian width (Å)

  mutable Eigen::MatrixXd beta; // β = S^{-1/2} (n_max × n_max)

  mutable std::shared_ptr<const SoapRadialTable> radial_;
  void init_radial_basis();

  [[nodiscard]] DescriptorValue get_descriptor(const Atom &a) const;
  [[nodiscard]] double descriptor_cutoff() const { return rcut; }
  [[nodiscard]] bool analytic_grads() const { return true; }
  [[nodiscard]] std::size_t descriptor_size() const;

  [[nodiscard]] std::vector<std::optional<Eigen::Index>>
  descriptor_index_map(const SpeciesRegistry &old_reg,
                       const SpeciesRegistry &new_reg) const;

private:

  [[nodiscard]] Eigen::Index coeff_index(int ch, int n, int l, int m) const {
    const int nlm = 2 * l_max + 1;
    return ((static_cast<Eigen::Index>(ch) * n_max + n) * (l_max + 1) + l) *
               nlm +
           (m + l_max);
  }

  [[nodiscard]] Eigen::VectorXcd
  compute_coefficients(const Atom &a, const SoapRadialTable &tab,
                       double K) const;

  [[nodiscard]] Eigen::VectorXd
  power_spectrum(const Eigen::VectorXcd &c,
                 const std::vector<double> &wl) const;

  void position_gradient(const Atom &a, const Eigen::VectorXcd &c,
                         const SoapRadialTable &tab, double K,
                         const std::vector<double> &wl, double norm,
                         DescriptorValue &out) const;
};

static_assert(CForceCalculator<SoapModel>);

static_assert(CDescriptorModel<SoapModel>);

} // namespace forcesmith
