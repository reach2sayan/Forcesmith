#pragma once

// Machine-learning potential abstraction.
//
// Idea: a configuration's local atomic environments reduce to a per-atom vector
// of DESCRIPTORS D_i; the per-atom energy is E_i = head(D_i) for a fittable
// "head" (the coefficients the optimizer drives). Forces are −dE/dr, which need
// the descriptor gradients dD_i/dr.
//
// Layering (mirrors the rest of the force layer):
//   * MLBaseImpl<Derived> is a CRTP ForceCalculator base. Its eval_forces is
//     descriptor-agnostic: it calls the concrete model's get_descriptor hook,
//     feeds the descriptor to the head, and assembles energy/forces/stress.
//   * Derived (e.g. SymmetryFunctionModel) supplies the descriptor:
//       DescriptorValue get_descriptor(const Atom&) const;
//       double          descriptor_cutoff() const;
//       bool            analytic_grads() const;
//   * The head is value-erased (EnergyHead, like Potential) so swapping the
//     descriptor→energy map (linear → kernel → NN) never multiplies the
//     ForceCalculator variant. Concrete heads share plumbing via HeadBase.
//
// The fittable parameters are the head coefficients (one head per element type);
// descriptor hyperparameters (η, Rs, cutoff …) are fixed.

#include "potfit/core/atom.hpp"
#include "potfit/core/erased.hpp"
#include "potfit/core/neighbor_list.hpp" // build_neighbor_list, bc_volume
#include "potfit/core/param.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp" // TypeArray

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace potfit {

// Per-neighbor descriptor gradient block: row k is dD_k/dr (a 3-vector).
using DescriptorGrad = Eigen::Matrix<double, Eigen::Dynamic, 3>;

// One atom's descriptor evaluation.
//   values     — D_i, length = descriptor_size
//   grad_self  — dD_i/dr_i           (S×3)
//   grad_neigh — dD_i/dr_j per neighbor jj, parallel to atom.neighbors (each S×3)
// grad_self/grad_neigh are only consumed when has_grad is true; a model that
// reports analytic_grads()==false may leave them empty and let MLBaseImpl
// finite-difference the forces instead.
struct DescriptorValue {
  Eigen::VectorXd values;
  bool has_grad = false;
  DescriptorGrad grad_self;
  std::vector<DescriptorGrad> grad_neigh;
};

// ── Energy head: value-erased descriptor → energy map ────────────────────────
namespace detail {
struct HeadConcept {
  virtual ~HeadConcept() = default;
  virtual double energy(const Eigen::VectorXd &D) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd &D) const = 0; // de/dD
  virtual std::size_t param_count() const = 0;
  virtual void gather_params(Eigen::VectorXd &x, std::size_t off) const = 0;
  virtual void scatter_params(const Eigen::VectorXd &x, std::size_t off) = 0;
  virtual std::unique_ptr<HeadConcept> clone() const = 0;
};
} // namespace detail

// CRTP base for the related concrete heads. Derived must implement:
//   double          energy_impl(const VectorXd&) const
//   Eigen::VectorXd grad_impl(const VectorXd&) const          // de/dD
//   std::array/vector<Param*>       field_ptrs()               // fittable slots
//   std::array/vector<const Param*> field_ptrs() const
// gather/scatter follow the Potential convention: write/read starting at `off`;
// the caller advances by param_count().
template <typename Derived> struct HeadBase {
  double energy(const Eigen::VectorXd &D) const { return self().energy_impl(D); }
  Eigen::VectorXd grad(const Eigen::VectorXd &D) const {
    return self().grad_impl(D);
  }
  std::size_t param_count() const {
    auto f = self().field_ptrs();
    return static_cast<std::size_t>(
        std::ranges::count_if(f, [](const Param *p) { return !p->fixed; }));
  }
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    for (const Param *p : self().field_ptrs()) {
      if (!p->fixed) {
        dst[off++] = p->value;
      }
    }
  }
  void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    for (Param *p : self().field_ptrs()) {
      if (!p->fixed) {
        p->value = src[off++];
      }
    }
  }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }
  Derived &self() { return static_cast<Derived &>(*this); }
};

// Linear head: E_i = Σ_k coeffs_k · D_k + bias.  de/dD = coeffs (constant), so
// forces reduce to coeffs · dD/dr.
struct LinearHead : HeadBase<LinearHead> {
  std::vector<Param> coeffs;
  Param bias{0.0, true};

  double energy_impl(const Eigen::VectorXd &D) const {
    double e = bias.value;
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
      e += coeffs[k].value * D[static_cast<Eigen::Index>(k)];
    }
    return e;
  }
  Eigen::VectorXd grad_impl(const Eigen::VectorXd &) const {
    Eigen::VectorXd g(static_cast<Eigen::Index>(coeffs.size()));
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
      g[static_cast<Eigen::Index>(k)] = coeffs[k].value;
    }
    return g;
  }
  std::vector<Param *> field_ptrs() {
    std::vector<Param *> f;
    f.reserve(coeffs.size() + 1);
    for (auto &c : coeffs) {
      f.push_back(&c);
    }
    f.push_back(&bias);
    return f;
  }
  std::vector<const Param *> field_ptrs() const {
    std::vector<const Param *> f;
    f.reserve(coeffs.size() + 1);
    for (const auto &c : coeffs) {
      f.push_back(&c);
    }
    f.push_back(&bias);
    return f;
  }
};

class EnergyHead : private detail::ErasedValue<detail::HeadConcept> {
  template <typename T> struct Model final : detail::HeadConcept {
    T impl_;
    explicit Model(T t) : impl_(std::move(t)) {}
    double energy(const Eigen::VectorXd &D) const override {
      return impl_.energy(D);
    }
    Eigen::VectorXd grad(const Eigen::VectorXd &D) const override {
      return impl_.grad(D);
    }
    std::size_t param_count() const override { return impl_.param_count(); }
    void gather_params(Eigen::VectorXd &x, std::size_t off) const override {
      impl_.gather_params(x, off);
    }
    void scatter_params(const Eigen::VectorXd &x, std::size_t off) override {
      impl_.scatter_params(x, off);
    }
    std::unique_ptr<detail::HeadConcept> clone() const override {
      return std::make_unique<Model>(*this);
    }
  };

  using Base = detail::ErasedValue<detail::HeadConcept>;

public:
  template <typename T>
    requires(!std::same_as<std::decay_t<T>, EnergyHead>)
  explicit EnergyHead(T &&t)
      : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

  EnergyHead(const EnergyHead &) = default;
  EnergyHead(EnergyHead &&) noexcept = default;
  EnergyHead &operator=(const EnergyHead &) = default;
  EnergyHead &operator=(EnergyHead &&) noexcept = default;

  double energy(const Eigen::VectorXd &D) const { return self_->energy(D); }
  Eigen::VectorXd grad(const Eigen::VectorXd &D) const {
    return self_->grad(D);
  }
  std::size_t param_count() const { return self_->param_count(); }
  void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
};

// ── CRTP ML force-calculator base ────────────────────────────────────────────
template <typename Derived>
struct MLBaseImpl : ForceCalculatorBase<Derived>, NoGlobals {
  // One head per element type, indexed by atom.type.index. The descriptor (and
  // its hyperparameters) lives in Derived.
  TypeArray<EnergyHead> heads;

  void eval_forces(Configuration &cfg) const {
    build_neighbor_list(cfg, max_cutoff());

    cfg.calc_energy = 0.0;
    cfg.calc_stress = SymTens::Zero();
    for (auto &a : cfg.atoms) {
      a.calc_force = Vec3::Zero();
    }

    const bool analytic = self().analytic_grads();

    for (auto &ai : cfg.atoms) {
      const DescriptorValue d = self().get_descriptor(ai);
      const EnergyHead &h = heads[ai.type.index];
      cfg.calc_energy += h.energy(d.values);
      if (analytic && d.has_grad) {
        accumulate_atom_forces(cfg, ai, d, h.grad(d.values));
      }
    }

    if (analytic) {
      cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)
    } else {
      // Bootstrap / validation path: no analytic gradients, so differentiate
      // the total energy numerically. O(N) energy evaluations × 6 — slow by
      // design; stress is not filled here.
      accumulate_forces_fd(cfg);
    }

    events::on_force_eval(
        events::ForceEvalStats{this->conf_index, force_rms(cfg), cfg});
  }

  std::size_t param_count() const {
    std::size_t n = 0;
    for (const auto &h : heads) {
      n += h.param_count();
    }
    return n;
  }

  void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    for (const auto &h : heads) {
      h.gather_params(dst, off);
      off += h.param_count();
    }
  }

  void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    for (auto &h : heads) {
      h.scatter_params(src, off);
      off += h.param_count();
    }
  }

  double max_cutoff() const { return self().descriptor_cutoff(); }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }

  // Forces from atom i's descriptor: F_m = −Σ_i (de/dD_i)·(dD_i/dr_m).
  // dD_i/dr_m is nonzero only for m=i (grad_self) and m∈neighbors(i).
  void accumulate_atom_forces(Configuration &cfg, Atom &ai,
                              const DescriptorValue &d,
                              const Eigen::VectorXd &dEdD) const {
    ai.calc_force -= d.grad_self.transpose() * dEdD;
    for (std::size_t jj = 0; jj < ai.neighbors.size(); ++jj) {
      Atom &aj = const_cast<Atom &>(*ai.neighbors[jj].neighbor);
      const Vec3 f_on_j = -(d.grad_neigh[jj].transpose() * dEdD);
      aj.calc_force += f_on_j;
      // Virial: bond ⊗ force-on-partner (matches Tersoff/EAM convention).
      cfg.calc_stress += ai.neighbors[jj].dist * f_on_j.transpose();
    }
  }

  // Total energy at the current geometry, rebuilding the neighbor list first.
  double total_energy(Configuration &cfg) const {
    build_neighbor_list(cfg, max_cutoff());
    double e = 0.0;
    for (auto &ai : cfg.atoms) {
      e += heads[ai.type.index].energy(self().get_descriptor(ai).values);
    }
    return e;
  }

  void accumulate_forces_fd(Configuration &cfg) const {
    constexpr double h = 1e-5;
    for (auto &am : cfg.atoms) {
      for (int c = 0; c < 3; ++c) {
        const double x0 = am.pos[c];
        am.pos[c] = x0 + h;
        const double ep = total_energy(cfg);
        am.pos[c] = x0 - h;
        const double em = total_energy(cfg);
        am.pos[c] = x0;
        am.calc_force[c] = -(ep - em) / (2.0 * h);
      }
    }
    build_neighbor_list(cfg, max_cutoff()); // restore the unperturbed list
  }
};

} // namespace potfit
