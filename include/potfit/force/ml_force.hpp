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
//     ForceCalculator variant. Concrete heads get their optimizer
//     param-plumbing from the HeadParams CRTP mixin.
//
// The fittable parameters are the head coefficients (one head per element
// type); descriptor hyperparameters (η, Rs, cutoff …) are fixed.

#include "potfit/core/atom.hpp"
#include "potfit/core/erased.hpp"
#include "potfit/core/neighbor_list.hpp" // build_neighbor_list, bc_volume
#include "potfit/core/param.hpp"
#include "potfit/events/signals.hpp"
#include "potfit/force/force_calculator_concept.hpp"
#include "potfit/force/potential_table.hpp" // TypeArray

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace potfit {

// Per-neighbor descriptor gradient block: row k is dD_k/dr (a 3-vector).
using DescriptorGrad = Eigen::Matrix<double, Eigen::Dynamic, 3>;

// One atom's descriptor evaluation.
//   values     — D_i, length = descriptor_size
//   grad_self  — dD_i/dr_i           (S×3)
//   grad_neigh — dD_i/dr_j per neighbor jj, parallel to atom.neighbors (each
//   S×3)
// grad_self/grad_neigh are only consumed when has_grad is true; a model that
// reports analytic_grads()==false may leave them empty and let MLBaseImpl
// finite-difference the forces instead.
struct DescriptorValue {
  Eigen::VectorXd values;
  bool has_grad = false;
  DescriptorGrad grad_self;
  std::vector<DescriptorGrad> grad_neigh;
};

namespace detail {
struct HeadConcept {
  virtual ~HeadConcept() = default;
  virtual double energy(const Eigen::VectorXd &D) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd &D) const = 0; // de/dD
  virtual std::size_t param_count() const = 0;
  virtual void gather_params(Eigen::VectorXd &x, std::size_t off) const = 0;
  virtual void scatter_params(const Eigen::VectorXd &x, std::size_t off) = 0;
  // Generic (de)serialization surface so the io layer can round-trip any head
  // without knowing its concrete type (keeps nlohmann out of this header).
  //   type_tag()     — "linear" | "mlp" …
  //   architecture() — shape ints; linear: {n_coeffs}; mlp: {in,h1,…,1}
  //   all_values()   — every parameter (free AND fixed), flat
  virtual std::string type_tag() const = 0;
  virtual std::vector<int> architecture() const = 0;
  virtual Eigen::VectorXd all_values() const = 0;
  virtual void set_all_values(const Eigen::VectorXd &v) = 0;
  virtual std::unique_ptr<HeadConcept> clone() const = 0;
};
} // namespace detail

// CRTP mixin that generates the optimizer param-plumbing for a concrete head
// from its fittable slots. The head itself supplies the maths and its slots:
//   double          energy(const VectorXd&) const
//   Eigen::VectorXd grad(const VectorXd&) const                // de/dD
//   std::array/vector<Param*>       field_ptrs()               // fittable
//   slots std::array/vector<const Param*> field_ptrs() const
// gather/scatter follow the Potential convention: write/read starting at `off`;
// the caller advances by param_count().
template <typename Derived> struct HeadParams {
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

  // Default (de)serialization over field_ptrs() — ALL params, free and fixed.
  // Derived must still supply type_tag() and architecture(). Heads that do not
  // store their parameters as Param* (e.g. MLPHead) override these.
  Eigen::VectorXd all_values() const;
  void set_all_values(const Eigen::VectorXd &v);

private:
  constexpr const Derived &self() const {
    return static_cast<const Derived &>(*this);
  }
  constexpr Derived &self() { return static_cast<Derived &>(*this); }
};

template <typename Derived>
Eigen::VectorXd HeadParams<Derived>::all_values() const {
  auto f = self().field_ptrs();
  Eigen::VectorXd v(static_cast<Eigen::Index>(f.size()));
  std::transform(f.begin(), f.end(), v.data(),
                 [](const auto &node) { return node->value; });
  return v;
}

template <typename Derived>
void HeadParams<Derived>::set_all_values(const Eigen::VectorXd &v) {
  auto f = self().field_ptrs();
  assert(v.size() == static_cast<Eigen::Index>(f.size()));
  std::transform(v.data(), v.data() + v.size(), f.begin(), f.begin(),
                 [](double value, auto *field) {
                   field->value = value;
                   return field;
                 });
}

// Linear head: E_i = Σ_k coeffs_k · D_k + bias.  de/dD = coeffs (constant), so
// forces reduce to coeffs · dD/dr.
struct LinearHead : HeadParams<LinearHead> {
  std::vector<Param> coeffs;
  Param bias{0.0, true};

  double energy(const Eigen::VectorXd &D) const;
  Eigen::VectorXd grad(const Eigen::VectorXd &) const;
  std::vector<Param *> field_ptrs();
  std::vector<const Param *> field_ptrs() const;

  // all_values()/set_all_values() inherited from HeadParams emit [coeffs…,
  // bias].
  constexpr std::string type_tag() const { return "linear"; }
  constexpr std::vector<int> architecture() const {
    return {static_cast<int>(coeffs.size())};
  }
};

// Multilayer-perceptron head (Behler–Parrinello style): E_i = MLP(D_i), a stack
// of dense layers with a nonlinear activation and a linear scalar output. The
// fittable parameters are all weights and biases. Forward and backward passes
// use Eigen matrix algebra; grad returns the EXACT input gradient de/dD
// (used to assemble forces), which is independent of the optimizer's
// finite-difference Jacobian over the weights themselves.
struct MLPHead : HeadParams<MLPHead> {
  enum class Act { Tanh, SiLU };

  std::vector<Eigen::MatrixXd> W; // W[l] is (out_l × in_l)
  std::vector<Eigen::VectorXd> b; // b[l] is (out_l)
  Act act = Act::Tanh;

  // Build from layer sizes [in, h1, …, 1] with small deterministic init (a tiny
  // LCG keyed off `seed`; avoids any RNG that would break reproducibility).
  static MLPHead make(const std::vector<int> &sizes, Act act = Act::Tanh,
                      std::uint64_t seed = 1);

  double energy(const Eigen::VectorXd &D) const;
  Eigen::VectorXd grad(const Eigen::VectorXd &D) const;

  std::size_t param_count() const;
  void gather_params(Eigen::VectorXd &dst, std::size_t off) const;
  void scatter_params(const Eigen::VectorXd &src, std::size_t off);
  Eigen::VectorXd all_values() const;
  void set_all_values(const Eigen::VectorXd &v);

  std::string type_tag() const { return "mlp"; }
  std::vector<int> architecture() const;

private:
  Eigen::VectorXd activate(const Eigen::VectorXd &z) const;
  Eigen::VectorXd activate_deriv(const Eigen::VectorXd &z) const;
};

class EnergyHead : private detail::ErasedValue<detail::HeadConcept> {
  template <typename T> struct Model final : detail::HeadConcept {
    T impl_;
    explicit Model(T t) : impl_(std::move(t)) {}
    constexpr double energy(const Eigen::VectorXd &D) const override {
      return impl_.energy(D);
    }
    constexpr Eigen::VectorXd grad(const Eigen::VectorXd &D) const override {
      return impl_.grad(D);
    }
    constexpr std::size_t param_count() const override {
      return impl_.param_count();
    }
    constexpr void gather_params(Eigen::VectorXd &x,
                                 std::size_t off) const override {
      impl_.gather_params(x, off);
    }
    constexpr void scatter_params(const Eigen::VectorXd &x,
                                  std::size_t off) override {
      impl_.scatter_params(x, off);
    }
    constexpr std::string type_tag() const override { return impl_.type_tag(); }
    constexpr std::vector<int> architecture() const override {
      return impl_.architecture();
    }
    constexpr Eigen::VectorXd all_values() const override {
      return impl_.all_values();
    }
    constexpr void set_all_values(const Eigen::VectorXd &v) override {
      impl_.set_all_values(v);
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

  constexpr double energy(const Eigen::VectorXd &D) const {
    return self_->energy(D);
  }
  constexpr Eigen::VectorXd grad(const Eigen::VectorXd &D) const {
    return self_->grad(D);
  }
  constexpr std::size_t param_count() const { return self_->param_count(); }
  constexpr void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  constexpr void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
  constexpr std::string type_tag() const { return self_->type_tag(); }
  constexpr std::vector<int> architecture() const {
    return self_->architecture();
  }
  constexpr Eigen::VectorXd all_values() const { return self_->all_values(); }
  constexpr void set_all_values(const Eigen::VectorXd &v) {
    self_->set_all_values(v);
  }
};

template <typename Derived>
struct MLBase : ForceCalculatorBase<Derived>, NoGlobals {
  // One head per element type, indexed by atom.type.index. The descriptor (and
  // its hyperparameters) lives in Derived.
  TypeArray<EnergyHead> heads;

  // ── Fit-time descriptor cache
  // During an optimization the
  // training geometry is FIXED — only the head params move — so the descriptor
  // D_i and its position-gradient dD_i/dr are constant. prepare() computes them
  // ONCE for every atom of every config; thereafter the per-iteration force
  // evaluation is cheap head algebra over cached arrays (no neighbor rebuild,
  // no get_descriptor, no finite-difference fallback).
  //
  // Stored POINTER-FREE: each cached atom keeps its neighbors as flat atom
  // indices (within the config) plus the bond vectors, so the cached force
  // assembly needs no live neighbor list. That is what lets the parallel
  // Jacobian (PotfitFunctor::df) accumulate forces into private per-column
  // buffers with zero shared mutable state.
  struct AtomCache {
    int type = 0;                           // element slot → head index
    Eigen::VectorXd values;                 // D_i  (descriptor_size)
    DescriptorGrad grad_self;               // dD_i/dr_i           (S×3)
    std::vector<DescriptorGrad> grad_neigh; // dD_i/dr_j per neighbor (S×3)
    std::vector<std::size_t> neigh_idx;     // neighbor atom index in the config
    std::vector<Vec3> neigh_dist;           // bond r_j − r_i (for the virial)
  };
  struct CacheData {
    std::vector<std::vector<AtomCache>> rows; // [config][atom]
    std::vector<double> volume;               // per-config cell volume
  };
  // shared_ptr<const>: copying the model for a parallel Jacobian column copies
  // the pointer (cheap, read-only sharing is thread-safe), never the data.
  mutable std::shared_ptr<const CacheData> cache_;

  void invalidate_cache() const { cache_.reset(); }
  [[nodiscard]] bool has_cache() const { return static_cast<bool>(cache_); }

  // One-time precompute over the whole training set. Builds neighbor lists,
  // warms up any lazy descriptor state (e.g. SOAP's radial basis)
  // single-threaded, then fills the cache with one balanced parallel_for over
  // the flat (config, atom) set — thousands of independent units, so it fills
  // every core instead of stranding threads on a handful of uneven configs.
  void prepare(std::span<Configuration> configs) const {
    auto data = std::make_shared<CacheData>();
    data->rows.resize(configs.size());
    data->volume.resize(configs.size());
    for (std::size_t c = 0; c < configs.size(); ++c) {
      build_neighbor_list(configs[c], max_cutoff());
      data->rows[c].resize(configs[c].atoms.size());
      data->volume[c] = bc_volume(configs[c].bc);
    }
    // Warm-up: force any lazy, model-internal init exactly once, serially, so
    // the parallel fill below never races on it.
    for (auto &cfg : configs) {
      if (!cfg.atoms.empty()) {
        (void)self().get_descriptor(cfg.atoms[0]);
        break;
      }
    }
    std::vector<std::pair<std::size_t, std::size_t>> work;
    for (std::size_t c = 0; c < configs.size(); ++c) {
      for (std::size_t a = 0; a < configs[c].atoms.size(); ++a) {
        work.emplace_back(c, a);
      }
    }
    std::for_each(std::execution::par, work.begin(), work.end(),
                  [&](const std::pair<std::size_t, std::size_t> &ca) {
                    fill_atom_cache(configs[ca.first], ca.second,
                                    data->rows[ca.first][ca.second]);
                  });
    cache_ = std::move(data);
  }

  void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                   double &energy, SymTens &stress) const {
    const std::shared_ptr<const CacheData> cache = cache_;
    const auto &rows = cache->rows[cache_index];
    energy = 0.0;
    stress = SymTens::Zero();
    std::ranges::for_each(forces, [](Vec3& f) { f.setZero(); });

    for (const AtomCache &cc : rows) {
      const EnergyHead &h = heads[static_cast<std::size_t>(cc.type)];
      energy += h.energy(cc.values);
      const Eigen::VectorXd dEdD = h.grad(cc.values);
      const std::size_t a = static_cast<std::size_t>(&cc - rows.data());
      forces[a] -= cc.grad_self.transpose() * dEdD;
      for (std::size_t jj = 0; jj < cc.neigh_idx.size(); ++jj) {
        const Vec3 f_on_j = -(cc.grad_neigh[jj].transpose() * dEdD);
        forces[cc.neigh_idx[jj]] += f_on_j;
        stress += cc.neigh_dist[jj] * f_on_j.transpose();
      }
    }
    stress /= cache->volume[cache_index];
  }

  // Full path (evaluate / predict mode): geometry may differ from the cached
  // training set, so always recompute. Used by force::evaluate and the
  // start/final RMSE checks, keeping those honest independent of the cache. Per
  // atom it gets the descriptor + its position-gradient (analytic when the
  // model supplies it, else a cheap O(neighbors) finite-difference of the
  // descriptor — NOT the old O(N²) finite-difference of the total energy) and
  // assembles forces directly.
  void eval_forces(Configuration &cfg) const {
    build_neighbor_list(cfg, max_cutoff());

    cfg.calc_energy = 0.0;
    cfg.calc_stress = SymTens::Zero();
    for (auto &a : cfg.atoms) {
      a.calc_force = Vec3::Zero();
    }

    for (auto &ai : cfg.atoms) {
      const DescriptorValue d = descriptor_with_grad(ai);
      const EnergyHead &h = heads[ai.type.index];
      cfg.calc_energy += h.energy(d.values);
      accumulate_atom_forces(cfg, ai, d, h.grad(d.values));
    }

    cfg.calc_stress /= bc_volume(cfg.bc); // virial → stress (per unit volume)

    events::on_force_eval(
        events::ForceEvalStats{this->conf_index, force_rms(cfg), cfg});
  }

  // Fit path: identical results to eval_forces(cfg) but served from the cache
  // for config `cache_index`. Falls back to the full recompute if no cache is
  // live.
  void eval_forces(Configuration &cfg, std::size_t cache_index) const {
    if (!cache_ || cache_index >= cache_->rows.size()) {
      eval_forces(cfg);
      return;
    }
    std::vector<Vec3> f(cfg.atoms.size());
    double e = 0.0;
    SymTens s = SymTens::Zero();
    eval_cached(cache_index, f, e, s);
    for (std::size_t a = 0; a < cfg.atoms.size(); ++a) {
      cfg.atoms[a].calc_force = f[a];
    }
    cfg.calc_energy = e;
    cfg.calc_stress = s;
    cfg.calc_limit = 0.0; // ML models have no embedding range penalty
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

  // Descriptor value + position-gradient for one atom: analytic when the model
  // supplies it (has_grad), else a cheap O(neighbors) finite-difference of the
  // descriptor. The descriptor of atom i depends on geometry only through each
  // neighbor's displacement, so dD_i/d(r_j) is obtained by perturbing that
  // neighbor's bond vector IN PLACE — no neighbor-list rebuild — and
  // dD_i/dr_i = −Σ_j dD_i/dr_j (translation invariance). `atom` is mutated only
  // transiently (each dist restored); callers must hold exclusive access to
  // atom's own neighbor list (true in the serial eval_forces and in prepare's
  // per-atom parallel tasks).
  DescriptorValue descriptor_with_grad(Atom &atom) const {
    DescriptorValue d = self().get_descriptor(atom);
    if (self().analytic_grads() && d.has_grad) {
      return d;
    }
    const Eigen::Index S = d.values.size();
    const std::size_t nn = atom.neighbors.size();
    d.grad_neigh.assign(nn, DescriptorGrad::Zero(S, 3));
    d.grad_self = DescriptorGrad::Zero(S, 3);
    constexpr double h = 1e-4;
    for (std::size_t jj = 0; jj < nn; ++jj) {
      DescriptorGrad g(S, 3);
      for (int cdim = 0; cdim < 3; ++cdim) {
        const double x0 = atom.neighbors[jj].dist[cdim];
        atom.neighbors[jj].dist[cdim] = x0 + h;
        const Eigen::VectorXd dp = self().get_descriptor(atom).values;
        atom.neighbors[jj].dist[cdim] = x0 - h;
        const Eigen::VectorXd dm = self().get_descriptor(atom).values;
        atom.neighbors[jj].dist[cdim] = x0;
        g.col(cdim) = (dp - dm) / (2.0 * h);
      }
      d.grad_neigh[jj] = g;
      d.grad_self -= g;
    }
    d.has_grad = true;
    return d;
  }

  // Fill one atom's cache row, reusing descriptor_with_grad for the value +
  // gradient and recording the pointer-free neighbor indices / bond vectors
  // used by eval_cached. Each call touches only `atom`'s own neighbor list, so
  // the parallel fill is race-free.
  void fill_atom_cache(Configuration &cfg, std::size_t a, AtomCache &cc) const {
    Atom &atom = cfg.atoms[a];
    cc.type = static_cast<int>(atom.type.index);
    const DescriptorValue d = descriptor_with_grad(atom);
    cc.values = d.values;
    cc.grad_self = d.grad_self;
    cc.grad_neigh = d.grad_neigh;
    const std::size_t nn = atom.neighbors.size();
    cc.neigh_idx.resize(nn);
    cc.neigh_dist.resize(nn);
    for (std::size_t jj = 0; jj < nn; ++jj) {
      cc.neigh_idx[jj] = static_cast<std::size_t>(atom.neighbors[jj].neighbor -
                                                  cfg.atoms.data());
      cc.neigh_dist[jj] = atom.neighbors[jj].dist;
    }
  }

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
};

} // namespace potfit
