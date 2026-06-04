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
#include <ranges>
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
  // Analytic parameter derivatives, in gather_params() order, used to build the
  // optimizer Jacobian without finite-differencing the residual over every
  // weight. has_param_jacobian()==false ⇒ the functor falls back to the FD
  // path.
  //   param_grad(D)   = ∂E/∂θ            (length param_count())
  //   dgrad_dparam(D) = ∂(∂E/∂D)/∂θ      (param_count() columns, one per param;
  //                                       rows = descriptor size) — the mixed
  //                     second derivative the force-residual columns need.
  virtual bool has_param_jacobian() const = 0;
  virtual Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const = 0;
  virtual Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const = 0;
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
  std::ranges::transform(f, v.data(), [](const auto *p) { return p->value; });
  return v;
}

template <typename Derived>
void HeadParams<Derived>::set_all_values(const Eigen::VectorXd &v) {
  auto f = self().field_ptrs();
  assert(v.size() == static_cast<Eigen::Index>(f.size()));
  for (auto [field, value] : std::views::zip(
           f, std::span(v.data(), static_cast<std::size_t>(v.size())))) {
    field->value = value;
  }
}

// Linear head: E_i = Σ_k coeffs_k · D_k + bias.  de/dD = coeffs (constant), so
// forces reduce to coeffs · dD/dr.
struct LinearHead : HeadParams<LinearHead> {
  std::vector<Param> coeffs;
  Param bias{0.0, true};

  double energy(const Eigen::VectorXd &D) const;
  Eigen::VectorXd grad(const Eigen::VectorXd &) const;
  // E = c·D + bias ⇒ ∂E/∂θ = D (per free coeff), and ∂(∂E/∂D)/∂θ = I (the
  // descriptor-gradient *is* the coeffs), so the force-Jacobian column for
  // coeff k is just −dD/dr column k. Exact and trivial.
  bool has_param_jacobian() const { return true; }
  Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const;
  Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const;
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
  // param_grad = ∂E/∂θ (backprop-to-weights). dgrad_dparam = ∂(∂E/∂D)/∂θ, the
  // mixed second derivative, computed by reverse-mode AD of u·(∂E/∂D) through
  // the augmented forward+backward graph (one sweep per descriptor component).
  bool has_param_jacobian() const { return true; }
  Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const;
  Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const;

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
  Eigen::VectorXd activate_deriv2(const Eigen::VectorXd &z) const; // σ″
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
    bool has_param_jacobian() const override {
      return impl_.has_param_jacobian();
    }
    Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const override {
      return impl_.param_grad(D);
    }
    Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const override {
      return impl_.dgrad_dparam(D);
    }
    std::size_t param_count() const override { return impl_.param_count(); }
    void gather_params(Eigen::VectorXd &x, std::size_t off) const override {
      impl_.gather_params(x, off);
    }
    void scatter_params(const Eigen::VectorXd &x, std::size_t off) override {
      impl_.scatter_params(x, off);
    }
    std::string type_tag() const override { return impl_.type_tag(); }
    std::vector<int> architecture() const override {
      return impl_.architecture();
    }
    Eigen::VectorXd all_values() const override { return impl_.all_values(); }
    void set_all_values(const Eigen::VectorXd &v) override {
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

  double energy(const Eigen::VectorXd &D) const { return self_->energy(D); }
  Eigen::VectorXd grad(const Eigen::VectorXd &D) const {
    return self_->grad(D);
  }
  bool has_param_jacobian() const { return self_->has_param_jacobian(); }
  Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const {
    return self_->param_grad(D);
  }
  Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const {
    return self_->dgrad_dparam(D);
  }
  std::size_t param_count() const { return self_->param_count(); }
  void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
  std::string type_tag() const { return self_->type_tag(); }
  std::vector<int> architecture() const { return self_->architecture(); }
  Eigen::VectorXd all_values() const { return self_->all_values(); }
  void set_all_values(const Eigen::VectorXd &v) { self_->set_all_values(v); }
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

  // True iff every head can supply its analytic parameter-Jacobian, so the
  // functor can build ∂F/∂θ directly instead of finite-differencing the
  // residual over each weight. Mixed head types per element ⇒ require ALL of
  // them.
  [[nodiscard]] bool has_param_jacobian() const {
    return heads.size() > 0 &&
           std::ranges::all_of(heads, [](const EnergyHead &h) {
             return h.has_param_jacobian();
           });
  }

  // ── Per-feature descriptor standardization ─────────────────────────────────
  // Least-squares/NN conditioning collapses when descriptor features span
  // orders of magnitude (SOAP only L2-normalizes per sample; G2 not at all), so
  // the LM fit plateaus after one step. prepare() whitens each feature to ≈zero
  // mean / unit variance over the training set, per element type. The affine
  // map D' = (D−μ)/σ is folded into the cached values AND their
  // position-gradients (each gradient row k scaled by 1/σ_k; μ is a constant
  // shift that drops out of the gradient), so eval_cached and the uncached
  // force path are unchanged apart from receiving whitened inputs. μ/σ are
  // FIXED transforms — not optimizer parameters — so they live outside
  // gather/scatter and are written with the model. Public so the io layer can
  // (de)serialize them and tests can assert.
  bool standardize_features = false;
  mutable TypeArray<Eigen::VectorXd> mean_; // per type, length descriptor_size
  mutable TypeArray<Eigen::VectorXd>
      inv_std_; // per type, 1/σ (0 for dead feats)

  [[nodiscard]] bool has_standardization() const {
    return standardize_features && inv_std_.size() > 0;
  }

  // One-time precompute over the whole training set. Builds neighbor lists,
  // warms up any lazy descriptor state (e.g. SOAP's radial basis)
  // single-threaded, then fills the cache with one balanced parallel_for over
  // the flat (config, atom) set — thousands of independent units, so it fills
  // every core instead of stranding threads on a handful of uneven configs.
  void prepare(std::span<Configuration> configs) const {
    auto data = std::make_shared<CacheData>();
    data->rows.resize(configs.size());
    data->volume.resize(configs.size());
    for (auto [cfg, row, vol] :
         std::views::zip(configs, data->rows, data->volume)) {
      build_neighbor_list(cfg, max_cutoff());
      row.resize(cfg.atoms.size());
      vol = bc_volume(cfg.bc);
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
    // Whiten the just-filled raw descriptors in place: derive per-type μ/σ from
    // the full training set, then standardize every cached value + gradient.
    // Same parallel work-list (each task still touches only its own AtomCache).
    if (standardize_features) {
      compute_standardization(*data);
      std::for_each(std::execution::par, work.begin(), work.end(),
                    [&](const std::pair<std::size_t, std::size_t> &ca) {
                      AtomCache &cc = data->rows[ca.first][ca.second];
                      standardize_in_place(cc.type, cc.values, cc.grad_self,
                                           cc.grad_neigh);
                    });
    }
    cache_ = std::move(data);
  }

  void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                   double &energy, SymTens &stress) const {
    const std::shared_ptr<const CacheData> cache = cache_;
    const auto &rows = cache->rows[cache_index];
    energy = 0.0;
    stress = SymTens::Zero();
    std::ranges::for_each(forces, [](Vec3 &f) { f.setZero(); });

    // rows is in atom order, so it pairs elementwise with the force span.
    for (auto [cc, fi] : std::views::zip(rows, forces)) {
      const EnergyHead &h = heads[static_cast<std::size_t>(cc.type)];
      energy += h.energy(cc.values);
      const Eigen::VectorXd dEdD = h.grad(cc.values);
      fi -= cc.grad_self.transpose() * dEdD;
      for (const auto &[gn, nidx, nd] :
           std::views::zip(cc.grad_neigh, cc.neigh_idx, cc.neigh_dist)) {
        const Vec3 f_on_j = -(gn.transpose() * dEdD);
        forces[nidx] += f_on_j;
        stress += nd * f_on_j.transpose();
      }
    }
    stress /= cache->volume[cache_index];
  }

  // Analytic ∂(residuals of config c)/∂θ, written into `fjac` at the config's
  // row block (forces, energy, optional stress; limit row stays 0). The
  // per-head parameter derivatives (param_grad = ∂E/∂θ, dgrad_dparam =
  // ∂(∂E/∂D)/∂θ) are contracted against the SAME cached spatial gradients
  // eval_cached uses, so the Jacobian is exact and consistent with the residual
  // — no descriptor recompute, no per-parameter finite difference. `col_off[t]`
  // is the first fjac column of head t (column analogue of the residual
  // row_offset). A config's rows are disjoint across configs, so a
  // config-parallel caller needs no locking.
  void eval_cached_jacobian(std::size_t cache_index, int row0,
                            const std::vector<int> &col_off,
                            double energy_weight, double stress_weight,
                            Eigen::MatrixXd &fjac) const {
    const std::shared_ptr<const CacheData> cache = cache_;
    const auto &rows = cache->rows[cache_index];
    const int na = static_cast<int>(rows.size());
    const int erow = row0 + 3 * na; // energy residual row
    const int srow = erow + 1;      // first stress row (if any)
    const double inv_vol = 1.0 / cache->volume[cache_index];
    // Stress component order matches eval_into: (00,11,22,01,02,12).
    static constexpr int sp[6] = {0, 1, 2, 0, 0, 1};
    static constexpr int sq[6] = {0, 1, 2, 1, 2, 2};

    for (int i = 0; i < na; ++i) {
      const AtomCache &cc = rows[i];
      const auto t = static_cast<std::size_t>(cc.type);
      const EnergyHead &h = heads[t];
      const int c0 = col_off[t];
      const int pc = col_off[t + 1] - c0; // head param count
      if (pc == 0) {
        continue;
      }

      // Energy column block: ∂E_i/∂θ into the energy row (weighted).
      const Eigen::VectorXd gE = h.param_grad(cc.values);
      fjac.block(erow, c0, 1, pc).noalias() += energy_weight * gE.transpose();

      // M_i = ∂(∂E_i/∂D)/∂θ   (S × pc) — the mixed second derivative.
      const Eigen::MatrixXd M = h.dgrad_dparam(cc.values);

      // Self force rows: ∂f_i/∂θ = −grad_self^T · M   (3 × pc).
      fjac.block(row0 + 3 * i, c0, 3, pc).noalias() -=
          cc.grad_self.transpose() * M;

      // Neighbor force rows (and stress): ∂f_on_j/∂θ = −grad_neigh^T · M.
      for (const auto &[gn, nidx, nd] :
           std::views::zip(cc.grad_neigh, cc.neigh_idx, cc.neigh_dist)) {
        const Eigen::MatrixXd fblk = -(gn.transpose() * M); // 3 × pc
        fjac.block(row0 + 3 * static_cast<int>(nidx), c0, 3, pc) += fblk;
        if (stress_weight > 0.0) {
          for (int s = 0; s < 6; ++s) {
            fjac.block(srow + s, c0, 1, pc) +=
                (stress_weight * inv_vol * nd[sp[s]]) * fblk.row(sq[s]);
          }
        }
      }
    }
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
      DescriptorValue d = descriptor_with_grad(ai);
      standardize_in_place(static_cast<int>(ai.type.index), d);
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

  constexpr std::size_t param_count() const {
    return std::ranges::fold_left(
        heads, std::size_t{0},
        [](std::size_t n, const auto &h) { return n + h.param_count(); });
  }

  void gather_params(Eigen::VectorXd &dst, std::size_t off) const {
    std::for_each(heads.begin(), heads.end(), [&](const auto &h) {
      h.gather_params(dst, off);
      off += h.param_count();
    });
  }

  void scatter_params(const Eigen::VectorXd &src, std::size_t off) {
    for (auto &h : heads) {
      h.scatter_params(src, off);
      off += h.param_count();
    }
  }

  constexpr double max_cutoff() const { return self().descriptor_cutoff(); }

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
    d.grad_neigh.assign(atom.neighbors.size(), DescriptorGrad::Zero(S, 3));
    d.grad_self = DescriptorGrad::Zero(S, 3);
    constexpr double h = 1e-4;
    for (auto [nb, gn] : std::views::zip(atom.neighbors, d.grad_neigh)) {
      DescriptorGrad g(S, 3);
      for (int cdim = 0; cdim < 3; ++cdim) {
        const double x0 = nb.dist[cdim];
        nb.dist[cdim] = x0 + h;
        const Eigen::VectorXd dp = self().get_descriptor(atom).values;
        nb.dist[cdim] = x0 - h;
        const Eigen::VectorXd dm = self().get_descriptor(atom).values;
        nb.dist[cdim] = x0;
        g.col(cdim) = (dp - dm) / (2.0 * h);
      }
      gn = g;
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
    cc.neigh_idx.resize(atom.neighbors.size());
    cc.neigh_dist.resize(atom.neighbors.size());
    for (auto [nb, nidx, nd] :
         std::views::zip(atom.neighbors, cc.neigh_idx, cc.neigh_dist)) {
      nidx = static_cast<std::size_t>(nb.neighbor - cfg.atoms.data());
      nd = nb.dist;
    }
  }

  // Forces from atom i's descriptor: F_m = −Σ_i (de/dD_i)·(dD_i/dr_m).
  // dD_i/dr_m is nonzero only for m=i (grad_self) and m∈neighbors(i).
  void accumulate_atom_forces(Configuration &cfg, Atom &ai,
                              const DescriptorValue &d,
                              const Eigen::VectorXd &dEdD) const {
    ai.calc_force -= d.grad_self.transpose() * dEdD;
    for (const auto &[nb, gn] : std::views::zip(ai.neighbors, d.grad_neigh)) {
      Atom &aj = const_cast<Atom &>(*nb.neighbor);
      const Vec3 f_on_j = -(gn.transpose() * dEdD);
      aj.calc_force += f_on_j;
      // Virial: bond ⊗ force-on-partner (matches Tersoff/EAM convention).
      cfg.calc_stress += nb.dist * f_on_j.transpose();
    }
  }

  // Apply the per-type whitening D'=(D−μ)/σ to a descriptor value and its
  // position-gradients in place. Each gradient row k scales by inv_std[k]; μ is
  // a constant shift and drops out. No-op until prepare() has computed the
  // stats (or when disabled / a type has no training atoms), so the pre-fit
  // baseline stays on raw descriptors.
  void standardize_in_place(int type, Eigen::VectorXd &values,
                            DescriptorGrad &grad_self,
                            std::vector<DescriptorGrad> &grad_neigh) const {
    if (!has_standardization()) {
      return;
    }
    const auto t = static_cast<std::size_t>(type);
    const Eigen::VectorXd &mu = mean_[t];
    const Eigen::VectorXd &iv = inv_std_[t];
    if (mu.size() != values.size()) {
      return; // type had no training atoms → identity
    }
    values = (values - mu).cwiseProduct(iv);
    grad_self = iv.asDiagonal() * grad_self;
    for (auto &g : grad_neigh) {
      g = iv.asDiagonal() * g;
    }
  }
  void standardize_in_place(int type, DescriptorValue &d) const {
    standardize_in_place(type, d.values, d.grad_self, d.grad_neigh);
  }

  // Population mean/variance of each descriptor feature over all cached atoms
  // of each element type → mean_/inv_std_. Near-constant features (var ≤ eps)
  // get inv_std 0, mapping them to a constant-0 input with zero gradient
  // (dropped), which avoids amplifying noise on dead channels.
  void compute_standardization(const CacheData &data) const {
    Eigen::Index S = 0;
    for (const auto &row : data.rows) {
      if (!row.empty()) {
        S = row.front().values.size();
        break;
      }
    }
    const std::size_t T = static_cast<std::size_t>(this->ntypes);
    if (S == 0 || T == 0) {
      return; // nothing to standardize
    }
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(S);
    std::vector<Eigen::VectorXd> sum(T, zero);
    std::vector<Eigen::VectorXd> sumsq(T, zero);
    std::vector<std::size_t> count(T, 0);
    for (const auto &row : data.rows) {
      for (const AtomCache &cc : row) {
        const auto t = static_cast<std::size_t>(cc.type);
        sum[t] += cc.values;
        sumsq[t] += cc.values.cwiseProduct(cc.values);
        ++count[t];
      }
    }
    constexpr double eps = 1e-12;
    TypeArray<Eigen::VectorXd> mean, inv_std;
    mean.reserve(T);
    inv_std.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
      if (count[t] == 0) {
        mean.emplace_back(Eigen::VectorXd{}); // empty → identity for this type
        inv_std.emplace_back(Eigen::VectorXd{});
        continue;
      }
      const double n = static_cast<double>(count[t]);
      Eigen::VectorXd mu = sum[t] / n;
      const Eigen::VectorXd var = (sumsq[t] / n) - mu.cwiseProduct(mu);
      Eigen::VectorXd iv(S);
      for (Eigen::Index k = 0; k < S; ++k) {
        iv[k] = var[k] > eps ? 1.0 / std::sqrt(var[k]) : 0.0;
      }
      mean.emplace_back(std::move(mu));
      inv_std.emplace_back(std::move(iv));
    }
    mean_ = std::move(mean);
    inv_std_ = std::move(inv_std);
  }
};

} // namespace potfit
