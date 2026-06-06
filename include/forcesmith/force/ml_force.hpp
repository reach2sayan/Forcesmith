#pragma once

// Machine-learning potential abstraction.
//
// Idea: a configuration's local atomic environments reduce to a per-atom vector
// of DESCRIPTORS D_i; the per-atom energy is E_i = head(D_i) for a fittable
// "head" (the coefficients the optimizer drives). Forces are −dE/dr, which need
// the descriptor gradients dD_i/dr.
//
// Layering (mirrors the rest of the force layer):
//   * MLBase<Derived> is a CRTP ForceCalculator base. Its eval_forces is
//     descriptor-agnostic: it calls the concrete model's get_descriptor hook,
//     feeds the descriptor to the head, and assembles energy/forces/stress.
//   * Derived (e.g. ACSF) supplies the descriptor:
//       DescriptorValue get_descriptor(const Atom&) const;
//       double          descriptor_cutoff() const;
//       bool            analytic_grads() const;
//   * The head is value-erased (EnergyHead, like Potential) so swapping the
//     descriptor→energy map never multiplies the ForceCalculator variant.

// The fittable parameters are the head coefficients (one head per element
// type); descriptor hyperparameters (η, Rs, cutoff …) are fixed.

#include "forcesmith/core/atom.hpp"
#include "forcesmith/core/erased.hpp"
#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/neighbor_list.hpp" // build_neighbor_list, bc_volume
#include "forcesmith/core/param.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/force/force_calculator_concept.hpp"
#include "forcesmith/force/potential_table.hpp" // TypeArray

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace forcesmith {

// Per-neighbour descriptor gradient block: row k is dD_k/dr (a 3-vector).
using DescriptorGrad = Eigen::Matrix<double, Eigen::Dynamic, 3>;
struct DescriptorValue {
  Eigen::VectorXd values; // D_i, length = descriptor_size
  bool has_grad = false;

  // only use when has_grad == true
  DescriptorGrad grad_self;               // dD_i/dr_i (S×3)
  std::vector<DescriptorGrad> grad_neigh; // dD_i/dr_j per neighbor jj
};

namespace detail {
// Inherits the optimizer param-plumbing virtuals (param_count, gather_params,
// scatter_params, gather_bounds) from FittableConcept (core/fit_params.hpp).
struct HeadConcept : FittableConcept {
  virtual double energy(const Eigen::VectorXd &D) const = 0;

  virtual Eigen::VectorXd grad(const Eigen::VectorXd &D) const = 0; // de/dD
  virtual bool
  constant_grad() const = 0; // ∂E/∂D and ∂(∂E/∂D)/∂θ const per elem

  virtual bool has_param_jacobian() const = 0;
  virtual Eigen::VectorXd
  param_grad(const Eigen::VectorXd &D) const = 0; // ∂E/∂θ  (1 x num_params)
  virtual Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D)
      const = 0; // ∂(∂E/∂D)/∂θ  (num_descritptor x num_params)

  // Generic (de)serialization surface so the io layer can round-trip any head
  // without knowing its concrete type (keeps nlohmann out of this header).
  //   type_tag()     — "linear" …
  //   architecture() — shape ints; linear: {n_coeffs}
  //   all_values()   — every parameter (free AND fixed), flat
  virtual std::string type_tag() const = 0;
  virtual std::vector<int> architecture() const = 0;

  virtual Eigen::VectorXd all_values() const = 0;
  virtual void set_all_values(const Eigen::VectorXd &v) = 0;

  virtual std::unique_ptr<HeadConcept> clone() const = 0;
  // Re-rank support (species count change). remapped() re-lays-out the head to
  // map.size() descriptor features, pulling coeff k from old index map[k]
  // (nullopt ⇒ 0), preserving the bias and per-coeff fixed flags. zero_like()
  // builds a fresh zero head of the same type sized to `n` features, for an
  // element that did not exist before the re-rank.
  virtual std::unique_ptr<HeadConcept>
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const = 0;
  virtual std::unique_ptr<HeadConcept> zero_like(Eigen::Index n) const = 0;
};
} // namespace detail

template <typename Derived> struct HeadParams : ParamSet<HeadParams<Derived>> {
  // ParamSet customization point: a head stores its coefficients as scattered
  // Param objects, so expose them as a Param& range over field_ptrs(). The
  // optimizer plumbing (param_count/gather/scatter/gather_bounds) is generated
  // from this by ParamSet.
  auto param_fields() {
    return self().field_ptrs() |
           std::views::transform([](Param *p) -> Param & { return *p; });
  }
  auto param_fields() const {
    return self().field_ptrs() | std::views::transform(
                                     [](const Param *p) -> const Param & {
                                       return *p;
                                     });
  }

  // Default (de)serialization over field_ptrs() — ALL params, free and fixed.
  // Derived must still supply type_tag() and architecture(). Heads that do not
  // store their parameters as Param* may override these.
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
  // Linear: de/dD = coeffs, independent of D — enables the batched cached path.
  constexpr bool constant_grad() const { return true; }
  // E = c·D + bias ⇒ ∂E/∂θ = D (per free coeff), and ∂(∂E/∂D)/∂θ = I (the
  // descriptor-gradient *is* the coeffs), so the force-Jacobian column for
  // coeff k is just −dD/dr column k. Exact and trivial.
  bool has_param_jacobian() const { return true; }
  Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const;
  Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const;
  std::vector<Param *> field_ptrs();
  std::vector<const Param *> field_ptrs() const;

  // Re-rank: copy coeff k from old index map[k] (nullopt ⇒ 0.0), keeping bias
  // and the coeffs' fixed flag. zero_like(n): n zero (free) coeffs + default
  // bias, for a newly-added element.
  [[nodiscard]] LinearHead
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const;
  [[nodiscard]] LinearHead zero_like(Eigen::Index n) const;

  // all_values()/set_all_values() inherited from HeadParams emit [coeffs…,
  // bias].
  constexpr std::string type_tag() const { return "linear"; }
  constexpr std::vector<int> architecture() const {
    return {static_cast<int>(coeffs.size())};
  }
};

class EnergyHead : private detail::ErasedValue<detail::HeadConcept> {
  template <typename T>
  struct Model final : detail::FittableModel<T, detail::HeadConcept> {
    using Base = detail::FittableModel<T, detail::HeadConcept>;
    using Base::Base;     // inherit the impl_-forwarding constructor
    using Base::impl_;    // bring impl_ into scope for the bodies below
    double energy(const Eigen::VectorXd &D) const override {
      return impl_.energy(D);
    }
    Eigen::VectorXd grad(const Eigen::VectorXd &D) const override {
      return impl_.grad(D);
    }
    bool constant_grad() const override { return impl_.constant_grad(); }
    bool has_param_jacobian() const override {
      return impl_.has_param_jacobian();
    }
    Eigen::VectorXd param_grad(const Eigen::VectorXd &D) const override {
      return impl_.param_grad(D);
    }
    Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd &D) const override {
      return impl_.dgrad_dparam(D);
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
    std::unique_ptr<detail::HeadConcept> remapped(
        const std::vector<std::optional<Eigen::Index>> &map) const override {
      return std::make_unique<Model>(impl_.remapped(map));
    }
    std::unique_ptr<detail::HeadConcept>
    zero_like(Eigen::Index n) const override {
      return std::make_unique<Model>(impl_.zero_like(n));
    }
  };

  using Base = detail::ErasedValue<detail::HeadConcept>;

  // Wrap an already-built concept (used by remapped()/zero_like()).
  explicit EnergyHead(std::unique_ptr<detail::HeadConcept> p)
      : Base(std::move(p)) {}

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
  bool constant_grad() const { return self_->constant_grad(); }
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
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    self_->gather_bounds(lo, hi, off);
  }
  std::string type_tag() const { return self_->type_tag(); }
  std::vector<int> architecture() const { return self_->architecture(); }
  Eigen::VectorXd all_values() const { return self_->all_values(); }
  void set_all_values(const Eigen::VectorXd &v) { self_->set_all_values(v); }
  [[nodiscard]] EnergyHead
  remapped(const std::vector<std::optional<Eigen::Index>> &map) const {
    return EnergyHead(self_->remapped(map));
  }
  [[nodiscard]] EnergyHead zero_like(Eigen::Index n) const {
    return EnergyHead(self_->zero_like(n));
  }
};

template <typename Derived>
struct MLBase : ForceCalculatorBase<Derived>, NoGlobals {
  // One head per element type, indexed by atom.type.index.
  TypeArray<EnergyHead> heads;

  // Fit-time descriptor cache - D_i and its dD_i/dr are constant
  // During a fit (FIXED geometry) descriptor.prepare() precomputes per atom.

  // Stored POINTER-FREE: each cached atom keeps its neighbours as flat atom
  // indices (within the config) plus the bond vectors, so the cached force
  // assembly needs no live neighbour list. That is what lets the parallel
  // Jacobian (ForcesmithFunctor::df) accumulate forces into private per-column
  // buffers with zero shared mutable state.
  struct AtomCache {
    int type = 0;           // element slot → head index
    Eigen::VectorXd values; // D_i  (descriptor_size S)

    // Every position-gradient block of D_i stacked COLUMN-WISE into one
    // contiguous S×(3·(1+nbonds)) matrix: columns [0,3) are dD_i/dr_i (the
    // centre), columns [3·(k+1), 3·(k+1)+3) are dD_i/dr_j for neighbour k
    // (parallel to neigh_idx/neigh_dist). Stacking lets eval_cached and the
    // Jacobian contract ALL blocks against ∂E/∂D in a single batched BLAS call
    // — replacing the per-block 3×S gemv flood that dominated the SOAP profile.
    Eigen::MatrixXd grad_all;
    std::vector<std::size_t> neigh_idx; // neighbor atom index in the config
    std::vector<Vec3> neigh_dist;       // bond r_j − r_i (for the virial)
  };
  // Per-(config, element type) grouping of EVERY gradient block of that type's
  // atoms, concatenated column-wise. For a LINEAR head ∂E/∂D is one vector c_t
  // per type, so the whole group's forces are a SINGLE gemv `grad^T · c_t`
  // (and the Jacobian a single gemm `grad^T · M_t`) — one BLAS call per
  // (config,type) instead of one per atom. Built only when every head reports
  // constant_grad(); otherwise `groups` stays empty and eval falls back to the
  // per-atom AtomCache path.
  struct ForceGroup {
    int type = 0;
    Eigen::MatrixXd
        grad; // S × (3·nblocks): block b occupies columns [3b, 3b+3)
    std::vector<int> target; // [nblocks] atom of block b's forces
    std::vector<Vec3> bond;  // [nblocks] bond vector for the virial
  };
  struct CacheData {
    std::vector<std::vector<AtomCache>> rows; // [config][atom] (energy/values)
    std::vector<std::vector<ForceGroup>> groups;  // [config][type]
    std::vector<double> volume; // per-config cell volume
    // M_t = ∂(∂E/∂D)/∂θ per type (linear heads): constant across atoms AND
    // optimizer iterations, so precompute
    std::vector<Eigen::MatrixXd> dgrad_param;
  };
  mutable std::shared_ptr<const CacheData> cache_;
  void invalidate_cache() const { cache_.reset(); }
  [[nodiscard]] constexpr bool has_cache() const {
    return static_cast<bool>(cache_);
  }

  [[nodiscard]] bool has_param_jacobian() const {
    return heads.size() > 0 &&
           std::ranges::all_of(heads, [](const EnergyHead &h) {
             return h.has_param_jacobian();
           });
  }

  // Every head linear ⇒ enable the batched per-(config,type) cache fast path.
  [[nodiscard]] bool all_constant_grad() const {
    return heads.size() > 0 &&
           std::ranges::all_of(
               heads, [](const EnergyHead &h) { return h.constant_grad(); });
  }

  [[nodiscard]] boost::leaf::result<Derived>
  remap(const SpeciesRegistry &old_reg, const SpeciesRegistry &new_reg) const {
    Derived out = self();
    const std::vector<std::optional<Eigen::Index>> idx =
        self().descriptor_index_map(old_reg, new_reg);
    const std::vector<std::optional<std::size_t>> old_of_new =
        old_slot_of_new(old_reg, new_reg);
    const std::size_t S_new = forcesmith::ntypes(new_reg);
    out.ntypes = S_new;
    const auto new_size = static_cast<Eigen::Index>(out.descriptor_size());

    TypeArray<EnergyHead> new_heads;
    new_heads.reserve(S_new);
    for (std::size_t t = 0; t < S_new; ++t) {
      if (const auto s_old = old_of_new[t]) {
        new_heads.emplace_back(heads[*s_old].remapped(idx));
      } else {
        // heads is non-empty (a seeded model always has >=1 head); heads[0]
        // just supplies the concrete head TYPE for the fresh zero head.
        new_heads.emplace_back(heads[0].zero_like(new_size));
      }
    }
    out.heads = std::move(new_heads);

    out.mean_ = {};    // stale (old-layout); prepare() recomputes on next fit
    out.inv_std_ = {}; // (standardize_features flag is preserved by the copy)
    out.invalidate_cache();
    return out;
  }

  // Per-feature descriptor standardization
  // prepare() whitens each feature to ≈zero mean / unit variance over the
  // training set, per element type.
  // The affine map D' = (D−μ)/σ is folded into the cached values WITH
  // position-gradients (each gradient row k scaled by 1/σ_k; μ is a constant
  // shift that drops out of the gradient), so eval_cached and the uncached
  // force path are unchanged apart from receiving whitened inputs.
  // μ/σ are FIXED FIXED transforms
  bool standardize_features = false;
  mutable TypeArray<Eigen::VectorXd> mean_; // per type, length descriptor_size
  mutable TypeArray<Eigen::VectorXd> inv_std_; // per type, 1/σ (0 for dead)

  [[nodiscard]] bool has_standardization() const {
    return standardize_features && inv_std_.size() > 0;
  }

private:
  // ── prepare() pipeline stages
  // ─────────────────────────────────────────────── Declared before prepare()
  // so its (non-dependent) calls resolve here. Flat (config, atom) index list
  // over the whole training set.
  using Worklist = std::vector<std::pair<std::size_t, std::size_t>>;

  // Stage 0 — the work-list: thousands of independent (config, atom) units, so
  // one balanced parallel_for fills every core instead of stranding threads on
  // a handful of uneven configs.
  static Worklist step_atom_worklist(std::span<Configuration> configs) {
    Worklist work;
    for (std::size_t c = 0; c < configs.size(); ++c) {
      for (std::size_t a = 0; a < configs[c].atoms.size(); ++a) {
        work.emplace_back(c, a);
      }
    }
    return work;
  }

  // Stage 1 — build each config's neighbour list (in parallel via the shared
  // helper; disjoint per config), then size its cache row and record the cell
  // volume in a cheap serial pass.
  CacheData step_allocate_cache(std::span<Configuration> configs) const {
    CacheData data;
    data.rows.resize(configs.size());
    data.volume.resize(configs.size());
    build_all_neighbor_lists(configs, max_cutoff());
    for (auto [cfg, row, vol] :
         std::views::zip(configs, data.rows, data.volume)) {
      row.resize(cfg.atoms.size());
      vol = bc_volume(cfg.bc);
    }
    return data;
  }

  // Stage 2 — force any lazy, model-internal descriptor init (e.g. SOAP's
  // radial basis) exactly once, serially, so the parallel fill below never
  // races on it.
  void step_warm_up_descriptors(std::span<Configuration> configs) const {
    for (auto &cfg : configs) {
      if (!cfg.atoms.empty()) {
        (void)self().get_descriptor(cfg.atoms[0]);
        break;
      }
    }
  }

  // Stage 3 — fill every atom's raw descriptor + gradients in parallel; each
  // task touches only its own AtomCache, so the fill is race-free.
  CacheData step_fill_cache(std::span<Configuration> configs,
                            const Worklist &work, CacheData data) const {
    std::for_each(std::execution::par, work.begin(), work.end(),
                  [&](const std::pair<std::size_t, std::size_t> &ca) {
                    fill_atom_cache(configs[ca.first], ca.second,
                                    data.rows[ca.first][ca.second]);
                  });
    return data;
  }

  // Stage 4 — whiten the just-filled raw descriptors in place: derive per-type
  // μ/σ from the full training set, then standardize every cached value +
  // gradient (same work-list, each task still local to its AtomCache). No-op
  // unless standardize_features.
  CacheData step_whiten_cache(const Worklist &work, CacheData data) const {
    if (!standardize_features) {
      return data;
    }
    compute_standardization(data);
    std::for_each(std::execution::par, work.begin(), work.end(),
                  [&](const std::pair<std::size_t, std::size_t> &ca) {
                    AtomCache &cc = data.rows[ca.first][ca.second];
                    standardize_cached(cc.type, cc.values, cc.grad_all);
                  });
    return data;
  }

  // Stage 5 — regroup the per-atom gradient blocks into per-(config,type)
  // concatenations for the batched linear-head force/Jacobian path. Each config
  // is independent → parallel over configs. After grouping, the per-atom grad
  // blocks are released so total memory stays flat (the data just moves). No-op
  // (groups left empty, per-atom layout kept) unless every head is linear.
  CacheData step_group_cache(CacheData data) const {
    // Sized to rows either way so eval can index groups[cache_index] safely;
    // left empty per config when not grouping (non-linear heads keep per-atom).
    data.groups.resize(data.rows.size());
    if (!all_constant_grad()) {
      return data; // keep per-atom AtomCache layout for non-linear heads
    }
    const auto T = static_cast<std::size_t>(this->ntypes);
    std::vector<std::size_t> cidx(data.rows.size());
    std::iota(cidx.begin(), cidx.end(), std::size_t{0});
    std::for_each(
        std::execution::par, cidx.begin(), cidx.end(), [&](std::size_t c) {
          auto &row = data.rows[c];
          std::vector<Eigen::Index> ncol(T, 0);
          Eigen::Index S = 0;
          for (const AtomCache &cc : row) {
            ncol[static_cast<std::size_t>(cc.type)] += cc.grad_all.cols();
            S = cc.grad_all.rows();
          }
          std::vector<ForceGroup> &grps = data.groups[c];
          std::vector<int> gidx(T, -1);
          for (std::size_t t = 0; t < T; ++t) {
            if (ncol[t] > 0) {
              gidx[t] = static_cast<int>(grps.size());
              ForceGroup g;
              g.type = static_cast<int>(t);
              g.grad.resize(S, ncol[t]);
              const auto nb = static_cast<std::size_t>(ncol[t] / 3);
              g.target.reserve(nb);
              g.bond.reserve(nb);
              grps.push_back(std::move(g));
            }
          }
          std::vector<Eigen::Index> off(T, 0);
          for (std::size_t a = 0; a < row.size(); ++a) {
            AtomCache &cc = row[a];
            ForceGroup &g = grps[static_cast<std::size_t>(
                gidx[static_cast<std::size_t>(cc.type)])];
            Eigen::Index &o = off[static_cast<std::size_t>(cc.type)];
            g.grad.middleCols(o, cc.grad_all.cols()) = cc.grad_all;
            o += cc.grad_all.cols();
            g.target.push_back(static_cast<int>(a)); // self block
            g.bond.emplace_back(Vec3::Zero());
            for (auto [ni, nd] : std::views::zip(cc.neigh_idx, cc.neigh_dist)) {
              g.target.push_back(static_cast<int>(ni));
              g.bond.push_back(nd);
            }
            cc.grad_all.resize(0, 0); // released: data now lives in the group
            cc.neigh_idx = {};
            cc.neigh_dist = {};
          }
        });

    // Precompute the constant per-type M_t = ∂(∂E/∂D)/∂θ once (linear heads):
    // independent of D and of the optimizer params, so the Jacobian never has
    // to build/zero it again. S is the descriptor size (first non-empty row).
    Eigen::Index S = 0;
    for (const auto &row : data.rows) {
      if (!row.empty()) {
        S = row.front().values.size();
        break;
      }
    }
    data.dgrad_param.resize(T);
    if (S > 0) {
      for (std::size_t t = 0; t < T; ++t) {
        data.dgrad_param[t] = heads[t].dgrad_dparam(Eigen::VectorXd::Zero(S));
      }
    }
    return data;
  }

public:
  void prepare(std::span<Configuration> configs) const {
    const Worklist work = step_atom_worklist(configs);

    CacheData data = step_allocate_cache(configs); // neighbour lists + sizing
    step_warm_up_descriptors(configs); // serial lazy descriptor init
    data = step_fill_cache(configs, work, std::move(data)); // parallel fill
    data = step_whiten_cache(work, std::move(data)); // optional whiten
    data = step_group_cache(std::move(data)); // batch linear heads per type

    cache_ = std::make_shared<CacheData>(std::move(data));
  }

  void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                   double &energy, SymTens &stress) const {
    const std::shared_ptr<const CacheData> cache = cache_;
    const auto &rows = cache->rows[cache_index];
    energy = 0.0;
    stress = SymTens::Zero();
    std::ranges::for_each(forces, [](Vec3 &f) { f.setZero(); });

    // Energy is per atom either way (cheap dot products).
    for (const AtomCache &cc : rows) {
      energy += heads[static_cast<std::size_t>(cc.type)].energy(cc.values);
    }

    const auto &groups = cache->groups[cache_index];
    if (!groups.empty()) {
      // Linear-head fast path: one gemv per (config,type). ∂E/∂D = c_t is
      // hoisted out of the atom loop; the whole type's blocks are contracted
      // at once, then scattered. contrib block b is grad_block_b^T · c_t.
      for (const ForceGroup &g : groups) {
        const EnergyHead &h = heads[static_cast<std::size_t>(g.type)];
        const Eigen::VectorXd dEdD =
            h.grad(Eigen::VectorXd{}); // const for linear
        const Eigen::VectorXd contrib = g.grad.transpose() * dEdD;
        for (std::size_t b = 0; b < g.target.size(); ++b) {
          const Vec3 cb = contrib.segment<3>(3 * static_cast<Eigen::Index>(b));
          forces[static_cast<std::size_t>(g.target[b])] -= cb;
          stress -= g.bond[b] * cb.transpose(); // bond==0 for self → no stress
        }
      }
    } else {
      // Fallback (non-linear heads): per-atom batched gemv over its blocks.
      for (auto [cc, fi] : std::views::zip(rows, forces)) {
        const EnergyHead &h = heads[static_cast<std::size_t>(cc.type)];
        const Eigen::VectorXd dEdD = h.grad(cc.values);
        const Eigen::VectorXd contrib = cc.grad_all.transpose() * dEdD;
        fi -= contrib.head<3>();
        Eigen::Index b = 1;
        for (const auto &[nidx, nd] :
             std::views::zip(cc.neigh_idx, cc.neigh_dist)) {
          const Vec3 f_on_j = -contrib.segment<3>(3 * b);
          forces[nidx] += f_on_j;
          stress += nd * f_on_j.transpose();
          ++b;
        }
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

    // Energy column block ∂E_i/∂θ is per atom (param_grad depends on D_i).
    for (int i = 0; i < na; ++i) {
      const AtomCache &cc = rows[i];
      const auto t = static_cast<std::size_t>(cc.type);
      const int c0 = col_off[t];
      const int pc = col_off[t + 1] - c0;
      if (pc == 0) {
        continue;
      }
      const Eigen::VectorXd gE = heads[t].param_grad(cc.values);
      fjac.block(erow, c0, 1, pc).noalias() += energy_weight * gE.transpose();
    }

    // Stress scatter shared by both layouts.
    const auto scatter_stress = [&](int c0, int pc, const Vec3 &nd,
                                    const Eigen::MatrixXd &fblk) {
      if (stress_weight > 0.0) {
        for (int s = 0; s < 6; ++s) {
          fjac.block(srow + s, c0, 1, pc) +=
              (stress_weight * inv_vol * nd[sp[s]]) * fblk.row(sq[s]);
        }
      }
    };

    const auto &groups = cache->groups[cache_index];
    if (!groups.empty()) {
      // Linear-head fast path: M_t = ∂(∂E/∂D)/∂θ is constant per type, so one
      // gemm `grad^T · M_t` covers a whole (config,type); scatter each 3-row
      // block. ∂f_target/∂θ = −block (self & neighbours alike; self bond==0).
      for (const ForceGroup &g : groups) {
        const auto t = static_cast<std::size_t>(g.type);
        const int c0 = col_off[t];
        const int pc = col_off[t + 1] - c0;
        if (pc == 0) {
          continue;
        }
        const Eigen::MatrixXd M =
            heads[t].dgrad_dparam(Eigen::VectorXd::Zero(g.grad.rows()));
        const Eigen::MatrixXd blocks = g.grad.transpose() * M; // (3·nblocks)×pc
        for (std::size_t b = 0; b < g.target.size(); ++b) {
          const Eigen::MatrixXd fblk =
              -blocks.middleRows(3 * static_cast<Eigen::Index>(b), 3); // 3×pc
          fjac.block(row0 + 3 * g.target[b], c0, 3, pc) += fblk;
          scatter_stress(c0, pc, g.bond[b], fblk);
        }
      }
    } else {
      // Fallback (non-linear heads): per-atom batched gemm over its blocks.
      for (int i = 0; i < na; ++i) {
        const AtomCache &cc = rows[i];
        const auto t = static_cast<std::size_t>(cc.type);
        const int c0 = col_off[t];
        const int pc = col_off[t + 1] - c0;
        if (pc == 0) {
          continue;
        }
        const Eigen::MatrixXd M = heads[t].dgrad_dparam(cc.values);
        const Eigen::MatrixXd blocks = cc.grad_all.transpose() * M;
        fjac.block(row0 + 3 * i, c0, 3, pc).noalias() -= blocks.topRows(3);
        Eigen::Index b = 1;
        for (const auto &[nidx, nd] :
             std::views::zip(cc.neigh_idx, cc.neigh_dist)) {
          const Eigen::MatrixXd fblk = -blocks.middleRows(3 * b, 3); // 3 × pc
          fjac.block(row0 + 3 * static_cast<int>(nidx), c0, 3, pc) += fblk;
          scatter_stress(c0, pc, nd, fblk);
          ++b;
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

  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    std::for_each(heads.begin(), heads.end(), [&](const auto &h) {
      h.gather_bounds(lo, hi, off);
      off += h.param_count();
    });
  }

  constexpr double max_cutoff() const { return self().descriptor_cutoff(); }

private:
  const Derived &self() const { return static_cast<const Derived &>(*this); }

  // Descriptor value + position-gradient for one atom: analytic when the model
  // supplies it (has_grad), else a cheap O(neighbors) finite-difference of the
  // descriptor. The descriptor of atom i depends on geometry only through each
  // neighbour's displacement, so dD_i/d(r_j) is obtained by perturbing that
  // neighbour's bond vector IN PLACE — no neighbour-list rebuild — and
  // dD_i/dr_i = −Σ_j dD_i/dr_j (translation invariance). `atom` is mutated only
  // transiently (each dist restored); callers must hold exclusive access to
  // atom's own neighbour list (true in the serial eval_forces and in prepare's
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
      for (auto &&[cdim, nbdist] :
           nb.dist | std::views::take(3) |
               std::views::enumerate) { // int cdim = 0; cdim < 3; ++cdim) {
        const double x0 = nbdist;
        nbdist = x0 + h;
        const Eigen::VectorXd dp = self().get_descriptor(atom).values;
        nbdist = x0 - h;
        const Eigen::VectorXd dm = self().get_descriptor(atom).values;
        nbdist = x0;
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
    // Stack the centre block + every neighbour block column-wise (see
    // AtomCache::grad_all). The producer side keeps the natural per-block
    // layout (DescriptorValue); only the cache is restacked for batched eval.
    const Eigen::Index S = d.values.size();
    const auto K = static_cast<Eigen::Index>(d.grad_neigh.size());
    cc.grad_all.resize(S, 3 * (K + 1));
    cc.grad_all.leftCols(3) = d.grad_self;
    for (Eigen::Index k = 0; k < K; ++k) {
      cc.grad_all.middleCols(3 * (k + 1), 3) = d.grad_neigh[k];
    }
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
  // a constant shift and drops out.
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

  void standardize_cached(int type, Eigen::VectorXd &values,
                          Eigen::MatrixXd &grad_all) const {
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
    grad_all = iv.asDiagonal() * grad_all;
  }

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
      const Eigen::VectorXd mu = sum[t] / n;
      const Eigen::VectorXd var = (sumsq[t] / n) - mu.array().square().matrix();

      mean.emplace_back(std::move(mu));
      inv_std.push_back(
          (var.array() > eps).select(var.array().rsqrt(), 0.0).matrix());
    }
    mean_ = std::move(mean);
    inv_std_ = std::move(inv_std);
  }
};

} // namespace forcesmith
