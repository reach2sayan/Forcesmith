#pragma once

#include "forcesmith/core/erased.hpp"          // ErasedValue
#include "forcesmith/core/fit_params.hpp"       // CloningModel
#include "forcesmith/core/force_calc_base.hpp"  // ForceCalcConcept
#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/angular_force.hpp"
#include "forcesmith/force/eam_force.hpp"
#include "forcesmith/force/pair_force.hpp"
#include "forcesmith/force/smoothness.hpp" // model_smoothness_count / _write
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/potentials/acsf.hpp"
#include "forcesmith/potentials/lmbtr.hpp"
#include "forcesmith/potentials/soap.hpp"

#include <Eigen/Core>
#include <boost/leaf/result.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace forcesmith {

class ForceCalculator : private detail::ErasedValue<detail::ForceCalcConcept> {
  template <CForceCalculator T>
  struct Model final
      : detail::CloningModel<Model<T>, T, detail::ForceCalcConcept> {
    using Base = detail::CloningModel<Model<T>, T, detail::ForceCalcConcept>;
    using Base::Base;  // inherit the impl_-forwarding constructor
    using Base::impl_; // bring impl_ into scope for the bodies below
    // clone() and the param-plumbing virtuals come from CloningModel.

    void eval_forces(Configuration &cfg) const override {
      impl_.eval_forces(cfg);
    }
    double max_cutoff() const override { return impl_.max_cutoff(); }
    std::size_t ntypes() const override { return impl_.ntypes; }
    std::size_t smoothness_count() const override {
      return model_smoothness_count(impl_);
    }
    void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                          double weight) const override {
      model_write_smoothness(impl_, x, off, weight);
    }

    // The descriptor-cache fit fast paths below are part of every calculator's
    // contract (analytic types inherit NoCache defaults; ML models implement
    // them in MLBase), so these forward unconditionally — no `if constexpr`.
    void eval_forces(Configuration &cfg,
                     std::size_t cache_index) const override {
      impl_.eval_forces(cfg, cache_index);
    }
    void prepare(std::span<Configuration> configs) const override {
      impl_.prepare(configs);
    }
    bool has_cache() const override { return impl_.has_cache(); }
    bool has_param_jacobian() const override {
      return impl_.has_param_jacobian();
    }
    bool has_standardization() const override {
      return impl_.has_standardization();
    }
    void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                     double &energy, SymTens &stress) const override {
      impl_.eval_cached(cache_index, forces, energy, stress);
    }
    void eval_cached_jacobian(std::size_t cache_index, int row0,
                              const std::vector<int> &col_off,
                              double energy_weight, double stress_weight,
                              Eigen::MatrixXd &fjac) const override {
      impl_.eval_cached_jacobian(cache_index, row0, col_off, energy_weight,
                                 stress_weight, fjac);
    }
    std::vector<std::size_t> head_param_counts() const override {
      return impl_.head_param_counts();
    }
    // remap stays probed: it is ML-only and couples to SpeciesRegistry/leaf, so
    // analytic types deliberately do NOT carry it (default → a leaf error).
    boost::leaf::result<std::unique_ptr<detail::ForceCalcConcept>>
    remap(const SpeciesRegistry &old_reg,
          const SpeciesRegistry &new_reg) const override {
      if constexpr (requires(const T &t, const SpeciesRegistry &o,
                             const SpeciesRegistry &n) { t.remap(o, n); }) {
        auto r = impl_.remap(old_reg, new_reg);
        if (!r) {
          return r.error();
        }
        std::unique_ptr<detail::ForceCalcConcept> erased =
            std::make_unique<Model<T>>(std::move(r.value()));
        return erased;
      } else {
        return boost::leaf::new_error(); // not an ML model
      }
    }
  };

  using Base = detail::ErasedValue<detail::ForceCalcConcept>;

  // Adopt an already-erased concept pointer (used by remap()).
  explicit ForceCalculator(std::unique_ptr<detail::ForceCalcConcept> p) noexcept
      : Base(std::move(p)) {}

public:
  // Default-constructs to an empty pair calculator. Matches the std::variant<…>
  // this replaced, which value-initialized its first alternative
  // (PairForceCalculator) — call sites like `ForceCalculator m;` followed by a
  // later assignment (checkpoint reload, file readers) rely on that.
  ForceCalculator() : ForceCalculator(PairForceCalculator{}) {}

  // Implicit (like the std::variant it replaces) so a concrete calculator
  // converts on assignment/return: `ForceCalculator m = make_pair_force(...)`.
  // The constraint keeps copy/move out of this template's hands.
  template <typename T>
    requires(!std::same_as<std::decay_t<T>, ForceCalculator> &&
             CForceCalculator<std::decay_t<T>>)
  ForceCalculator(T &&t)
      : Base(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(t))) {}

  ForceCalculator(const ForceCalculator &) = default;
  ForceCalculator(ForceCalculator &&) noexcept = default;
  ForceCalculator &operator=(const ForceCalculator &) = default;
  ForceCalculator &operator=(ForceCalculator &&) noexcept = default;

  template <class T> const T *target() const noexcept {
    auto *m = dynamic_cast<const Model<T> *>(self_.get());
    return m ? &m->impl_ : nullptr;
  }
  template <class T> T *target() noexcept {
    auto *m = dynamic_cast<Model<T> *>(self_.get());
    return m ? &m->impl_ : nullptr;
  }

  FORCE_INLINE void eval_forces(Configuration &cfg) const {
    self_->eval_forces(cfg);
  }
  FORCE_INLINE void eval_forces(Configuration &cfg,
                                std::size_t cache_index) const {
    self_->eval_forces(cfg, cache_index);
  }
  FORCE_INLINE double max_cutoff() const { return self_->max_cutoff(); }
  constexpr std::size_t ntypes() const { return self_->ntypes(); }

  constexpr std::size_t param_count() const { return self_->param_count(); }
  constexpr void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    self_->gather_params(x, off);
  }
  void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    self_->scatter_params(x, off);
  }
  constexpr void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                               std::size_t off) const {
    self_->gather_bounds(lo, hi, off);
  }
  constexpr std::size_t smoothness_count() const {
    return self_->smoothness_count();
  }
  void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                        double weight) const {
    self_->write_smoothness(x, off, weight);
  }
  void prepare(std::span<Configuration> configs) const {
    self_->prepare(configs);
  }
  FORCE_INLINE bool has_cache() const { return self_->has_cache(); }
  FORCE_INLINE bool has_param_jacobian() const {
    return self_->has_param_jacobian();
  }
  FORCE_INLINE bool has_standardization() const {
    return self_->has_standardization();
  }
  FORCE_INLINE void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                                double &energy, SymTens &stress) const {
    self_->eval_cached(cache_index, forces, energy, stress);
  }
  FORCE_INLINE void
  eval_cached_jacobian(std::size_t cache_index, int row0,
                       const std::vector<int> &col_off, double energy_weight,
                       double stress_weight, Eigen::MatrixXd &fjac) const {
    self_->eval_cached_jacobian(cache_index, row0, col_off, energy_weight,
                                stress_weight, fjac);
  }
  std::vector<std::size_t> head_param_counts() const {
    return self_->head_param_counts();
  }

  // ML species re-rank — rewraps the erased concept pointer the virtual returns
  // back into a ForceCalculator value. Leaf error for non-ML models.
  [[nodiscard]] boost::leaf::result<ForceCalculator>
  remap(const SpeciesRegistry &old_reg, const SpeciesRegistry &new_reg) const {
    auto r = self_->remap(old_reg, new_reg);
    if (!r) {
      return r.error();
    }
    return ForceCalculator{std::move(r.value())};
  }
};

} // namespace forcesmith
