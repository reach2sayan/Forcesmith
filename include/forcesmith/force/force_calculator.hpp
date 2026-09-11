#pragma once

#include "forcesmith/core/erasure.hpp"
#include "forcesmith/core/families.hpp"
#include "forcesmith/core/fit_params.hpp" // CloningModel
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

#include <boost/mp11/algorithm.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace forcesmith {

template <typename Families, typename Model, typename F>
bool visit_family(Model &model, F &&f) {
  bool matched = false;
  boost::mp11::mp_for_each<
      boost::mp11::mp_transform<boost::mp11::mp_identity, Families>>(
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (matched) {
          return;
        }
        if (auto *calc = model.template target<T>()) {
          matched = true;
          f(*calc);
        }
      });
  return matched;
}

template <typename Families, typename Model> bool holds_any(const Model &model) {
  return visit_family<Families>(model, [](const auto &) {});
}

template <class T>
  requires requires(const T &t) { t.ntypes; }
std::size_t ntypes_of(const T &t) {
  return t.ntypes;
}

namespace te_detail {

struct CForceCalcTE
    : boost::mpl::vector<
          te::ValueBuiltins, te::CFittableTE,
          te::has_eval_forces<void(Configuration &) const>,
          te::has_eval_forces<void(Configuration &, std::size_t) const>,
          te::has_max_cutoff<double() const>,
          te::has_prepare<void(std::span<Configuration>) const>,
          te::has_has_cache<bool() const>,
          te::has_has_param_jacobian<bool() const>,
          te::has_has_standardization<bool() const>,
          te::has_eval_cached<void(std::size_t, std::span<Vec3>, double &,
                                   SymTens &) const>,
          te::has_eval_cached_jacobian<void(std::size_t, int,
                                            const std::vector<int> &, double,
                                            double, Eigen::MatrixXd &) const>,
          te::has_head_param_counts<std::vector<std::size_t>() const>,
          te::has_ntypes_of<std::size_t(const te::_self &)>,
          te::has_model_smoothness_count<std::size_t(const te::_self &)>,
          te::has_model_write_smoothness<void(
              const te::_self &, Eigen::VectorXd &, std::size_t, double)>> {};
} // namespace te_detail

class ForceCalculator {
  boost::type_erasure::any<te_detail::CForceCalcTE> a_;

public:
  ForceCalculator() : a_{PairForceCalculator{}} {}

  template <typename T>
    requires(!std::same_as<std::decay_t<T>, ForceCalculator> &&
             CForceCalculator<std::decay_t<T>>)
  ForceCalculator(T &&t) : a_{std::forward<T>(t)} {}

  ForceCalculator(const ForceCalculator &) = default;
  ForceCalculator(ForceCalculator &&) = default;
  ForceCalculator &operator=(const ForceCalculator &) = default;
  ForceCalculator &operator=(ForceCalculator &&) = default;

  template <class T> const T *target() const noexcept {
    return boost::type_erasure::any_cast<const T *>(&a_);
  }
  template <class T> T *target() noexcept {
    return boost::type_erasure::any_cast<T *>(&a_);
  }

  FORCE_INLINE void eval_forces(Configuration &cfg) const {
    a_.eval_forces(cfg);
  }
  FORCE_INLINE void eval_forces(Configuration &cfg,
                                std::size_t cache_index) const {
    a_.eval_forces(cfg, cache_index);
  }
  FORCE_INLINE double max_cutoff() const { return a_.max_cutoff(); }
  std::size_t ntypes() const { return ntypes_of(a_); }

  std::size_t param_count() const { return a_.param_count(); }
  void gather_params(Eigen::VectorXd &x, std::size_t off) const {
    a_.gather_params(x, off);
  }
  void scatter_params(const Eigen::VectorXd &x, std::size_t off) {
    a_.scatter_params(x, off);
  }
  void gather_bounds(Eigen::VectorXd &lo, Eigen::VectorXd &hi,
                     std::size_t off) const {
    a_.gather_bounds(lo, hi, off);
  }
  std::size_t smoothness_count() const { return model_smoothness_count(a_); }
  void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                        double weight) const {
    model_write_smoothness(a_, x, off, weight);
  }
  void prepare(std::span<Configuration> configs) const { a_.prepare(configs); }
  FORCE_INLINE bool has_cache() const { return a_.has_cache(); }
  FORCE_INLINE bool has_param_jacobian() const {
    return a_.has_param_jacobian();
  }
  FORCE_INLINE bool has_standardization() const {
    return a_.has_standardization();
  }
  FORCE_INLINE void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                                double &energy, SymTens &stress) const {
    a_.eval_cached(cache_index, forces, energy, stress);
  }
  FORCE_INLINE void eval_cached_jacobian(std::size_t cache_index, int row0,
                                         const std::vector<int> &col_off,
                                         double energy_weight,
                                         double stress_weight,
                                         Eigen::MatrixXd &fjac) const {
    a_.eval_cached_jacobian(cache_index, row0, col_off, energy_weight,
                            stress_weight, fjac);
  }
  std::vector<std::size_t> head_param_counts() const {
    return a_.head_param_counts();
  }

  [[nodiscard]] boost::leaf::result<ForceCalculator>
  remap(const SpeciesRegistry &old_reg, const SpeciesRegistry &new_reg) const;
};

inline boost::leaf::result<ForceCalculator>
ForceCalculator::remap(const SpeciesRegistry &old_reg,
                       const SpeciesRegistry &new_reg) const {
  std::optional<boost::leaf::result<ForceCalculator>> out;
  const bool is_ml = visit_family<MLFamilies>(*this, [&](const auto &ml) {
    out = [&]() -> boost::leaf::result<ForceCalculator> {
      BOOST_LEAF_AUTO(m, ml.remap(old_reg, new_reg));
      return ForceCalculator{std::move(m)};
    }();
  });
  if (!is_ml) {
    return boost::leaf::new_error(); // not an ML model
  }
  return std::move(*out);
}

} // namespace forcesmith
