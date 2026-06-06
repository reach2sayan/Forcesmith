#pragma once

#include "forcesmith/core/atom.hpp"       // Configuration
#include "forcesmith/core/fit_params.hpp" // FittableConcept
#include "forcesmith/core/species.hpp"    // SpeciesRegistry
#include "forcesmith/core/types.hpp"      // Vec3, SymTens

#include <Eigen/Core>
#include <boost/leaf/result.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace forcesmith {

class ForceCalculator; // defined in forcesmith/force/force_calculator.hpp

namespace detail {

// The erased interface every force calculator satisfies — the open analogue of
// the old ForceCalculator std::variant. Inherits the optimizer param-plumbing
// virtuals (param_count / gather_params / scatter_params / gather_bounds) from
// FittableConcept (core/fit_params.hpp).
//
// The OPTIONAL methods below are pure here but defaulted in
// ForceCalculator::Model<T> via `if constexpr (requires …)` (the same idiom
// RadialPotential uses for prepare_site/eval_at), so an analytic calculator
// that lacks the descriptor-cache fit fast paths still models the interface.
struct ForceCalcConcept : FittableConcept {
  // required: every calculator implements these
  virtual void eval_forces(Configuration &cfg) const = 0;
  virtual double max_cutoff() const = 0;
  virtual std::size_t ntypes() const = 0;
  virtual std::size_t smoothness_count() const = 0;
  virtual void write_smoothness(Eigen::VectorXd &x, std::size_t off,
                                double weight) const = 0;

  // optional: defaulted in Model<T>
  // Indexed (descriptor-cache) force eval; default forwards to
  // eval_forces(cfg).
  virtual void eval_forces(Configuration &cfg,
                           std::size_t cache_index) const = 0;

  // Pre-cache descriptors / prime spline sites; default no-op.
  virtual void prepare(std::span<Configuration> configs) const = 0;

  // Fit fast-path predicates; default false.
  virtual bool has_cache() const = 0;
  virtual bool has_param_jacobian() const = 0;
  virtual void eval_cached(std::size_t cache_index, std::span<Vec3> forces,
                           double &energy, SymTens &stress) const = 0;
  virtual void eval_cached_jacobian(std::size_t cache_index, int row0,
                                    const std::vector<int> &col_off,
                                    double energy_weight, double stress_weight,
                                    Eigen::MatrixXd &fjac) const = 0;
  virtual std::vector<std::size_t> head_param_counts() const = 0;

  // ML species re-rank. Returns the remapped calculator already erased into a
  // fresh concept pointer; ForceCalculator::remap rewraps it into a
  // ForceCalculator value. (The virtual cannot return result<ForceCalculator>
  // directly — ForceCalculator is incomplete here.) Default: a leaf error,
  // since only ML models (ACSF/SOAP/LMBTR) supply remap.
  virtual boost::leaf::result<std::unique_ptr<ForceCalcConcept>>
  remap(const SpeciesRegistry &old_reg,
        const SpeciesRegistry &new_reg) const = 0;

  virtual std::unique_ptr<ForceCalcConcept> clone() const = 0;
};

} // namespace detail
} // namespace forcesmith
