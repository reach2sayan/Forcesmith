#include "potfit/optimization/potfit_functor.hpp"

#include "potfit/events/signals.hpp"
#include "potfit/force/smoothness.hpp"

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <execution>
#include <numeric>
#include <span>
#include <vector>

namespace potfit {

namespace {

// Per-config residual count: 3 force components per atom + energy + limit, plus
// the 6 stress components when stress fitting is enabled. Mirrors the layout
// written by eval_into and the total in count_residuals.
int config_residual_count(const Configuration &cfg, double stress_weight) {
  return 3 * static_cast<int>(cfg.atoms.size()) + 2 +
         (stress_weight > 0.0 ? 6 : 0);
}

int count_residuals(std::span<Configuration> configs, double stress_weight) {
  return std::transform_reduce(
      configs.begin(), configs.end(), 0, std::plus<>{}, [&](const auto &cfg) {
        return config_residual_count(cfg, stress_weight);
      });
}

// Single bounded arena shared by every parallel region in this TU, plus a
// process-wide cap on total live worker threads. The cap is what keeps the
// nested df()→eval_into parallelism (columns over configs) from
// oversubscribing: TBB's work-stealing composes the two levels within one arena
// instead of spawning threads² . Function-local statics → thread-safe one-time
// init, alive for the whole process.
tbb::task_arena &shared_arena() {
  static tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                                tbb::info::default_concurrency());
  static tbb::task_arena arena(tbb::info::default_concurrency());
  return arena;
}

// Pure residual evaluation: scatter params into `model`, evaluate every config
// in parallel into its own disjoint slice of `fvec`, then append the (serial)
// smoothness block. No shared-state mutation beyond `model`/`fvec`/`configs`,
// so it is safe to call on per-thread copies from df(). Does NOT touch iter_ or
// fire on_iteration — that is the caller's concern. Must run inside an arena.
void eval_into(std::span<Configuration> configs, ForceCalculator &model,
               const Eigen::VectorXd &x, Eigen::VectorXd &fvec,
               double energy_weight, double stress_weight, double smooth_weight,
               const std::vector<int> &row_offset, int smooth_count) {
  std::visit([&](auto &m) { m.scatter_params(x, std::size_t{0}); }, model);

  // Per-config diagnostic events would otherwise fire from every worker thread;
  // mute them for the parallel region (no slots are attached in practice).
  events::ScopedForceEvalSuppress suppress_events;

  std::for_each(
      std::execution::par, configs.begin(), configs.end(),
      [&](Configuration &cfg) {
        // Contiguous span → index of this config is its offset from the base.
        const std::size_t c = static_cast<std::size_t>(&cfg - configs.data());
        // eval_forces mutates only this config (forces/energy/stress/limit +
        // its own neighbor list); the model is read-only here.
        std::visit([&](auto &m) { m.eval_forces(cfg); }, model);

        int row = row_offset[c];
        for (const auto &atom : cfg.atoms) {
          fvec[row++] = atom.calc_force[0] - atom.ref.force[0];
          fvec[row++] = atom.calc_force[1] - atom.ref.force[1];
          fvec[row++] = atom.calc_force[2] - atom.ref.force[2];
        }
        fvec[row++] = energy_weight * (cfg.calc_energy - cfg.ref.energy);
        if (stress_weight > 0.0) {
          fvec[row++] =
              stress_weight * (cfg.calc_stress(0, 0) - cfg.ref.stress(0, 0));
          fvec[row++] =
              stress_weight * (cfg.calc_stress(1, 1) - cfg.ref.stress(1, 1));
          fvec[row++] =
              stress_weight * (cfg.calc_stress(2, 2) - cfg.ref.stress(2, 2));
          fvec[row++] =
              stress_weight * (cfg.calc_stress(0, 1) - cfg.ref.stress(0, 1));
          fvec[row++] =
              stress_weight * (cfg.calc_stress(0, 2) - cfg.ref.stress(0, 2));
          fvec[row++] =
              stress_weight * (cfg.calc_stress(1, 2) - cfg.ref.stress(1, 2));
        }
        // EAM/ADP out-of-range ρ punishment (already weighted; squared by the
        // LM objective → matches potfit's dsquare(limit_p)). Zero for models
        // without an embedding term.
        fvec[row++] = cfg.calc_limit;
      });

  // Tikhonov curvature penalty on free knots, appended after the data
  // residuals.
  if (smooth_count > 0) {
    std::visit(
        [&](auto &m) {
          model_write_smoothness(m, fvec,
                                 static_cast<std::size_t>(row_offset.back()),
                                 smooth_weight);
        },
        model);
  }
}

} // namespace

PotfitFunctor::PotfitFunctor(std::span<Configuration> configs,
                             ForceCalculator model, double energy_weight,
                             double stress_weight, double smooth_weight)
    : configs_(configs), model_(std::move(model)),
      energy_weight_(energy_weight), stress_weight_(stress_weight),
      smooth_weight_(smooth_weight),
      smooth_count_{
          smooth_weight > 0.0
              ? static_cast<int>(std::visit(
                    [](const auto &m) { return model_smoothness_count(m); },
                    model_))
              : 0},
      inputs_{static_cast<int>(
          std::visit([](const auto &m) { return m.param_count(); }, model_))},
      values_{count_residuals(configs, stress_weight) + smooth_count_} {
  // Prefix sum of per-config residual counts; back() = start of smoothness
  // block.
  row_offset_.reserve(configs_.size() + 1);
  int acc = 0;
  row_offset_.push_back(0);
  for (const auto &cfg : configs_) {
    acc += config_residual_count(cfg, stress_weight_);
    row_offset_.push_back(acc);
  }
}

int PotfitFunctor::operator()(const Eigen::VectorXd &x,
                              Eigen::VectorXd &fvec) const {
  shared_arena().execute([&] {
    eval_into(configs_, model_, x, fvec, energy_weight_, stress_weight_,
              smooth_weight_, row_offset_, smooth_count_);
  });

  events::on_iteration(
      events::IterationStats{++iter_, fvec.squaredNorm(), 0.0});

  return 0;
}

int PotfitFunctor::df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
  constexpr double delta = 1e-5;

  // Central finite-difference Jacobian. Columns are evaluated SERIALLY (no
  // per-thread copies): the LM solver calls operator() then df() sequentially,
  // so reusing the shared model_/configs_ is safe, and operator() re-scatters
  // params on its next entry, making the post-df perturbed state harmless.
  // Parallelism comes for free inside each column: eval_into still evaluates
  // the configurations in parallel. (Column-level parallelism is deliberately
  // left for later — it only helps when #configs < #cores.)
  Eigen::VectorXd fp(values_), fm(values_), xp = x;
  shared_arena().execute([&] {
    for (int j = 0; j < inputs_; ++j) {
      xp[j] += delta;
      eval_into(configs_, model_, xp, fp, energy_weight_, stress_weight_,
                smooth_weight_, row_offset_, smooth_count_);
      xp[j] -= 2.0 * delta;
      eval_into(configs_, model_, xp, fm, energy_weight_, stress_weight_,
                smooth_weight_, row_offset_, smooth_count_);
      xp[j] += delta;
      fjac.col(j) = (fp - fm) / (2.0 * delta);
    }
  });
  return 0;
}

} // namespace potfit
