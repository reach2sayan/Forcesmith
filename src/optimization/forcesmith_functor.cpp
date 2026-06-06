#include "forcesmith/optimization/forcesmith_functor.hpp"

#include "forcesmith/events/signals.hpp"

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace forcesmith {

namespace {

// Per-config residual count: 3 force components per atom + energy + limit, plus
// the 6 stress components when stress fitting is enabled. Mirrors the layout
// written by eval_into and the total in count_residuals.
FORCE_INLINE std::size_t config_residual_count(const Configuration &cfg,
                                               double stress_weight) {
  return 3 * cfg.atoms.size() + 2 + (stress_weight > 0.0 ? 6 : 0);
}

FORCE_INLINE std::size_t count_residuals(std::span<Configuration> configs,
                                         double stress_weight) {
  return std::transform_reduce(
      configs.begin(), configs.end(), 0, std::plus<>{}, [&](const auto &cfg) {
        return config_residual_count(cfg, stress_weight);
      });
}

tbb::task_arena &shared_arena() {
  static tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                                tbb::info::default_concurrency());
  static tbb::task_arena arena(tbb::info::default_concurrency());
  return arena;
}

// Pure residual evaluation: scatter params into `model`,
// (a) evaluate every config into its own disjoint slice of `fvec`,
// then append the (serial) smoothness block.
// No shared-state mutation beyond `model`/`fvec`/`configs`,
// so it is safe to call on per-thread copies from df().
void eval_into(std::span<Configuration> configs, ForceCalculator &model,
               const Eigen::VectorXd &x, Eigen::VectorXd &fvec,
               double energy_weight, double stress_weight, double smooth_weight,
               const std::vector<int> &row_offset, int smooth_count) {
  model.scatter_params(x, std::size_t{0});

  // Per-config diagnostic events would otherwise fire from every worker thread;
  // mute them for the parallel region (no slots are attached in practice).
  events::ScopedForceEvalSuppress suppress_events;

  std::for_each(
      std::execution::par, configs.begin(), configs.end(),
      [&](Configuration &cfg) {
        // Contiguous span → index of this config is its offset from the base.
        const std::size_t c = static_cast<std::size_t>(&cfg - configs.data());
        model.eval_forces(cfg, c);

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
        // LM objective
        // Zero for models without an embedding term.
        fvec[row++] = cfg.calc_limit;
      });

  // Tikhonov curvature penalty on free knots, appended after the data
  // residuals.
  if (smooth_count > 0) {
    model.write_smoothness(fvec, static_cast<std::size_t>(row_offset.back()),
                           smooth_weight);
  }
}

void df_cached_parallel(ForceCalculator &m0, const Eigen::VectorXd &x,
                        Eigen::MatrixXd &fjac, std::span<Configuration> configs,
                        const std::vector<int> &row_offset, int values,
                        int inputs, double energy_weight,
                        double stress_weight) {
  const auto sizes = configs | std::views::transform([](const auto &cfg) {
                       return cfg.atoms.size();
                     });
  const std::size_t max_atoms = configs.empty() ? 0 : std::ranges::max(sizes);
  std::vector<int> cols(static_cast<std::size_t>(inputs));
  std::iota(cols.begin(), cols.end(), 0);
  constexpr double delta = 1e-5;

  std::for_each(std::execution::par, cols.begin(), cols.end(), [&](int j) {
    ForceCalculator m = m0; // private head copy; cache_ shared read-only
    Eigen::VectorXd xp = x, fp(values), fm(values);
    std::vector<Vec3> fbuf(max_atoms);
    auto eval_all = [&](const Eigen::VectorXd &xx, Eigen::VectorXd &out) {
      m.scatter_params(xx, std::size_t{0});
      for (const auto &[c, config] : configs | std::views::enumerate) {
        const std::size_t na = config.atoms.size();
        double e = 0.0;
        SymTens s = SymTens::Zero();
        m.eval_cached(c, std::span<Vec3>(fbuf.data(), na), e, s);
        int row = row_offset[c];
        for (std::size_t a = 0; a < na; ++a) {
          const Vec3 &rf = config.atoms[a].ref.force;
          out[row++] = fbuf[a][0] - rf[0];
          out[row++] = fbuf[a][1] - rf[1];
          out[row++] = fbuf[a][2] - rf[2];
        }
        out[row++] = energy_weight * (e - config.ref.energy);
        if (stress_weight > 0.0) {
          const SymTens &rs = config.ref.stress;
          out[row++] = stress_weight * (s(0, 0) - rs(0, 0));
          out[row++] = stress_weight * (s(1, 1) - rs(1, 1));
          out[row++] = stress_weight * (s(2, 2) - rs(2, 2));
          out[row++] = stress_weight * (s(0, 1) - rs(0, 1));
          out[row++] = stress_weight * (s(0, 2) - rs(0, 2));
          out[row++] = stress_weight * (s(1, 2) - rs(1, 2));
        }
        out[row++] = 0.0; // limit residual (ML models have none)
      }
    };
    xp[j] += delta;
    eval_all(xp, fp);
    xp[j] -= 2.0 * delta;
    eval_all(xp, fm);
    fjac.col(j) = (fp - fm) / (2.0 * delta);
  });
}

void df_cached_analytic(ForceCalculator &m0, const Eigen::VectorXd &x,
                        Eigen::MatrixXd &fjac, std::span<Configuration> configs,
                        const std::vector<int> &row_offset,
                        double energy_weight, double stress_weight) {
  fjac.setZero();
  ForceCalculator m = m0; // private copy;
  m.scatter_params(x, std::size_t{0});

  // Column offset per head/type — the column analogue of row_offset.
  const std::vector<std::size_t> head_pc = m.head_param_counts();
  std::vector<int> col_off(head_pc.size() + 1, 0);
  for (std::size_t t = 0; t < head_pc.size(); ++t) {
    col_off[t + 1] = col_off[t] + static_cast<int>(head_pc[t]);
  }

  std::vector<std::size_t> cidx(configs.size());
  std::iota(cidx.begin(), cidx.end(), std::size_t{0});
  std::for_each(std::execution::par, cidx.begin(), cidx.end(),
                [&](std::size_t c) {
                  m.eval_cached_jacobian(c, row_offset[c], col_off,
                                         energy_weight, stress_weight, fjac);
                });
}

} // namespace

ForcesmithFunctor::ForcesmithFunctor(std::span<Configuration> configs,
                                     ForceCalculator model,
                                     double energy_weight, double stress_weight,
                                     double smooth_weight)
    : configs_(configs), model_(std::move(model)),
      energy_weight_(energy_weight), stress_weight_(stress_weight),
      smooth_weight_(smooth_weight),
      smooth_count_{smooth_weight > 0.0
                        ? static_cast<int>(model_.smoothness_count())
                        : 0},
      inputs_{static_cast<int>(model_.param_count())},
      values_{static_cast<int>(count_residuals(configs, stress_weight)) +
              smooth_count_} {
  // Prefix sum of per-config residual counts; back() = start of smoothness
  // block.
  row_offset_.reserve(configs_.size() + 1);
  int acc = 0;
  row_offset_.push_back(0);
  for (const auto &cfg : configs_) {
    acc += config_residual_count(cfg, stress_weight_);
    row_offset_.push_back(acc);
  }

  // Precompute the descriptor cache ONCE so every residual/Jacobian evaluation
  // below is cheap cached algebra.
  shared_arena().execute([&] { model_.prepare(configs_); });
}

int ForcesmithFunctor::operator()(const Eigen::VectorXd &x,
                                  Eigen::VectorXd &fvec) const {
  shared_arena().execute([&] {
    eval_into(configs_, model_, x, fvec, energy_weight_, stress_weight_,
              smooth_weight_, row_offset_, smooth_count_);
  });
  last_fvec_ = fvec;
  events::on_iteration(
      events::IterationStats{++iter_, fvec.squaredNorm(), grad_norm_});
  return 0;
}

bool ForcesmithFunctor::df_cached(const Eigen::VectorXd &x,
                                  Eigen::MatrixXd &fjac) const {
  bool handled = false;
  shared_arena().execute([&] {
    if (!model_.has_cache()) {
      return; // analytic calculators recompute on the finite-difference path
    }
    handled = true;
    if (model_.has_param_jacobian()) {
      df_cached_analytic(model_, x, fjac, configs_, row_offset_, energy_weight_,
                         stress_weight_);
    } else {
      df_cached_parallel(model_, x, fjac, configs_, row_offset_, values_,
                         inputs_, energy_weight_, stress_weight_);
    }
  });
  return handled;
}

int ForcesmithFunctor::df(const Eigen::VectorXd &x,
                          Eigen::MatrixXd &fjac) const {
  if (!df_cached(x, fjac)) {
    constexpr double delta = 1e-5;
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
  }

  // Gradient of the least-squares objective ½‖f‖² is g = Jᵀf;
  if (last_fvec_.size() == fjac.rows()) {
    grad_norm_ = (fjac.transpose() * last_fvec_).norm();
  }
  return 0;
}

} // namespace forcesmith
