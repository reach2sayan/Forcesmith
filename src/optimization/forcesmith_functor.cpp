#include "forcesmith/optimization/forcesmith_functor.hpp"

#include "forcesmith/core/families.hpp"
#include "forcesmith/events/signals.hpp"
#include "forcesmith/optimization/fd_jacobian.hpp"
#include "forcesmith/optimization/residual_layout.hpp"

#include <tbb/enumerable_thread_specific.h>
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

using opt::config_residual_count;
using opt::count_residuals;
using opt::write_config_residuals;

namespace {

tbb::task_arena &shared_arena() {
  static tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                                tbb::info::default_concurrency());
  static tbb::task_arena arena(tbb::info::default_concurrency());
  return arena;
}

void eval_into(std::span<Configuration> configs, ForceCalculator &model,
               const Eigen::VectorXd &x, Eigen::VectorXd &fvec,
               double energy_weight, double stress_weight, double smooth_weight,
               const std::vector<int> &row_offset, int smooth_count) {
  model.scatter_params(x, std::size_t{0});

  events::ScopedForceEvalSuppress suppress_events;

  std::for_each(std::execution::par, configs.begin(), configs.end(),
                [&](Configuration &cfg) {
                  const std::size_t c =
                      static_cast<std::size_t>(&cfg - configs.data());
                  model.eval_forces(cfg, c);

                  write_config_residuals(cfg, fvec, row_offset[c],
                                         energy_weight, stress_weight);
                });

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

void df_streaming(ForceCalculator &m0, const Eigen::VectorXd &x,
                  Eigen::MatrixXd &fjac, std::span<Configuration> configs,
                  const std::vector<int> &row_offset, double energy_weight,
                  double stress_weight) {
  fjac.setZero();

  const std::vector<std::size_t> head_pc = m0.head_param_counts();
  std::vector<int> col_off(head_pc.size() + 1, 0);
  for (std::size_t t = 0; t < head_pc.size(); ++t) {
    col_off[t + 1] = col_off[t] + static_cast<int>(head_pc[t]);
  }

  tbb::enumerable_thread_specific<ForceCalculator> tls([&] {
    ForceCalculator m = m0;
    m.scatter_params(x, std::size_t{0});
    return m;
  });

  std::vector<std::size_t> cidx(configs.size());
  std::iota(cidx.begin(), cidx.end(), std::size_t{0});
  std::for_each(std::execution::par, cidx.begin(), cidx.end(),
                [&](std::size_t c) {
                  ForceCalculator &m = tls.local();
                  m.prepare(configs.subspan(c, 1)); // single-config cache → 0
                  m.eval_cached_jacobian(0, row_offset[c], col_off,
                                         energy_weight, stress_weight, fjac);
                });
}

bool with_analytic_calculator(const ForceCalculator &m, auto &&f) {
  bool used = false;
  visit_family<ModelFamilies>(m, [&](const auto &calc) {
    if (calc.has_analytic_jacobian()) {
      used = true;
      f(calc);
    }
  });
  return used;
}

bool df_analytic(const ForceCalculator &m0, const Eigen::VectorXd &x,
                 Eigen::MatrixXd &fjac, std::span<Configuration> configs,
                 const std::vector<int> &row_offset, double energy_weight,
                 double stress_weight) {
  ForceCalculator m = m0;
  m.scatter_params(x, std::size_t{0});
  return with_analytic_calculator(m, [&](const auto &calc) {
    fjac.setZero();
    events::ScopedForceEvalSuppress suppress_events;
    std::for_each(std::execution::par, configs.begin(), configs.end(),
                  [&](Configuration &cfg) {
                    const std::size_t c =
                        static_cast<std::size_t>(&cfg - configs.data());
                    calc.write_param_jacobian(cfg, row_offset[c], energy_weight,
                                              stress_weight, fjac);
                  });
  });
}

} // namespace

ForcesmithFunctor::ForcesmithFunctor(std::span<Configuration> configs,
                                     ForceCalculator model,
                                     double energy_weight, double stress_weight,
                                     double smooth_weight, bool stream_jacobian)
    : configs_{configs}, model_{std::move(model)},
      energy_weight_{energy_weight}, stress_weight_{stress_weight},
      smooth_weight_{smooth_weight}, stream_jacobian_{stream_jacobian},
      smooth_count_{smooth_weight > 0.0
                        ? static_cast<int>(model_.smoothness_count())
                        : 0},
      inputs_{static_cast<int>(model_.param_count())},
      values_{static_cast<int>(count_residuals(configs, stress_weight)) +
              smooth_count_} {
  row_offset_.reserve(configs_.size() + 1);
  int acc = 0;
  row_offset_.push_back(0);
  for (const auto &cfg : configs_) {
    acc += static_cast<int>(config_residual_count(cfg, stress_weight_));
    row_offset_.push_back(acc);
  }

  if (!stream_jacobian_) {
    shared_arena().execute([&] { model_.prepare(configs_); });
  }
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
  if (stream_jacobian_ && model_.has_param_jacobian()) {
    shared_arena().execute([&] {
      df_streaming(model_, x, fjac, configs_, row_offset_, energy_weight_,
                   stress_weight_);
    });
    if (last_fvec_.size() == fjac.rows()) {
      grad_norm_ = (fjac.transpose() * last_fvec_).norm();
    }
    return 0;
  }

  if (smooth_count_ == 0) {
    bool exact = false;
    shared_arena().execute([&] {
      exact = df_analytic(model_, x, fjac, configs_, row_offset_,
                          energy_weight_, stress_weight_);
    });
    if (exact) {
      if (last_fvec_.size() == fjac.rows()) {
        grad_norm_ = (fjac.transpose() * last_fvec_).norm();
      }
      return 0;
    }
  }

  if (!df_cached(x, fjac)) {
    const auto eval_at = [&](const Eigen::VectorXd &xv, Eigen::VectorXd &out) {
      out.resize(values_);
      eval_into(configs_, model_, xv, out, energy_weight_, stress_weight_,
                smooth_weight_, row_offset_, smooth_count_);
    };
    shared_arena().execute([&] { opt::fd_jacobian(eval_at, x, fjac); });
  }

  if (last_fvec_.size() == fjac.rows()) {
    grad_norm_ = (fjac.transpose() * last_fvec_).norm();
  }
  return 0;
}

} // namespace forcesmith
