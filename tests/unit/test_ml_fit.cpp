#include "forcesmith/force/force_calculator.hpp"
#include "forcesmith/optimization/optimizer.hpp"
#include "forcesmith/potentials/acsf.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <variant>
#include <vector>

using namespace forcesmith;

namespace {

// A few small single-species clusters (type slot 0), deterministic positions.
std::vector<Configuration> make_dataset() {
  std::vector<Configuration> cfgs;
  const double coords[3][6][3] = {
      {{0, 0, 0}, {2.2, 0, 0}, {0, 2.3, 0}, {0, 0, 2.1}, {2.0, 2.0, 0}, {1.0, 1.0, 1.5}},
      {{0, 0, 0}, {2.5, 0.2, 0}, {0.3, 2.1, 0.4}, {-2.0, 0.1, 0.5}, {0.2, -2.2, 0.3}, {1.1, 1.2, 2.0}},
      {{0, 0, 0}, {2.1, 0.5, 0.5}, {0.6, 2.0, -0.3}, {-1.9, 0.4, 1.0}, {0.5, 0.6, 2.4}, {-1.0, -1.5, -1.0}},
  };
  for (auto &cset : coords) {
    Configuration cfg;
    cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
    for (auto &p : cset) {
      Atom a;
      a.type = 0;
      a.pos = {p[0], p[1], p[2]};
      cfg.atoms.push_back(a);
    }
    cfgs.push_back(std::move(cfg));
  }
  return cfgs;
}

ACSF base_model() {
  ACSF m;
  m.ntypes = 1;
  m.rcut = 5.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
  return m;
}

double force_rmse(ForceCalculator &model, std::vector<Configuration> &cfgs) {
  double ss = 0.0;
  int n = 0;
  for (auto &cfg : cfgs) {
    std::visit([&](auto &m) { m.eval_forces(cfg); }, model);
    for (const auto &a : cfg.atoms) {
      ss += (a.calc_force - a.ref.force).squaredNorm();
      n += 3;
    }
  }
  return std::sqrt(ss / n);
}

} // namespace

TEST(MLFit, LinearTrainingReducesForceRMSE) {
  // Teacher: G2 descriptor + linear head with known coefficients. Use it to
  // label reference forces.
  auto teacher = base_model();
  {
    LinearHead h;
    h.coeffs = {Param{0.6}, Param{-0.35}, Param{0.2}};
    h.bias = Param{0.0, true};
    teacher.heads.reserve(1);
    teacher.heads.emplace_back(EnergyHead{std::move(h)});
  }
  auto data = make_dataset();
  for (auto &cfg : data) {
    teacher.eval_forces(cfg);
    for (auto &a : cfg.atoms) {
      a.ref.force = a.calc_force; // label
    }
  }

  // Student: same descriptor, a linear head started at zero coefficients. The
  // fit should recover the teacher's coeffs and drive the force RMSE down.
  auto student = base_model();
  {
    LinearHead h;
    h.coeffs = {Param{0.0, false}, Param{0.0, false}, Param{0.0, false}};
    h.bias = Param{0.0, true}; // forces-only fit: bias has zero force Jacobian
    student.heads.reserve(1);
    student.heads.emplace_back(EnergyHead{std::move(h)});
  }
  ForceCalculator model{std::move(student)};

  const double before = force_rmse(model, data);

  OptimizerOptions opts;
  opts.energy_weight = 0.0; // forces-only fit (no energy labels)
  opts.stress_weight = 0.0;
  run_optimizer(std::span<Configuration>(data), model, opts);

  const double after = force_rmse(model, data);

  EXPECT_GT(before, 0.0);
  EXPECT_LT(after, before) << "before=" << before << " after=" << after;
}
