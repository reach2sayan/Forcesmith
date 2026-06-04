// Analytic parameter-Jacobian verification against finite differences.
//
// The central-difference Jacobian is a perfect oracle: head-level checks compare
// param_grad (∂E/∂θ) and dgrad_dparam (∂(∂E/∂D)/∂θ) to FD of energy()/grad();
// the end-to-end check compares PotfitFunctor::df (which routes to the analytic
// df_cached_analytic for heads that support it) to a direct central-difference of
// the residual vector. Phase 1 covers LinearHead (M = I) end-to-end and MLPHead's
// param_grad; the MLP mixed second derivative is verified in Phase 2.

#include "potfit/force/force_calculator.hpp"
#include "potfit/force/ml_force.hpp"
#include "potfit/optimization/potfit_functor.hpp"

#include <gtest/gtest.h>

#include <span>
#include <variant>
#include <vector>

using namespace potfit;

namespace {

Configuration make_cluster() {
  Configuration cfg;
  cfg.bc = PeriodicBC(100.0 * Mat3::Identity());
  Atom a0, a1, a2, a3;
  a0.type = 0; a0.pos = {0.0,  0.0,  0.0};
  a1.type = 0; a1.pos = {2.1,  0.3, -0.2};
  a2.type = 0; a2.pos = {0.4,  2.0,  0.5};
  a3.type = 0; a3.pos = {-1.3, 0.7,  1.9};
  cfg.atoms = {a0, a1, a2, a3};
  return cfg;
}

// SymmetryFunction descriptor (S = 3 radial features) with a LinearHead.
ForceCalculator make_symfunc_linear(const std::vector<double> &coeffs) {
  SymmetryFunctionModel m;
  m.ntypes = 1;
  m.rcut = 5.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
  LinearHead h;
  for (double cv : coeffs) {
    h.coeffs.push_back(Param{cv, false});
  }
  h.bias = Param{0.0, true};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{std::move(h)});
  return ForceCalculator{std::move(m)};
}

// Same S = 3 descriptor with an MLP head of the given layer sizes (sizes[0] = 3).
ForceCalculator make_symfunc_mlp(const std::vector<int> &sizes,
                                 MLPHead::Act act) {
  SymmetryFunctionModel m;
  m.ntypes = 1;
  m.rcut = 5.0;
  m.radial = {{0.5, 0.0}, {1.2, 1.5}, {0.3, 2.5}};
  m.heads.reserve(1);
  m.heads.emplace_back(EnergyHead{MLPHead::make(sizes, act, 7)});
  return ForceCalculator{std::move(m)};
}

// ── Generic FD oracles over a head's own parameters ──────────────────────────
template <typename Head>
Eigen::VectorXd fd_param_grad(Head head, const Eigen::VectorXd &D,
                              double d = 1e-6) {
  const int P = static_cast<int>(head.param_count());
  Eigen::VectorXd th(P);
  head.gather_params(th, 0);
  Eigen::VectorXd g(P);
  for (int k = 0; k < P; ++k) {
    Eigen::VectorXd tp = th;
    tp[k] = th[k] + d;
    head.scatter_params(tp, 0);
    const double ep = head.energy(D);
    tp[k] = th[k] - d;
    head.scatter_params(tp, 0);
    const double em = head.energy(D);
    g[k] = (ep - em) / (2.0 * d);
  }
  head.scatter_params(th, 0);
  return g;
}

template <typename Head>
Eigen::MatrixXd fd_dgrad_dparam(Head head, const Eigen::VectorXd &D,
                                double d = 1e-6) {
  const int P = static_cast<int>(head.param_count());
  const int S = static_cast<int>(head.grad(D).size());
  Eigen::MatrixXd M(S, P);
  Eigen::VectorXd th(P);
  head.gather_params(th, 0);
  for (int k = 0; k < P; ++k) {
    Eigen::VectorXd tp = th;
    tp[k] = th[k] + d;
    head.scatter_params(tp, 0);
    const Eigen::VectorXd gp = head.grad(D);
    tp[k] = th[k] - d;
    head.scatter_params(tp, 0);
    const Eigen::VectorXd gm = head.grad(D);
    M.col(k) = (gp - gm) / (2.0 * d);
  }
  head.scatter_params(th, 0);
  return M;
}

// Central-difference Jacobian of the residual vector — the end-to-end oracle.
Eigen::MatrixXd fd_residual_jacobian(const PotfitFunctor &f,
                                     const Eigen::VectorXd &x, double d = 1e-6) {
  Eigen::MatrixXd J(f.values(), x.size());
  Eigen::VectorXd xp = x, fp(f.values()), fm(f.values());
  for (Eigen::Index k = 0; k < x.size(); ++k) {
    xp[k] = x[k] + d;
    f(xp, fp);
    xp[k] = x[k] - d;
    f(xp, fm);
    xp[k] = x[k];
    J.col(k) = (fp - fm) / (2.0 * d);
  }
  return J;
}

} // namespace

// ── Head-level: LinearHead ───────────────────────────────────────────────────

TEST(MlJacobian, LinearHead_ParamGrad) {
  LinearHead h;
  for (double cv : {0.6, -0.35, 0.2}) {
    h.coeffs.push_back(Param{cv, false});
  }
  h.bias = Param{0.0, true};
  Eigen::VectorXd D(3);
  D << 1.1, -0.7, 0.4;

  const Eigen::VectorXd g = h.param_grad(D);
  EXPECT_LT((g - D).cwiseAbs().maxCoeff(), 1e-9); // ∂E/∂coeff_k = D_k
  EXPECT_LT((g - fd_param_grad(h, D)).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(MlJacobian, LinearHead_DgradDparam_IsIdentity) {
  LinearHead h;
  for (double cv : {0.6, -0.35, 0.2}) {
    h.coeffs.push_back(Param{cv, false});
  }
  h.bias = Param{0.0, true};
  Eigen::VectorXd D(3);
  D << 1.1, -0.7, 0.4;

  const Eigen::MatrixXd M = h.dgrad_dparam(D);
  EXPECT_LT((M - Eigen::MatrixXd::Identity(3, 3)).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((M - fd_dgrad_dparam(h, D)).cwiseAbs().maxCoeff(), 1e-6);
}

// ── Head-level: MLPHead param_grad (Phase 1; dgrad_dparam in Phase 2) ─────────

TEST(MlJacobian, MlpHead_ParamGrad_Tanh) {
  MLPHead h = MLPHead::make({3, 5, 1}, MLPHead::Act::Tanh, 7);
  Eigen::VectorXd D(3);
  D << 0.4, -0.9, 1.3;
  const Eigen::VectorXd g = h.param_grad(D);
  ASSERT_EQ(g.size(), static_cast<Eigen::Index>(h.param_count()));
  EXPECT_LT((g - fd_param_grad(h, D)).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(MlJacobian, MlpHead_ParamGrad_SiLU) {
  MLPHead h = MLPHead::make({3, 6, 4, 1}, MLPHead::Act::SiLU, 11);
  Eigen::VectorXd D(3);
  D << -0.6, 0.7, 0.2;
  const Eigen::VectorXd g = h.param_grad(D);
  EXPECT_LT((g - fd_param_grad(h, D)).cwiseAbs().maxCoeff(), 1e-6);
}

// The crux: MLPHead::dgrad_dparam (analytic forward-over-reverse) vs central FD
// of grad(). Isolates the σ″ recurrence — no force/cache/assembly involved.
TEST(MlJacobian, MlpHead_DgradDparam_Tanh_OneHidden) {
  MLPHead h = MLPHead::make({3, 5, 1}, MLPHead::Act::Tanh, 7);
  Eigen::VectorXd D(3);
  D << 0.4, -0.9, 1.3;
  const Eigen::MatrixXd M = h.dgrad_dparam(D);
  ASSERT_EQ(M.rows(), 3);
  ASSERT_EQ(M.cols(), static_cast<Eigen::Index>(h.param_count()));
  EXPECT_LT((M - fd_dgrad_dparam(h, D)).cwiseAbs().maxCoeff(), 1e-5);
}

TEST(MlJacobian, MlpHead_DgradDparam_Tanh_TwoHidden) {
  MLPHead h = MLPHead::make({3, 6, 4, 1}, MLPHead::Act::Tanh, 11);
  Eigen::VectorXd D(3);
  D << -0.6, 0.7, 0.2;
  EXPECT_LT((h.dgrad_dparam(D) - fd_dgrad_dparam(h, D)).cwiseAbs().maxCoeff(),
            1e-5);
}

TEST(MlJacobian, MlpHead_DgradDparam_SiLU_TwoHidden) {
  MLPHead h = MLPHead::make({3, 5, 5, 1}, MLPHead::Act::SiLU, 3);
  Eigen::VectorXd D(3);
  D << 0.8, -0.4, 0.1;
  EXPECT_LT((h.dgrad_dparam(D) - fd_dgrad_dparam(h, D)).cwiseAbs().maxCoeff(),
            1e-5);
}

// ── End-to-end: analytic df vs FD of the residual (LinearHead path) ──────────

TEST(MlJacobian, SymFuncLinear_AnalyticDf_MatchesFD) {
  ForceCalculator model = make_symfunc_linear({0.6, -0.35, 0.2});
  std::vector<Configuration> configs = {make_cluster()};
  PotfitFunctor f(std::span<Configuration>(configs), model, /*energy_weight=*/0.5);

  Eigen::VectorXd x(f.inputs());
  std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); },
             f.model());

  Eigen::MatrixXd Jana(f.values(), f.inputs());
  f.df(x, Jana);
  const Eigen::MatrixXd Jfd = fd_residual_jacobian(f, x);

  const double scale = 1.0 + Jfd.cwiseAbs().maxCoeff();
  EXPECT_LT((Jana - Jfd).cwiseAbs().maxCoeff(), 1e-6 * scale)
      << "analytic vs FD residual Jacobian mismatch";
}

TEST(MlJacobian, SymFuncLinear_AnalyticDf_MatchesFD_WithStress) {
  ForceCalculator model = make_symfunc_linear({0.6, -0.35, 0.2});
  std::vector<Configuration> configs = {make_cluster()};
  PotfitFunctor f(std::span<Configuration>(configs), model,
                  /*energy_weight=*/0.5, /*stress_weight=*/0.3);

  Eigen::VectorXd x(f.inputs());
  std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); },
             f.model());

  Eigen::MatrixXd Jana(f.values(), f.inputs());
  f.df(x, Jana);
  const Eigen::MatrixXd Jfd = fd_residual_jacobian(f, x);

  const double scale = 1.0 + Jfd.cwiseAbs().maxCoeff();
  EXPECT_LT((Jana - Jfd).cwiseAbs().maxCoeff(), 1e-6 * scale)
      << "analytic vs FD residual Jacobian mismatch (with stress)";
}

// End-to-end MLP: analytic df (df_cached_analytic + MLPHead derivatives + cache
// + standardization) vs FD of the residual.
TEST(MlJacobian, SymFuncMlp_AnalyticDf_MatchesFD) {
  for (auto act : {MLPHead::Act::Tanh, MLPHead::Act::SiLU}) {
    ForceCalculator model = make_symfunc_mlp({3, 6, 1}, act);
    std::vector<Configuration> configs = {make_cluster()};
    PotfitFunctor f(std::span<Configuration>(configs), model,
                    /*energy_weight=*/0.5, /*stress_weight=*/0.3);

    Eigen::VectorXd x(f.inputs());
    std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); },
               f.model());

    Eigen::MatrixXd Jana(f.values(), f.inputs());
    f.df(x, Jana);
    const Eigen::MatrixXd Jfd = fd_residual_jacobian(f, x);

    const double scale = 1.0 + Jfd.cwiseAbs().maxCoeff();
    EXPECT_LT((Jana - Jfd).cwiseAbs().maxCoeff(), 1e-5 * scale)
        << "MLP analytic vs FD Jacobian mismatch (act="
        << (act == MLPHead::Act::Tanh ? "tanh" : "silu") << ")";
  }
}
