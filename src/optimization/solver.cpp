#include "potfit/optimization/solver.hpp"

#include "potfit/optimization/line_search.hpp"

#include <boost/math/optimization/differential_evolution.hpp>
#include <Eigen/Cholesky>
#include <unsupported/Eigen/NonLinearOptimization>

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <ranges>
#include <vector>

namespace {
// Process-wide threading hygiene, applied once at library-load static-init time
// (before any BLAS computation and before main; solver.o is always linked, so
// this initializer always fires). Lets every run go without an
// OPENBLAS_NUM_THREADS=1 prefix.
//
// Ipopt's MUMPS (libcoinmumps) drags in OpenBLAS, which spawns nproc idle
// threads on load (parked in futex_wait) and oversubscribes the box — pure
// overhead, since the hot path is Eigen → threaded MKL. Shrink its pool to 1.
// Weak symbol: a no-op when OpenBLAS isn't present.
//
// The companion concern — MUMPS also linking the MKL *sequential* threading
// layer, which could fight our TBB layer — is handled at link time, not here:
// Dependencies.cmake links libmkl_tbb_thread directly (and ahead of
// libcoinmumps), so in MKL's layered model the TBB threading symbols bind first.
// (MKL's runtime mkl_set_threading_layer() lives only in libmkl_rt, not the
// layered libs, so it isn't an option here.)
extern "C" void openblas_set_num_threads(int) __attribute__((weak));

const int kThreadingHygiene = [] {
  if (openblas_set_num_threads)
    openblas_set_num_threads(1);
  return 0;
}();
} // namespace

namespace potfit {

namespace {

// Adapts std::function<VectorXd(VectorXd)> to the Eigen DenseFunctor interface.
struct FunctorAdapter {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  ResidualFn fn;
  int n_inputs;
  int n_values;
  JacobianFn jac; // optional; when set, df delegates to it (PotfitFunctor::df)

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    fvec = std::invoke(fn, x);
    return 0;
  }

  // Prefer the supplied (typed, parallel) Jacobian; otherwise fall back to a
  // local central finite-difference Jacobian (Δ = 1e-5).
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
    if (jac) {
      std::invoke(jac, x, fjac);
      return 0;
    }
    constexpr double delta = 1e-5;
    Eigen::VectorXd fp, fm, xp = x;
    for (int j : std::views::iota(0, n_inputs)) {
      xp[j] += delta;
      fp = fn(xp);
      xp[j] -= 2.0 * delta;
      fm = fn(xp);
      xp[j] += delta;
      fjac.col(j) = (fp - fm) / (2.0 * delta);
    }
    return 0;
  }

  constexpr int inputs() const { return n_inputs; }
  constexpr int values() const { return n_values; }
};

} // namespace

int EigenLMSolver::minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
                            int n_vals) const {
  FunctorAdapter adapter{std::move(f), static_cast<int>(x.size()), n_vals,
                         std::move(jac)};
  Eigen::LevenbergMarquardt<FunctorAdapter> lm(adapter);
  // max_iter counts LM *iterations* (one Jacobian + trust-region step each), to
  // match IpoptSolver. Eigen's lm.minimize() is
  // instead bounded by maxfev (function evaluations), so the same --maxiter flag
  // meant wildly different work across solvers. Drive minimizeOneStep directly
  // and set maxfev high enough that the per-step search is never the limiter.
  namespace LM = Eigen::LevenbergMarquardtSpace;
  lm.parameters.maxfev = std::max(max_iter, 1) * 100 + 100;
  lm.parameters.xtol = xtol;
  lm.parameters.ftol = ftol;
  if (lm.minimizeInit(x) == LM::ImproperInputParameters)
    return static_cast<int>(LM::ImproperInputParameters);
  LM::Status status = LM::Running; // minimizeInit leaves NotStarted; prime the loop
  for (int i = 0; i < max_iter && status == LM::Running; ++i)
    status = lm.minimizeOneStep(x);
  // Still Running ⇒ stopped by the iteration cap, not a numerical failure; report
  // it as the max-iter-reached code (2), matching IpoptSolver.
  return status == LM::Running ? 2 : static_cast<int>(status);
}

int EigenHybridSolver::minimize(Eigen::VectorXd &x, ResidualFn f,
                                JacobianFn /*jac*/, int /*n_vals*/) const {
  const int D = static_cast<int>(x.size());

  // Gradient functor: g(x)[j] = Fᵀ ∂F/∂xⱼ  (central FD, δ=1e-5)
  struct GradFunctor {
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> fn;
    int D;

    int operator()(const Eigen::VectorXd &xv, Eigen::VectorXd &grad) const {
      constexpr double delta = 1e-5;
      const Eigen::VectorXd f0 = fn(xv);
      grad.resize(D);
      Eigen::VectorXd xp = xv;
      for (int j = 0; j < D; ++j) {
        xp[j] += delta;
        const Eigen::VectorXd fp = fn(xp);
        xp[j] -= 2.0 * delta;
        const Eigen::VectorXd fm = fn(xp);
        xp[j] += delta;
        grad[j] = f0.dot((fp - fm) / (2.0 * delta));
      }
      return 0;
    }

    constexpr int inputs() const { return D; }
    constexpr int values() const { return D; } // square system
  };

  GradFunctor gf{std::move(f), D};
  Eigen::HybridNonLinearSolver<GradFunctor> solver(gf);
  solver.parameters.maxfev = max_iter * D;
  solver.parameters.xtol = xtol;
  return static_cast<int>(solver.solveNumericalDiff(x));
}

int BoostDESolver::minimize(Eigen::VectorXd &x, ResidualFn f,
                            JacobianFn /*jac*/, int /*n_vals*/) const {
  const int D = static_cast<int>(x.size());

  // Build per-parameter bounds; auto-derive from current x if not provided.
  std::vector<double> lb = lower_bounds;
  std::vector<double> ub = upper_bounds;
  if (lb.empty() || ub.empty()) {
    lb.resize(D);
    ub.resize(D);
    for (int j = 0; j < D; ++j) {
      const double v = x[j];
      const double half = std::max(2.0 * std::abs(v), 5e-4);
      lb[j] = v - half;
      ub[j] = v + half;
    }
  }

  using DEParams = boost::math::optimization::differential_evolution_parameters<
      std::vector<double>>;
  DEParams params;
  params.lower_bounds = lb;
  params.upper_bounds = ub;
  params.mutation_factor = mutation_factor;
  params.crossover_probability = crossover_probability;
  params.NP = NP_factor * static_cast<std::size_t>(D);
  params.max_generations = max_generations;
  params.threads = 1;

  std::vector<double> ig(x.data(), x.data() + D);
  params.initial_guess = &ig;

  auto cost = [&](const std::vector<double> &v) {
    const Eigen::VectorXd ev = Eigen::Map<const Eigen::VectorXd>(v.data(), D);
    return f(ev).squaredNorm();
  };

  std::mt19937_64 rng(seed > 0
                          ? static_cast<std::uint64_t>(seed)
                          : static_cast<std::uint64_t>(std::random_device{}()));
  const auto best =
      boost::math::optimization::differential_evolution(cost, params, rng);
  x = Eigen::Map<const Eigen::VectorXd>(best.data(), D);
  return 0;
}

int LineSearchSolver::minimize(Eigen::VectorXd &x, ResidualFn f,
                               JacobianFn /*jac*/, int /*n_vals*/) const {
  const int D = static_cast<int>(x.size());
  if (D == 0)
    return 0;

  const PowellDirectionSet powell{f};

  // Direction set, initialised to the unit basis.
  std::vector<Eigen::VectorXd> dirs(D, Eigen::VectorXd::Zero(D));
  for (int i = 0; i < D; ++i)
    dirs[i][i] = 1.0;

  for (int iter = 0; iter < max_iter; ++iter) {
    const Eigen::VectorXd p0 = x;
    const double fp = powell.phi(x);

    const auto s = powell.sweep(x, dirs);
    const double fret = powell.phi(x);

    // Converged once a whole sweep barely moves the parameters.
    if ((x - p0).norm() <= xtol)
      break;

    // Adopt the conjugate direction iff Powell's criterion holds, then replace
    // the most-effective old direction with it.
    powell.conjugate_direction(p0, x, fp, fret, s.del)
        .transform([&](const Eigen::VectorXd &xi) {
          powell.line_min(x, xi);
          dirs[s.ibig] = dirs.back();
          dirs.back() = xi;
          return 0;
        });
  }
  return 0;
}
void LineSearchSolver::PowellDirectionSet::line_min(
    Eigen::VectorXd &x, const Eigen::VectorXd &dir) const {
  const double before = phi(x);
  const Eigen::VectorXd save = x;
  linmin(x, dir, f);
  if (phi(x) <= before) {
    return;
  }
  x = save; // +dir worsened φ — try the opposite direction
  const Eigen::VectorXd neg = -dir;
  linmin(x, neg, f);
  if (phi(x) > before) {
    x = save;
  }
}

LineSearchSolver::PowellDirectionSet::SweepResult
LineSearchSolver::PowellDirectionSet::sweep(
    Eigen::VectorXd &x, const std::vector<Eigen::VectorXd> &dirs) const {
  SweepResult s;
  for (int i = 0; i < static_cast<int>(dirs.size()); ++i) {
    const double before = phi(x);
    line_min(x, dirs[i]);
    const double dec = before - phi(x);
    if (dec > s.del) {
      s.del = dec;
      s.ibig = i;
    }
  }
  return s;
}

std::optional<Eigen::VectorXd>
LineSearchSolver::PowellDirectionSet::conjugate_direction(
    const Eigen::VectorXd &p0, const Eigen::VectorXd &x, double fp, double fret,
    double del) const {
  const Eigen::VectorXd xi = x - p0;  // net direction moved this sweep
  const Eigen::VectorXd ptt = x + xi; // extrapolated point 2x − p0
  const double fptt = phi(ptt);
  if (fptt >= fp) {
    return std::nullopt;
  }
  const double t =
      2.0 * (fp - 2.0 * fret + fptt) * std::pow(fp - fret - del, 2) -
      del * std::pow(fp - fptt, 2);
  return t < 0.0 ? std::optional<Eigen::VectorXd>{std::move(xi)} : std::nullopt;
}

Solver make_default_solver(int max_iter, double xtol, double ftol) {
  return Solver(EigenLMSolver{max_iter, xtol, ftol});
}

} // namespace potfit
