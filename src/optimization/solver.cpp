#include "forcesmith/optimization/solver.hpp"

#include "forcesmith/optimization/fd_jacobian.hpp"

#include "forcesmith/optimization/line_search.hpp"

#include <Eigen/Cholesky>
#include <boost/math/optimization/differential_evolution.hpp>
#include <unsupported/Eigen/NonLinearOptimization>

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <ranges>
#include <vector>

namespace {
extern "C" void openblas_set_num_threads(int) __attribute__((weak));

const int kThreadingHygiene = [] {
  if (openblas_set_num_threads)
    openblas_set_num_threads(1);
  return 0;
}();
} // namespace

namespace forcesmith {

namespace {

struct FunctorAdapter {
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  ResidualFn fn;
  int n_inputs;
  int n_values;
  JacobianFn
      jac; // optional; when set, df delegates to it (ForcesmithFunctor::df)

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const {
    fvec = std::invoke(fn, x);
    return 0;
  }

  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const {
    if (jac) {
      std::invoke(jac, x, fjac);
      return 0;
    }
    opt::fd_jacobian(fn, x, fjac);
    return 0;
  }

  constexpr int inputs() const { return n_inputs; }
  constexpr int values() const { return n_values; }
};

} // namespace

int EigenLMSolver::minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
                            int n_vals, const Eigen::VectorXd & /*lower*/,
                            const Eigen::VectorXd & /*upper*/) const {
  FunctorAdapter adapter{std::move(f), static_cast<int>(x.size()), n_vals,
                         std::move(jac)};
  Eigen::LevenbergMarquardt<FunctorAdapter> lm(adapter);
  namespace LM = Eigen::LevenbergMarquardtSpace;
  lm.parameters.maxfev = std::max(max_iter, 1) * 100 + 100;
  lm.parameters.xtol = xtol;
  lm.parameters.ftol = ftol;
  if (lm.minimizeInit(x) == LM::ImproperInputParameters) {
    return static_cast<int>(LM::ImproperInputParameters);
  }
  LM::Status status =
      LM::Running; // minimizeInit leaves NotStarted; prime the loop
  for (int i = 0; i < max_iter && status == LM::Running; ++i)
    status = lm.minimizeOneStep(x);
  return status == LM::Running ? 2 : static_cast<int>(status);
}

int EigenHybridSolver::minimize(Eigen::VectorXd &x, ResidualFn f,
                                JacobianFn /*jac*/, int /*n_vals*/,
                                const Eigen::VectorXd & /*lower*/,
                                const Eigen::VectorXd & /*upper*/) const {
  const int D = static_cast<int>(x.size());

  struct GradFunctor {
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> fn;
    int D;

    int operator()(const Eigen::VectorXd &xv, Eigen::VectorXd &grad) const {
      opt::fd_gradient(fn, xv, grad);
      return 0;
    }

    constexpr int inputs() const { return D; }
    constexpr int values() const { return D; } // square system
  };

  GradFunctor gf{.fn = std::move(f), .D = D};
  Eigen::HybridNonLinearSolver<GradFunctor> solver(gf);
  solver.parameters.maxfev = max_iter * D;
  solver.parameters.xtol = xtol;
  return static_cast<int>(solver.solveNumericalDiff(x));
}

int BoostDESolver::minimize(Eigen::VectorXd &x, ResidualFn f,
                            JacobianFn /*jac*/, int /*n_vals*/,
                            const Eigen::VectorXd &lower,
                            const Eigen::VectorXd &upper) const {
  const int D = static_cast<int>(x.size());

  std::vector<double> lb(D), ub(D);
  for (int j = 0; j < D; ++j) {
    const double v = x[j];
    const double half = std::max(2.0 * std::abs(v), 5e-4);
    lb[j] = std::isfinite(lower[j]) ? lower[j] : v - half;
    ub[j] = std::isfinite(upper[j]) ? upper[j] : v + half;
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

  std::vector<double> ig(D);
  for (int j = 0; j < D; ++j)
    ig[j] = std::clamp(x[j], lb[j], ub[j]);
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
                               JacobianFn /*jac*/, int /*n_vals*/,
                               const Eigen::VectorXd & /*lower*/,
                               const Eigen::VectorXd & /*upper*/) const {
  const int D = static_cast<int>(x.size());
  if (D == 0) {
    return 0;
  }

  const PowellDirectionSet powell{f};

  std::vector<Eigen::VectorXd> dirs(D, Eigen::VectorXd::Zero(D));
  for (int i = 0; i < D; ++i) {
    dirs[i] = Eigen::VectorXd::Unit(D, i);
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    const Eigen::VectorXd p0 = x;
    const double fp = powell.phi(x);

    const auto s = powell.sweep(x, dirs);
    const double fret = powell.phi(x);

    if ((x - p0).norm() <= xtol) {
      break;
    }

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
  for (const auto i : std::views::iota(std::size_t{0}, dirs.size())) {
    const double before = phi(x);
    line_min(x, dirs[i]);
    const double dec = before - phi(x);
    if (dec > s.del) {
      s.del = dec;
      s.ibig = static_cast<int>(i);
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

int NormalEquationsSolver::minimize(Eigen::VectorXd &x, ResidualFn f,
                                    JacobianFn jac, int n_vals,
                                    const Eigen::VectorXd & /*lower*/,
                                    const Eigen::VectorXd & /*upper*/) const {
  const Eigen::Index P = x.size();
  if (P == 0 || n_vals == 0) {
    return 0;
  }

  const Eigen::VectorXd r0 = std::invoke(f, x);
  Eigen::MatrixXd J(n_vals, P);
  std::invoke(jac, x, J);

  Eigen::MatrixXd AtA = J.transpose() * J; // P×P (small)
  const Eigen::VectorXd Atb = J.transpose() * r0;

  double lambda = ridge;
  if (lambda <= 0.0) {
    lambda = 1e-8 * std::max(AtA.diagonal().mean(), 1.0);
  }
  AtA.diagonal().array() += lambda;

  x -= AtA.ldlt().solve(Atb);
  return 0;
}

Solver make_default_solver(int max_iter, double xtol, double ftol) {
  return Solver(
      EigenLMSolver{.max_iter = max_iter, .xtol = xtol, .ftol = ftol});
}

} // namespace forcesmith
