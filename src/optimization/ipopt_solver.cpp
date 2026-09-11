#include "forcesmith/optimization/ipopt_solver.hpp"

#include "forcesmith/optimization/fd_jacobian.hpp"

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>

#include <algorithm>
#include <functional>
#include <ranges>

namespace forcesmith {

namespace {

constexpr double kIpoptInf = 2e19;

class ForcesmithTNLP final : public Ipopt::TNLP {
public:
  ForcesmithTNLP(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac, int n_vals,
                 const Eigen::VectorXd &lower, const Eigen::VectorXd &upper)
      : x_(x), f_(std::move(f)), jac_(std::move(jac)),
        n_(static_cast<int>(x.size())), n_vals_(n_vals), lower_(lower),
        upper_(upper) {}

  bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m, Ipopt::Index &nnz_jac_g,
                    Ipopt::Index &nnz_h_lag,
                    IndexStyleEnum &index_style) override {
    n = n_;
    m = 0;
    nnz_jac_g = 0;
    nnz_h_lag = 0;
    index_style = TNLP::C_STYLE;
    return true;
  }

  bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l, Ipopt::Number *x_u,
                       Ipopt::Index /*m*/, Ipopt::Number * /*g_l*/,
                       Ipopt::Number * /*g_u*/) override {
    for (Ipopt::Index j = 0; j < n; ++j) {
      x_l[j] = std::max(lower_[j], -kIpoptInf);
      x_u[j] = std::min(upper_[j], kIpoptInf);
    }
    return true;
  }

  bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x,
                          bool /*init_z*/, Ipopt::Number * /*z_L*/,
                          Ipopt::Number * /*z_U*/, Ipopt::Index /*m*/,
                          bool /*init_lambda*/,
                          Ipopt::Number * /*lambda*/) override {
    if (init_x) {
      std::copy_n(x_.data(), n, x);
    }
    return true;
  }

  bool eval_f(Ipopt::Index n, const Ipopt::Number *x, bool /*new_x*/,
              Ipopt::Number &obj_value) override {
    const Eigen::VectorXd r = f_(Eigen::Map<const Eigen::VectorXd>(x, n));
    obj_value = 0.5 * r.squaredNorm();
    return true;
  }

  bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool /*new_x*/,
                   Ipopt::Number *grad_f) override {
    const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x, n);
    const Eigen::VectorXd r = f_(xv);
    Eigen::MatrixXd J(n_vals_, n);
    if (jac_) {
      jac_(xv, J);
    } else {
      finite_diff_jacobian(xv, r.size(), J);
    }
    Eigen::Map<Eigen::VectorXd>(grad_f, n) = J.transpose() * r;
    return true;
  }

  bool eval_g(Ipopt::Index /*n*/, const Ipopt::Number * /*x*/, bool /*new_x*/,
              Ipopt::Index /*m*/, Ipopt::Number * /*g*/) override {
    return true; // no constraints
  }

  bool eval_jac_g(Ipopt::Index /*n*/, const Ipopt::Number * /*x*/,
                  bool /*new_x*/, Ipopt::Index /*m*/, Ipopt::Index /*nele_jac*/,
                  Ipopt::Index * /*iRow*/, Ipopt::Index * /*jCol*/,
                  Ipopt::Number * /*values*/) override {
    return true; // no constraints
  }

  bool eval_h(Ipopt::Index, const Ipopt::Number *, bool, Ipopt::Number,
              Ipopt::Index, const Ipopt::Number *, bool, Ipopt::Index,
              Ipopt::Index *, Ipopt::Index *, Ipopt::Number *) override {
    return false; // L-BFGS approximation
  }

  void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                         const Ipopt::Number *x, const Ipopt::Number *,
                         const Ipopt::Number *, Ipopt::Index,
                         const Ipopt::Number *, const Ipopt::Number *,
                         Ipopt::Number, const Ipopt::IpoptData *,
                         Ipopt::IpoptCalculatedQuantities *) override {
    std::copy_n(x, n, x_.data());
    status_ = status;
  }

  Ipopt::SolverReturn status() const { return status_; }

private:
  void finite_diff_jacobian(const Eigen::VectorXd &x, Eigen::Index m,
                            Eigen::MatrixXd &J) const {
    J.resize(m, x.size());
    opt::fd_jacobian(f_, x, J);
  }

  Eigen::VectorXd &x_;
  ResidualFn f_;
  JacobianFn jac_;
  int n_;
  int n_vals_;
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
  Ipopt::SolverReturn status_{Ipopt::INTERNAL_ERROR};
};

} // namespace

int IpoptSolver::minimize(Eigen::VectorXd &x, ResidualFn f, JacobianFn jac,
                          int n_vals, const Eigen::VectorXd &lower,
                          const Eigen::VectorXd &upper) const {
  Ipopt::SmartPtr<ForcesmithTNLP> tnlp =
      new ForcesmithTNLP(x, std::move(f), std::move(jac), n_vals, lower, upper);

  Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
  app->Options()->SetNumericValue("tol", tol);
  app->Options()->SetNumericValue("acceptable_tol", acceptable_tol);
  app->Options()->SetIntegerValue("max_iter", max_iter);
  app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  if (silent) {
    app->Options()->SetIntegerValue("print_level", 0);
    app->Options()->SetStringValue("sb", "yes"); // suppress startup banner
  }

  const Ipopt::ApplicationReturnStatus init_status = app->Initialize();
  if (init_status != Ipopt::Solve_Succeeded) {
    return 0;
  }

  Ipopt::SmartPtr<Ipopt::TNLP> base = Ipopt::GetRawPtr(tnlp);
  app->OptimizeTNLP(base);

  switch (tnlp->status()) {
  case Ipopt::SUCCESS:
    return 1;
  case Ipopt::STOP_AT_ACCEPTABLE_POINT:
    return 2;
  default:
    return 0;
  }
}

} // namespace forcesmith
