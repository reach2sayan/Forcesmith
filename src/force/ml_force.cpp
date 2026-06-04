#include "potfit/force/ml_force.hpp"

#include <algorithm>
#include <ranges>
#include <span>

namespace potfit {

// ---- LinearHead
// --------------------------------------------------------------

double LinearHead::energy(const Eigen::VectorXd &D) const {
  Eigen::VectorXd c(coeffs.size());
  std::ranges::transform(coeffs, c.begin(), &Param::value);
  double e = bias.value + c.dot(D);
  return e;
}

Eigen::VectorXd LinearHead::grad(const Eigen::VectorXd &) const {
  Eigen::VectorXd g(static_cast<Eigen::Index>(coeffs.size()));
  std::ranges::transform(coeffs, g.data(), &Param::value);
  return g;
}

// ∂E/∂θ for free params, in gather order [free coeffs…, bias?]: ∂E/∂coeff_k =
// D_k, ∂E/∂bias = 1.
Eigen::VectorXd LinearHead::param_grad(const Eigen::VectorXd &D) const {
  Eigen::VectorXd g(static_cast<Eigen::Index>(param_count()));
  Eigen::Index o = 0;
  for (auto [c, d] : std::views::zip(
           coeffs, std::span(D.data(), static_cast<std::size_t>(D.size())))) {
    if (!c.fixed) {
      g[o++] = d;
    }
  }
  if (!bias.fixed) {
    g[o++] = 1.0;
  }
  return g;
}

// ∂(∂E/∂D)/∂θ. ∂E/∂D = coeffs, so the column for free coeff k is e_k and the
// (free) bias column is zero ⇒ M is the identity restricted to free-coeff
// columns.
Eigen::MatrixXd LinearHead::dgrad_dparam(const Eigen::VectorXd &D) const {
  Eigen::MatrixXd M =
      Eigen::MatrixXd::Zero(D.size(), static_cast<Eigen::Index>(param_count()));
  Eigen::Index o = 0;
  for (auto [k, c] : coeffs | std::views::enumerate) {
    if (!c.fixed) {
      M(k, o++) =
          1.0; // descriptor dim k ↔ ∂E/∂D_k; free bias leaves a 0 column
    }
  }
  return M;
}

std::vector<Param *> LinearHead::field_ptrs() {
  std::vector<Param *> f;
  f.reserve(coeffs.size() + 1);
  auto ptrs = coeffs | std::views::transform([](auto &c) { return &c; });
  std::ranges::copy(ptrs, std::back_inserter(f));
  f.push_back(&bias);
  return f;
}

std::vector<const Param *> LinearHead::field_ptrs() const {
  std::vector<const Param *> f;
  f.reserve(coeffs.size() + 1);
  std::ranges::transform(coeffs, std::back_inserter(f),
                         [](const auto &c) { return &c; });

  f.push_back(&bias);
  return f;
}

} // namespace potfit
