#include "forcesmith/force/ml_force.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <ranges>
#include <span>

namespace forcesmith {

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

LinearHead LinearHead::remapped(
    const std::vector<std::optional<Eigen::Index>> &map) const {
  LinearHead out;
  out.bias = bias; // keep bias value + fixed flag
  const bool fixed = coeffs.empty() ? false : coeffs.front().fixed;
  out.coeffs.assign(map.size(), Param{0.0, fixed});
  for (auto [k, src] : map | std::views::enumerate) {
    if (src) {
      out.coeffs[static_cast<std::size_t>(k)] =
          coeffs[static_cast<std::size_t>(*src)];
    }
  }
  return out;
}

LinearHead LinearHead::zero_like(Eigen::Index n) {
  LinearHead out;
  out.coeffs.assign(static_cast<std::size_t>(n), Param{0.0, false});
  return out; // default bias{0.0, true}
}

nlohmann::json build_json_from_head(const LinearHead &lh) {
  nlohmann::json head;
  head["type"] = lh.type_tag();
  const Eigen::VectorXd vals = lh.all_values();
  const std::vector<int> arch = lh.architecture();
  const int n = arch.empty() ? 0 : arch[0];
  head["coeffs"] = nlohmann::json::array();
  for (int k = 0; k < n; ++k) {
    head["coeffs"].push_back(vals[k]);
  }
  head["bias"] = vals[n];
  return head;
}

} // namespace forcesmith
