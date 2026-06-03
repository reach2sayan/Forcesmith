#include "potfit/force/ml_force.hpp"

#include <algorithm>
#include <cmath>

namespace potfit {

// ---- LinearHead
// --------------------------------------------------------------

double LinearHead::energy_impl(const Eigen::VectorXd &D) const {
  Eigen::VectorXd c(coeffs.size());
  std::ranges::transform(coeffs, c.begin(), &Param::value);
  double e = bias.value + c.dot(D);
  return e;
}

Eigen::VectorXd LinearHead::grad_impl(const Eigen::VectorXd &) const {
  Eigen::VectorXd g(static_cast<Eigen::Index>(coeffs.size()));
  std::ranges::transform(coeffs, g.data(), &Param::value);
  return g;
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

MLPHead MLPHead::make(const std::vector<int> &sizes, Act act,
                      std::uint64_t seed) {
  MLPHead h;
  h.act = act;
  std::uint64_t s = seed * 2654435761u + 1;
  auto next = [&s]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((s >> 11) & 0xFFFFF) / 0xFFFFF * 2.0 - 1.0;
  };
  for (std::size_t l = 0; l + 1 < sizes.size(); ++l) {
    const int in = sizes[l], out = sizes[l + 1];
    const double scale = std::sqrt(1.0 / std::max(1, in)); // Xavier-ish
    Eigen::MatrixXd Wl(out, in);
    for (int r = 0; r < out; ++r) {
      for (int c = 0; c < in; ++c) {
        Wl(r, c) = next() * scale;
      }
    }
    h.W.push_back(std::move(Wl));
    h.b.push_back(Eigen::VectorXd::Zero(out));
  }
  return h;
}

double MLPHead::energy_impl(const Eigen::VectorXd &D) const {
  Eigen::VectorXd a = D;
  for (std::size_t l = 0; l + 1 < W.size(); ++l) {
    a = activate(W[l] * a + b[l]);
  }
  a = W.back() * a + b.back();
  return a[0];
}


Eigen::VectorXd MLPHead::grad_impl(const Eigen::VectorXd &D) const {
  const std::size_t L = W.size();
  std::vector<Eigen::VectorXd> z(L);
  Eigen::VectorXd a = D;
  for (std::size_t l = 0; l < L; ++l) {
    const auto &Wl = W[l];
    const auto &bl = b[l];
    auto &zl = z[l];
    zl.noalias() = Wl * a;
    zl += bl;
    const bool is_output_layer = (l + 1 == L);
    a = is_output_layer ? zl : activate(zl);
  }
  // delta_l = dE/dz_l; top layer is linear so delta_{L-1} = 1.
  Eigen::VectorXd delta = Eigen::VectorXd::Ones(1);
  for (std::size_t l = L - 1; l > 0; --l) {
    delta = W[l].transpose() * delta;
    delta.array() *= activate_deriv(z[l - 1]).array();
  }
  return W[0].transpose() * delta; // de/dD = dE/da_0
}

std::size_t MLPHead::param_count() const {
  auto wnb = std::views::zip(W, b);
  return std::transform_reduce(wnb.begin(), wnb.end(), std::size_t{0},
                               std::plus<>{}, [](const auto &wb) {
                                 const auto &[Wl, bl] = wb;
                                 return static_cast<std::size_t>(Wl.size() +
                                                                 bl.size());
                               });
}

void MLPHead::gather_params(Eigen::VectorXd &dst, std::size_t off) const {
  auto o = static_cast<Eigen::Index>(off);
  for (std::size_t l = 0; l < W.size(); ++l) {
    dst.segment(o, W[l].size()) =
        Eigen::Map<const Eigen::VectorXd>(W[l].data(), W[l].size());
    o += W[l].size();
    dst.segment(o, b[l].size()) = b[l];
    o += b[l].size();
  }
}

void MLPHead::scatter_params(const Eigen::VectorXd &src, std::size_t off) {
  auto o = static_cast<Eigen::Index>(off);
  for (std::size_t l = 0; l < W.size(); ++l) {
    Eigen::Map<Eigen::VectorXd>(W[l].data(), W[l].size()) =
        src.segment(o, W[l].size());
    o += W[l].size();
    b[l] = src.segment(o, b[l].size());
    o += b[l].size();
  }
}

Eigen::VectorXd MLPHead::all_values() const {
  Eigen::VectorXd v(static_cast<Eigen::Index>(param_count()));
  gather_params(v, 0);
  return v;
}

void MLPHead::set_all_values(const Eigen::VectorXd &v) { scatter_params(v, 0); }

std::vector<int> MLPHead::architecture() const {
  std::vector<int> a;
  if (!W.empty()) {
    a.reserve(W.size() + 1);
    a.push_back(static_cast<int>(W.front().cols()));
    std::ranges::transform(W, std::back_inserter(a), [](const auto &Wl) {
      return static_cast<int>(Wl.rows());
    });
  }
  return a;
}

Eigen::VectorXd MLPHead::activate(const Eigen::VectorXd &z) const {
  switch (act) {
  case Act::SiLU:
    return (z.array() / (1.0 + (-z.array()).exp())).matrix();
  case Act::Tanh:
  default:
    return z.array().tanh().matrix();
  }
}

Eigen::VectorXd MLPHead::activate_deriv(const Eigen::VectorXd &z) const {
  switch (act) {
  case Act::SiLU: {
    Eigen::ArrayXd s = 1.0 / (1.0 + (-z.array()).exp());
    return (s * (1.0 + z.array() * (1.0 - s))).matrix();
  }
  case Act::Tanh:
  default: {
    Eigen::ArrayXd t = z.array().tanh();
    return (1.0 - t * t).matrix();
  }
  }
}

} // namespace potfit
