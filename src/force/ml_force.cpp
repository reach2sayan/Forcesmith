#include "potfit/force/ml_force.hpp"

#include <algorithm>
#include <cmath>
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

MLPHead MLPHead::make(const std::vector<int> &sizes, Act act,
                      std::uint64_t seed) {
  MLPHead h;
  h.act = act;
  std::uint64_t s = seed * 2654435761u + 1;
  auto next = [&s]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((s >> 11) & 0xFFFFF) / 0xFFFFF * 2.0 - 1.0;
  };
  for (auto [in, out] : std::views::zip(sizes, sizes | std::views::drop(1))) {
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

double MLPHead::energy(const Eigen::VectorXd &D) const {
  Eigen::VectorXd a = D;
  for (auto [Wl, bl] : std::views::zip(W, b) | std::views::take(W.size() - 1)) {
    a = activate(Wl * a + bl);
  }
  a = W.back() * a + b.back();
  return a[0];
}

Eigen::VectorXd MLPHead::grad(const Eigen::VectorXd &D) const {
  std::vector<Eigen::VectorXd> z(W.size());
  Eigen::VectorXd a = D;
  for (auto [Wl, bl, zl] : std::views::zip(W, b, z)) {
    zl.noalias() = Wl * a;
    zl += bl;
    const bool is_output_layer = (&zl == &z.back());
    a = is_output_layer ? zl : activate(zl);
  }
  // delta_l = dE/dz_l; top layer is linear so delta_{L-1} = 1. Walk the hidden
  // layers back-to-front, pairing W[l] with the pre-activation z[l-1] it feeds.
  Eigen::VectorXd delta = Eigen::VectorXd::Ones(1);
  for (auto [Wl, zprev] :
       std::views::zip(W | std::views::drop(1), z) | std::views::reverse) {
    delta = Wl.transpose() * delta;
    delta.array() *= activate_deriv(zprev).array();
  }
  return W[0].transpose() * delta; // de/dD = dE/da_0
}

// ∂E/∂θ via backprop-to-weights: ∂E/∂W_l = δ_l a_{l-1}ᵀ, ∂E/∂b_l = δ_l. Same
// forward + δ recurrence as grad(); flattened in gather_params() order.
Eigen::VectorXd MLPHead::param_grad(const Eigen::VectorXd &D) const {
  const std::size_t L = W.size();
  std::vector<Eigen::VectorXd> z(L),
      ain(L); // ain[l] = a_{l-1} (input to layer l)
  Eigen::VectorXd a = D;
  for (auto [Wl, bl, zl, al] : std::views::zip(W, b, z, ain)) {
    al = a;
    zl.noalias() = Wl * a;
    zl += bl;
    a = (&zl == &z.back()) ? zl : activate(zl); // top layer is linear
  }

  // δ_l = ∂E/∂z_l, back-to-front (a reverse scan, so kept index-based).
  std::vector<Eigen::VectorXd> delta(L);
  delta[L - 1] = Eigen::VectorXd::Ones(1);
  for (std::size_t l = L - 1; l > 0; --l) {
    const std::size_t prev = l - 1;
    delta[prev].noalias() = W[l].transpose() * delta[l];
    delta[prev].array() *= activate_deriv(z[prev]).array();
  }

  // ∂E/∂W_l = δ_l a_{l-1}ᵀ, ∂E/∂b_l = δ_l, flattened in gather_params() order.
  Eigen::VectorXd g(static_cast<Eigen::Index>(param_count()));
  Eigen::Index o = 0;
  for (auto [dl, al] : std::views::zip(delta, ain)) {
    const Eigen::MatrixXd GW = dl * al.transpose(); // out_l × in_l
    g.segment(o, GW.size()) =
        Eigen::Map<const Eigen::VectorXd>(GW.data(), GW.size());
    o += GW.size();
    g.segment(o, dl.size()) = dl;
    o += dl.size();
  }
  return g;
}

// M = ∂(∂E/∂D)/∂θ, the mixed second derivative the force-residual Jacobian
// columns need. Computed exactly by reverse-mode AD of the scalar φ = u·(∂E/∂D)
// through the AUGMENTED graph (forward pass + the δ backprop that produces
// ∂E/∂D), once per u = e_s. Backpropagating φ through that graph yields ∇_θφ =
// row s of M — the explicit-W₀ term, the σ″ coupling and the weight/bias terms
// all emerge from the adjoints, no per-parameter case analysis. S sweeps, each
// O(network), vs the FD path's O(param_count) full passes.
Eigen::MatrixXd MLPHead::dgrad_dparam(const Eigen::VectorXd &D) const {
  const int L = static_cast<int>(W.size());
  const int S = static_cast<int>(D.size());
  const int P = static_cast<int>(param_count());

  // Forward: z_l, layer input a_{l-1}, and σ′/σ″ at the hidden pre-activations
  // (left empty for the linear output layer).
  std::vector<Eigen::VectorXd> z(L), ain(L), sp(L), spp(L);
  Eigen::VectorXd a = D;
  for (auto [Wl, bl, zl, al, spl, sppl] :
       std::views::zip(W, b, z, ain, sp, spp)) {
    al = a;
    zl.noalias() = Wl * a;
    zl += bl;
    if (&zl != &z.back()) {
      spl = activate_deriv(zl);
      sppl = activate_deriv2(zl);
      a = activate(zl);
    } else {
      a = zl;
    }
  }
  // Reverse: δ_l and the pre-σ′ signal r_l = W_{l+1}ᵀ δ_{l+1} (so δ_l =
  // σ′⊙r_l).
  std::vector<Eigen::VectorXd> delta(L), r(L);
  delta.back() = Eigen::VectorXd::Ones(1);
  for (std::size_t l = L - 1; l > 0; --l) {
    const std::size_t prev = l - 1;
    r[prev].noalias() = W[l].transpose() * delta[l];
    delta[prev].resize(r[prev].size());
    delta[prev].array() = sp[prev].array() * r[prev].array();
  }

  // Adjoint buffers, sized once and zeroed per sweep.
  Eigen::MatrixXd M(S, P);
  std::vector<Eigen::VectorXd> dbar(L), zbar(L), bbar(L);
  std::vector<Eigen::MatrixXd> Wbar(L);
  for (int l = 0; l < L; ++l) {
    dbar[l].resizeLike(delta[l]);
    zbar[l].resizeLike(z[l]);
    bbar[l].resizeLike(b[l]);
    Wbar[l].resizeLike(W[l]);
  }

  for (int s : std::views::iota(0, S)) {
    for (auto [db, zb, bb, Wb] : std::views::zip(dbar, zbar, bbar, Wbar)) {
      db.setZero();
      zb.setZero();
      bb.setZero();
      Wb.setZero();
    }

    // G: g = W₀ᵀδ₀, φ = e_s·g ⇒ δ̄₀ = W₀ e_s, and the explicit ∂g_s/∂W₀[:,s] =
    // δ₀.
    dbar[0] = W[0].col(s);
    Wbar[0].col(s) += delta[0];

    // Reverse the δ recurrence (forward order was l=L-2…0, so reverse is
    // 0…L-2).
    for (int l = 0; l <= L - 2; ++l) {
      const Eigen::VectorXd tbar = (sp[l].array() * dbar[l].array()).matrix();
      zbar[l] = (spp[l].array() * r[l].array() * dbar[l].array()).matrix();
      dbar[l + 1].noalias() += W[l + 1] * tbar;
      Wbar[l + 1].noalias() += delta[l + 1] * tbar.transpose();
    }

    // Reverse the forward pass: a_l only feeds z_{l+1}, so ā_l = W_{l+1}ᵀ
    // z̄_{l+1} (zero at the top hidden layer); z̄_l adds σ′⊙ā_l to its σ″ part.
    for (int l = L - 2; l >= 0; --l) {
      if (l < L - 2) {
        const Eigen::VectorXd abar = W[l + 1].transpose() * zbar[l + 1];
        zbar[l] += (sp[l].array() * abar.array()).matrix();
      }
      Wbar[l].noalias() += zbar[l] * ain[l].transpose();
      bbar[l] += zbar[l];
    }

    // Row s of M = ∇_θφ, flattened in gather_params() order [W₀,b₀,W₁,b₁,…].
    Eigen::Index o = 0;
    for (auto [Wb, bb] : std::views::zip(Wbar, bbar)) {
      M.block(s, o, 1, Wb.size()) =
          Eigen::Map<const Eigen::RowVectorXd>(Wb.data(), Wb.size());
      o += Wb.size();
      M.block(s, o, 1, bb.size()) = bb.transpose();
      o += bb.size();
    }
  }

  return M;
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
  for (const auto &[Wl, bl] : std::views::zip(W, b)) {
    dst.segment(o, Wl.size()) =
        Eigen::Map<const Eigen::VectorXd>(Wl.data(), Wl.size());
    o += Wl.size();
    dst.segment(o, bl.size()) = bl;
    o += bl.size();
  }
}

void MLPHead::scatter_params(const Eigen::VectorXd &src, std::size_t off) {
  auto o = static_cast<Eigen::Index>(off);
  for (auto &&[Wl, bl] : std::views::zip(W, b)) {
    Eigen::Map<Eigen::VectorXd>(Wl.data(), Wl.size()) =
        src.segment(o, Wl.size());
    o += Wl.size();
    bl = src.segment(o, bl.size());
    o += bl.size();
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

// σ″: Tanh  −2σ(1−σ²) with σ=tanh;  SiLU  s(1−s)(2 + z(1−2s)) with
// s=sigmoid(z).
Eigen::VectorXd MLPHead::activate_deriv2(const Eigen::VectorXd &z) const {
  switch (act) {
  case Act::SiLU: {
    Eigen::ArrayXd s = 1.0 / (1.0 + (-z.array()).exp());
    return (s * (1.0 - s) * (2.0 + z.array() * (1.0 - 2.0 * s))).matrix();
  }
  case Act::Tanh:
  default: {
    Eigen::ArrayXd t = z.array().tanh();
    return (-2.0 * t * (1.0 - t * t)).matrix();
  }
  }
}

} // namespace potfit
