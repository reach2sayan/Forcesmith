#include "forcesmith/potentials/soap.hpp"

#include <Eigen/Eigenvalues>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "forcesmith/force/descriptor_layout.hpp" // pair_ordinal

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

namespace forcesmith {

// Tabulated radial projection J(a,l)(r) = ∫ r'² φ_a(r') e^{-(r'²+r²)/2σ²} i_l(r
// r'/σ²) dr' as a cubic spline in r over [0,rcut], one per (a,l). Replacing the
// per-neighbour adaptive Bessel quadrature with a spline lookup is the dominant
// SOAP speedup.
struct SoapRadialTable {
  int nm = 0;
  int lm1 = 0; // l_max + 1
  std::vector<boost::math::interpolators::cardinal_cubic_b_spline<double>> spl;
  constexpr double J(int a, int l, double r) const {
    return spl[static_cast<std::size_t>(a) * lm1 + l](r);
  }
  // dJ/dr from the spline derivative (used by the analytic radial gradient).
  constexpr double dJdr(int a, int l, double r) const {
    return spl[static_cast<std::size_t>(a) * lm1 + l].prime(r);
  }
};

namespace {

// Modified spherical Bessel i_l(x) = sqrt(π/2x)·I_{l+1/2}(x), with the x→0
// limits.
FORCE_INLINE double sph_bessel_i(int l, double x) {
  if (x < 1e-12) {
    return l == 0 ? 1.0 : 0.0;
  }
  return std::sqrt(std::numbers::pi / (2.0 * x)) *
         boost::math::cyl_bessel_i(l + 0.5, x);
}

// Build the radial spline table by quadrature on a uniform r-grid. One-time:
// G·n_max·(l_max+1) integrals, then cubic splines (each holds its own copy).
std::shared_ptr<const SoapRadialTable>
build_radial_table(int nm, int lm, double rcut, double sigma) {
  constexpr int G = 512; // grid points on [0,rcut]; cubic error ~ (rcut/G)^4
  const double step = rcut / (G - 1);
  const double inv_2s2 = 1.0 / (2.0 * sigma * sigma);
  auto phi = [](int aparam, double r, double rc) {
    return std::pow(rc - r, aparam + 2);
  };
  auto t = std::make_shared<SoapRadialTable>();
  t->nm = nm;
  t->lm1 = lm + 1;
  t->spl.reserve(static_cast<std::size_t>(nm) * (lm + 1));
  std::vector<double> vals(G);
  for (int a = 0; a < nm; ++a) {
    for (int l = 0; l <= lm; ++l) {
      for (int g = 0; g < G; ++g) {
        const double r = g * step;
        auto integrand = [&](double rp) {
          const double x = r * rp / (sigma * sigma);
          return rp * rp * phi(a, rp, rcut) *
                 std::exp(-(rp * rp + r * r) * inv_2s2) * sph_bessel_i(l, x);
        };
        vals[static_cast<std::size_t>(g)] =
            boost::math::quadrature::gauss_kronrod<double, 31>::integrate(
                integrand, 0.0, rcut, 5, 1e-9);
      }
      t->spl.emplace_back(vals.data(), vals.size(), 0.0, step);
    }
  }
  return t;
}

} // namespace

namespace {

// Orthonormalization matrix β = S^{-1/2} of the polynomial radial basis. The
// overlap S_ab = ∫_0^rcut φ_a φ_b r² dr is closed form: with u = rcut−r,
//   S_ab = rcut^{a+b+7} [1/(a+b+5) − 2/(a+b+6) + 1/(a+b+7)].
// A pure function of (n_max, rcut) — safe to call from the const, parallel
// get_descriptor with no shared state.
Eigen::MatrixXd radial_beta(int n_max, double rcut) {
  const int n = n_max;
  const Eigen::MatrixXd S =
      Eigen::MatrixXd::NullaryExpr(n, n, [&](Eigen::Index a, Eigen::Index b) {
        const double s = static_cast<double>(a + b);
        return std::pow(rcut, s + 7.0) *
               (1.0 / (s + 5.0) - 2.0 / (s + 6.0) + 1.0 / (s + 7.0));
      });
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
  const Eigen::VectorXd inv_sqrt =
      es.eigenvalues().cwiseMax(1e-12).cwiseInverse().cwiseSqrt();
  return es.eigenvectors() * inv_sqrt.asDiagonal() *
         es.eigenvectors().transpose();
}

// Cosine cutoff f_c(r) = ½(1 + cos(π r / rcut)) for r < rcut, else 0. Shared by
// the coefficient fill and the gradient (which also needs f_c').
FORCE_INLINE double cutoff_fc(double r, double rc) {
  return 0.5 * (1.0 + std::cos(std::numbers::pi * r / rc));
}

// Canonical enumeration of the power-spectrum slots p^{αβ}_{n n' l} in flat
// order, invoking fn(flat_index, sa, sb, n, n2, l). The single source of truth
// for the descriptor layout — shared by power_spectrum() (which computes each
// value) and descriptor_index_map() (which re-slots them on a re-rank), so the
// two can never drift. Diagonal pairs (sa==sb) use n≤n'; off-diagonal use all
// n,n'.
template <class F> void for_each_ps_slot(int S, int nm, int lm, F &&fn) {
  Eigen::Index idx = 0;
  for (auto [sa, sb] : upper_triangle(S)) {
    for (int n = 0; n < nm; ++n) {
      const int n2start = (sa == sb) ? n : 0;
      for (int n2 = n2start; n2 < nm; ++n2) {
        for (int l = 0; l <= lm; ++l) {
          fn(idx, static_cast<int>(sa), static_cast<int>(sb), n, n2, l);
          ++idx;
        }
      }
    }
  }
}

// Power-spectrum descriptor length for an arbitrary species count (the same
// formula as SoapModel::descriptor_size, but parameterized by S for re-rank).
std::size_t ps_size(int S, int nm, int lm) {
  const std::size_t s = static_cast<std::size_t>(S);
  const std::size_t n = static_cast<std::size_t>(nm);
  const std::size_t same = s * (n * (n + 1) / 2);      // α=β: n≤n'
  const std::size_t cross = (s * (s - 1) / 2) * n * n; // α<β: all n,n'
  return (static_cast<std::size_t>(lm) + 1) * (same + cross);
}

// First flat index of each species-pair block, indexed by pair_ordinal. Used by
// the re-rank index map to translate a within-block offset between layouts.
std::vector<Eigen::Index> ps_pair_bases(int S, int nm, int lm) {
  std::vector<Eigen::Index> base(static_cast<std::size_t>(S) * (S + 1) / 2, -1);
  for_each_ps_slot(
      S, nm, lm, [&](Eigen::Index idx, int sa, int sb, int, int, int) {
        const std::size_t po = pair_ordinal(static_cast<std::size_t>(sa),
                                            static_cast<std::size_t>(sb),
                                            static_cast<std::size_t>(S));
        if (base[po] < 0) {
          base[po] = idx;
        }
      });
  return base;
}

} // namespace

// Optional single-threaded precompute of the radial basis. get_descriptor
// auto-computes β on demand if this was never called, so callers never have to.
void SoapModel::init_radial_basis() {
  beta = radial_beta(n_max, rcut);
  radial_ = build_radial_table(n_max, l_max, rcut, sigma);
}

std::size_t SoapModel::descriptor_size() const {
  return ps_size(static_cast<int>(ntypes), n_max, l_max);
}

// Orchestrator: lazily build the radial caches, then run the three steps
// (coefficients → power spectrum → analytic gradient) that share the
// expansion-coefficient store c and the flat (ch,n,l,m) index layout.
DescriptorValue SoapModel::get_descriptor(const Atom &a) const {
  const int nm = n_max;
  const int lm = l_max;

  // Ensure the radial orthonormalization and the tabulated radial projection
  // are built (lazy; prepare()'s serial warm-up does this once before any
  // parallel fill, so no race in the parallel path).
  if (beta.rows() != nm || beta.cols() != nm) {
    beta = radial_beta(nm, rcut);
  }
  if (!radial_) {
    radial_ = build_radial_table(nm, lm, rcut, sigma);
  }
  const SoapRadialTable &tab = *radial_;
  const double K = 4.0 * std::numbers::pi /
                   std::pow(2.0 * std::numbers::pi * sigma * sigma, 1.5);

  // Per-l power-spectrum weights w_l = √(8π²/(2l+1)).
  std::vector<double> wl(lm + 1);
  std::ranges::transform(std::views::iota(0, lm + 1), wl.begin(), [](int l) {
    return std::sqrt(8.0 * std::numbers::pi * std::numbers::pi /
                     (2.0 * l + 1.0));
  });

  // Step 1 — expansion coefficients.
  const Eigen::VectorXcd c = compute_coefficients(a, tab, K);

  // Step 2 — power spectrum + L2 normalization. ‖p‖ is kept for step 3.
  DescriptorValue out;
  out.values = power_spectrum(c, wl);
  const double norm = out.values.norm();
  if (norm > 1e-12) {
    out.values /= norm;
  }

  // Step 3 — analytic position gradient dD/dr.
  position_gradient(a, c, tab, K, wl, norm, out);

  return out;
}

Eigen::VectorXcd SoapModel::compute_coefficients(const Atom &a,
                                                 const SoapRadialTable &tab,
                                                 double K) const {
  const int S = static_cast<int>(ntypes);
  const int nm = n_max;
  const int lm = l_max;
  const int nlm = 2 * lm + 1;

  // Flat complex coefficient store c[channel][n][l][m] (see coeff_index).
  Eigen::VectorXcd c = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(S) *
                                              nm * (lm + 1) * nlm);

  for (const auto &nb : a.neighbors) {
    const double r = nb.dist.norm();
    if (r < 1e-12 || r >= rcut) {
      continue;
    }
    const int ch = static_cast<int>(nb.neighbor->type.index);
    if (ch < 0 || ch >= S) {
      continue;
    }
    const double fc = cutoff_fc(r, rcut);

    // Radial projection I(n,l) = Σ_a β(n,a) J(a,l)(r), J from the spline table.
    Eigen::MatrixXd J(nm, lm + 1); // basis × l
    for (int l = 0; l <= lm; ++l) {
      for (int abasis = 0; abasis < nm; ++abasis) {
        J(abasis, l) = tab.J(abasis, l, r);
      }
    }
    const Eigen::MatrixXd I = beta * J; // (n × l+1)

    // Angular factors Y*_{lm}(r̂_j).
    const Eigen::Vector3d u = nb.dist / r;
    const double theta = std::acos(std::clamp(u.z(), -1.0, 1.0));
    const double phi_ang = std::atan2(u.y(), u.x());

    for (int l = 0; l <= lm; ++l) {
      // Conjugated angular factors for this l, ordered m = −l..l (contiguous).
      Eigen::VectorXcd Yl(2 * l + 1);
      for (int m = -l; m <= l; ++m) {
        Yl(m + l) = std::conj(boost::math::spherical_harmonic(
            static_cast<unsigned>(l), m, theta, phi_ang));
      }
      // c[ch,n,l,·] += K·f_c·I(n,l)·Y* — vectorized over m (the contiguous
      // axis).
      for (int n = 0; n < nm; ++n) {
        c.segment(coeff_index(ch, n, l, -l), 2 * l + 1) +=
            (K * fc * I(n, l)) * Yl;
      }
    }
  }
  return c;
}

Eigen::VectorXd SoapModel::power_spectrum(const Eigen::VectorXcd &c,
                                          const std::vector<double> &wl) const {
  const int S = static_cast<int>(ntypes);
  const int nm = n_max;
  const int lm = l_max;

  // Power spectrum p^{αβ}_{n n' l}, flattened in the canonical order (shared
  // with descriptor_index_map via for_each_ps_slot).
  std::vector<double> vals;
  vals.reserve(descriptor_size());

  for_each_ps_slot(
      S, nm, lm, [&](Eigen::Index, int sa, int sb, int n, int n2, int l) {
        const Eigen::Index len = 2 * l + 1;
        const auto acc = c.segment(coeff_index(sa, n, l, -l), len)
                             .dot(c.segment(coeff_index(sb, n2, l, -l), len));
        vals.push_back(wl[l] * acc.real());
      });
  return Eigen::Map<Eigen::VectorXd>(vals.data(),
                                     static_cast<Eigen::Index>(vals.size()));
}

void SoapModel::position_gradient(const Atom &a, const Eigen::VectorXcd &c,
                                  const SoapRadialTable &tab, double K,
                                  const std::vector<double> &wl, double norm,
                                  DescriptorValue &out) const {
  const int S = static_cast<int>(ntypes);
  const int nm = n_max;
  const int lm = l_max;

  // Analytic position-gradient dD/dr
  // Only neighbour j's term in c depends on its bond vector d_j = r_j − r_i, so
  // dc[ch_j,n,l,m]/dd_j = K[ d(f_c·I_{nl})/dr · conj(Y_lm) · û   (radial)
  //                          + f_c·I_{nl} · conj(dY_lm/dd) ]     (angular).
  // Chain through p = w_l·Re(Σ_m conj(c_α)·c_β) and the L2-normalize Jacobian
  // (I − D Dᵀ)/‖p‖. grad_neigh[j] = dD/dr_j = dD/dd_j; grad_self = −Σ_j
  // dD/dd_j.
  const Eigen::Index Sd = out.values.size();
  out.grad_self = DescriptorGrad::Zero(Sd, 3);
  out.grad_neigh.assign(a.neighbors.size(), DescriptorGrad::Zero(Sd, 3));
  out.has_grad = true;

  const double pi = std::numbers::pi;
  std::array<Eigen::VectorXcd, 3> dc;

  for (std::size_t jj = 0; jj < a.neighbors.size(); ++jj) {
    const auto &nb = a.neighbors[jj];
    const double r = nb.dist.norm();
    if (r < 1e-12 || r >= rcut) {
      continue;
    }
    const int chj = static_cast<int>(nb.neighbor->type.index);
    if (chj < 0 || chj >= S) {
      continue;
    }
    const double fc = cutoff_fc(r, rcut);
    const double fcp = -0.5 * pi / rcut * std::sin(pi * r / rcut); // f_c'(r)

    // Radial projection I(n,l) and its r-derivative (spline + spline.prime()).
    Eigen::MatrixXd Jm(nm, lm + 1), dJm(nm, lm + 1);
    for (int l = 0; l <= lm; ++l) {
      for (int abasis = 0; abasis < nm; ++abasis) {
        Jm(abasis, l) = tab.J(abasis, l, r);
        dJm(abasis, l) = tab.dJdr(abasis, l, r);
      }
    }
    const Eigen::MatrixXd Inl = beta * Jm;   // I(n,l)
    const Eigen::MatrixXd Ipnl = beta * dJm; // dI/dr(n,l)

    // Spherical-coordinate frame at û = d_j/r and the gradients ∇θ = θ̂/r,
    // ∇φ = φ̂/(r·sinθ).
    const Eigen::Vector3d u = nb.dist / r;
    const double theta = std::acos(std::clamp(u.z(), -1.0, 1.0));
    const double phi = std::atan2(u.y(), u.x());
    const double st = std::sin(theta), ctn = std::cos(theta);
    const double cph = std::cos(phi), sph = std::sin(phi);
    const Eigen::Vector3d thetahat(ctn * cph, ctn * sph, -st);
    const Eigen::Vector3d phihat(-sph, cph, 0.0);
    // Guard the 1/sinθ coordinate singularity on the z-axis: clamp only the
    // derivative divisions, keeping gradients finite everywhere (exact away
    // from the poles, approximate only within ~1e-8 rad of the axis — far rarer
    // than a finite-difference kink).
    const double st_safe = std::max(st, 1e-8);

    for (auto &v : dc) {
      v = Eigen::VectorXcd::Zero(c.size());
    }
    const std::complex<double> exp_minus_i_phi =
        std::exp(std::complex<double>(0.0, -phi));

    const double inv_r = 1.0 / r;
    const double inv_r_st = 1.0 / (r * st_safe);
    const double cot_theta = ctn / st_safe;

    const auto theta_c = thetahat.cast<std::complex<double>>();
    const auto phi_c = phihat.cast<std::complex<double>>();

    for (int l = 0; l <= lm; ++l) {
      for (int m = -l; m <= l; ++m) {
        const auto Ylm = boost::math::spherical_harmonic(
            static_cast<unsigned>(l), m, theta, phi);

        const auto Ymp1 =
            (m < l) ? boost::math::spherical_harmonic(static_cast<unsigned>(l),
                                                      m + 1, theta, phi)
                    : std::complex<double>{0.0, 0.0};

        const double Cc = std::sqrt(static_cast<double>(l - m) * (l + m + 1));

        const auto dYdth = static_cast<double>(m) * cot_theta * Ylm +
                           Cc * exp_minus_i_phi * Ymp1;

        const auto dYdph =
            std::complex<double>{0.0, static_cast<double>(m)} * Ylm;

        const auto a1 = dYdth * inv_r;
        const auto a2 = dYdph * inv_r_st;

        const Eigen::Vector3cd dYcd = (a1 * theta_c + a2 * phi_c).conjugate();

        const auto Yc = std::conj(Ylm);

        for (int n = 0; n < nm; ++n) {
          const double In = Inl(n, l);
          const double Ipn = Ipnl(n, l);

          const double Rr = fc * In;
          const double dRr = fcp * In + fc * Ipn;

          const auto radial = K * dRr * Yc;
          const auto ang = K * Rr;

          const Eigen::Index id = coeff_index(chj, n, l, m);

          dc[0][id] = radial * u[0] + ang * dYcd[0];
          dc[1][id] = radial * u[1] + ang * dYcd[1];
          dc[2][id] = radial * u[2] + ang * dYcd[2];
        }
      }
    }

    // dp_k/dd = w_l·Re( conj(dc_α)·c_β + conj(c_α)·dc_β ), summed over m.
    // Eigen's a.dot(b) already conjugates a; dc is nonzero only on channel
    // ch_j, so the δ_{α,ch_j}/δ_{β,ch_j} selection falls out of the zero
    // segments.
    DescriptorGrad dpc(Sd, 3);
    Eigen::Index kk = 0;
    for (auto [sa, sb] : upper_triangle(S)) {
      for (int n = 0; n < nm; ++n) {
        const int n2start = (sa == sb) ? n : 0;
        for (int n2 = n2start; n2 < nm; ++n2) {
          for (int l = 0; l <= lm; ++l) {
            const Eigen::Index len = 2 * l + 1;
            for (int k = 0; k < 3; ++k) {
              const auto t1 =
                  dc[static_cast<std::size_t>(k)]
                      .segment(coeff_index(sa, n, l, -l), len)
                      .dot(c.segment(coeff_index(sb, n2, l, -l), len));
              const auto t2 = c.segment(coeff_index(sa, n, l, -l), len)
                                  .dot(dc[static_cast<std::size_t>(k)].segment(
                                      coeff_index(sb, n2, l, -l), len));
              dpc(kk, k) = wl[l] * (t1 + t2).real();
            }
            ++kk;
          }
        }
      }
    }

    // L2-normalize Jacobian: dD/dd = (dp − D (D·dp)) / ‖p‖.
    if (norm > 1e-12) {
      for (int k = 0; k < 3; ++k) {
        const Eigen::VectorXd col = dpc.col(k);
        dpc.col(k) = (col - out.values * out.values.dot(col)) / norm;
      }
    }
    out.grad_neigh[jj] = dpc;
    out.grad_self -= dpc;
  }
}

std::vector<std::optional<Eigen::Index>>
SoapModel::descriptor_index_map(const SpeciesRegistry &old_reg,
                                const SpeciesRegistry &new_reg) const {
  const int S_old = static_cast<int>(forcesmith::ntypes(old_reg));
  const int S_new = static_cast<int>(forcesmith::ntypes(new_reg));
  const int nm = n_max;
  const int lm = l_max;
  const auto old_of_new = old_slot_of_new(old_reg, new_reg);

  // Block bases per layout: a within-block (n,n2,l) offset is identical between
  // a retained pair and its old counterpart. Both registries are Z-sorted, so a
  // canonical new pair (sa≤sb) maps to a canonical old pair (a_old≤c_old) and a
  // diagonal pair stays diagonal — the block structure matches element-wise.
  const auto new_base = ps_pair_bases(S_new, nm, lm);
  const auto old_base = ps_pair_bases(S_old, nm, lm);

  std::vector<std::optional<Eigen::Index>> map(ps_size(S_new, nm, lm));
  for_each_ps_slot(
      S_new, nm, lm, [&](Eigen::Index idx, int sa, int sb, int, int, int) {
        const auto a_old = old_of_new[static_cast<std::size_t>(sa)];
        const auto c_old = old_of_new[static_cast<std::size_t>(sb)];
        if (!a_old || !c_old) {
          return; // a brand-new pair: leave nullopt (zero-filled)
        }
        const std::size_t po_new = pair_ordinal(
            static_cast<std::size_t>(sa), static_cast<std::size_t>(sb),
            static_cast<std::size_t>(S_new));
        const std::size_t po_old =
            pair_ordinal(*a_old, *c_old, static_cast<std::size_t>(S_old));
        const Eigen::Index off = idx - new_base[po_new];
        map[static_cast<std::size_t>(idx)] = old_base[po_old] + off;
      });
  return map;
}

} // namespace forcesmith
