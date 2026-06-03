#include "potfit/potentials/soap.hpp"

#include <Eigen/Eigenvalues>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <execution>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <vector>

namespace potfit {

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

} // namespace

// Optional single-threaded precompute of the radial basis. get_descriptor
// auto-computes β on demand if this was never called, so callers never have to.
void SoapModel::init_radial_basis() { beta = radial_beta(n_max, rcut); }

std::size_t SoapModel::descriptor_size() const {
  const std::size_t S = ntypes;
  const std::size_t nm = static_cast<std::size_t>(n_max);
  const std::size_t same = S * (nm * (nm + 1) / 2);      // α=β: n≤n'
  const std::size_t cross = (S * (S - 1) / 2) * nm * nm; // α<β: all n,n'
  return (static_cast<std::size_t>(l_max) + 1) * (same + cross);
}

DescriptorValue SoapModel::get_descriptor(const Atom &a) const {

  // Cosine cutoff f_c(r) = ½(1 + cos(π r / rcut)) for r < rcut, else 0.
  auto cutoff = [](double r, double rc) {
    return 0.5 * (1.0 + std::cos(std::numbers::pi * r / rc));
  };

  const int S = static_cast<int>(ntypes);
  const int nm = n_max;
  const int lm = l_max;
  const int nlm = 2 * lm + 1;

  // Flat complex coefficient store c[channel][n][l][m], m the fastest axis
  // (offset by +lm) so the 2l+1 coefficients of a fixed (ch,n,l) are contiguous
  // and the power-spectrum m-sum is a single Eigen dot product.
  auto idx = [&](int ch, int n, int l, int m) -> Eigen::Index {
    return ((static_cast<Eigen::Index>(ch) * nm + n) * (lm + 1) + l) * nlm +
           (m + lm);
  };
  // Unnormalized polynomial radial basis φ_a(r) = (rcut − r)^{a+2}, a =
  // 0..n_max−1.
  auto phi = [](int aparam, double r, double rc) {
    return std::pow(rc - r, aparam + 2);
  };

  // Modified spherical Bessel function of the first kind,
  // i_l(x) = sqrt(π/2x) · I_{l+1/2}(x), with the correct x→0 limits
  // (i_0(0)=1, i_l(0)=0 for l>0).
  auto sph_bessel_i = [](int l, double x) {
    if (x < 1e-12) {
      return l == 0 ? 1.0 : 0.0;
    }
    return std::sqrt(std::numbers::pi / (2.0 * x)) *
           boost::math::cyl_bessel_i(l + 0.5, x);
  };

  Eigen::VectorXcd c = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(S) *
                                              nm * (lm + 1) * nlm);

  const double K = 4.0 * std::numbers::pi /
                   std::pow(2.0 * std::numbers::pi * sigma * sigma, 1.5);
  const double inv_2s2 = 1.0 / (2.0 * sigma * sigma);

  for (const auto &nb : a.neighbors) {
    const double r = nb.dist.norm();
    if (r < 1e-12 || r >= rcut) {
      continue;
    }
    const int ch = static_cast<int>(nb.neighbor->type.index);
    if (ch < 0 || ch >= S) {
      continue;
    }
    const double fc = cutoff(r, rcut);

    // Radial projection I(n,l) = Σ_a β(n,a) ∫ r'² φ_a(r') e^{-(r'²+r²)/2σ²}
    //                                          i_l(r r'/σ²) dr'.
    Eigen::MatrixXd J(nm, lm + 1); // basis × l
    auto total =
        static_cast<std::size_t>(nm) * static_cast<std::size_t>(lm + 1);
    std::vector<std::size_t> flattened_indices(total);
    std::ranges::iota(flattened_indices, std::size_t{0});

    std::for_each(
        std::execution::par, flattened_indices.begin(), flattened_indices.end(),
        [&](std::size_t k) {
          const int abasis = static_cast<int>(k % nm);
          const int l = static_cast<int>(k / nm);
          auto integrand = [&](double rp) {
            const double x = r * rp / (sigma * sigma);
            return rp * rp * phi(abasis, rp, rcut) *
                   std::exp(-(rp * rp + r * r) * inv_2s2) * sph_bessel_i(l, x);
          };

          J(abasis, l) =
              boost::math::quadrature::gauss_kronrod<double, 31>::integrate(
                  integrand, 0.0, rcut, 5, 1e-9);
        });
    if (beta.rows() != nm || beta.cols() != nm) {
      beta = radial_beta(nm, rcut);
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
        c.segment(idx(ch, n, l, -l), 2 * l + 1) += (K * fc * I(n, l)) * Yl;
      }
    }
  }

  // Power spectrum p^{αβ}_{n n' l}, flattened in a fixed order.
  DescriptorValue out;
  out.has_grad = false;
  std::vector<double> vals;
  vals.reserve(descriptor_size());

  std::vector<double> wl(lm + 1);
  std::ranges::transform(std::views::iota(0, lm + 1), wl.begin(), [](int l) {
    return std::sqrt(8.0 * std::numbers::pi * std::numbers::pi /
                     (2.0 * l + 1.0));
  });

  for (auto [sa, sb] : upper_triangle(S)) {
    for (int n = 0; n < nm; ++n) {
      const int n2start = (sa == sb) ? n : 0;

      for (int n2 = n2start; n2 < nm; ++n2) {
        for (int l = 0; l <= lm; ++l) {
          const Eigen::Index len = 2 * l + 1;

          const auto acc = c.segment(idx(sa, n, l, -l), len)
                               .dot(c.segment(idx(sb, n2, l, -l), len));

          vals.push_back(wl[l] * acc.real());
        }
      }
    }
  }
  out.values = Eigen::Map<Eigen::VectorXd>(
      vals.data(), static_cast<Eigen::Index>(vals.size()));
  const double norm = out.values.norm();
  if (norm > 1e-12) {
    out.values /= norm;
  }
  return out;
}

} // namespace potfit
