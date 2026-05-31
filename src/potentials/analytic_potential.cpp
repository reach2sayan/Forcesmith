#include "potfit/potentials/analytic_potential.hpp"

#include <cmath>
#include <numbers>

namespace potfit {

namespace {

// Tersoff / SW smooth cosine cutoff and its derivative.
double fc(double r, double R, double S) noexcept {
  if (r <= R) {
    return 1.0;
  }
  if (r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return 0.5 + 0.5 * std::cos(x);
}

double dfc(double r, double R, double S) noexcept {
  if (r <= R || r >= S) {
    return 0.0;
  }
  const double x = std::numbers::pi * (r - R) / (S - R);
  return -0.5 * std::numbers::pi / (S - R) * std::sin(x);
}

} // namespace

// ── Lennard-Jones ───────────────────────────────────────────────────────────
double LennardJones::eval_impl(double r) const {
  const auto [ep, sig] = params;
  const double sr6 = std::pow(sig / r, 6);
  return 4.0 * ep * (sr6 * sr6 - sr6);
}
double LennardJones::deriv_impl(double r) const {
  const auto [ep, sig] = params;
  const double sr6 = std::pow(sig / r, 6);
  return 4.0 * ep * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;
}

// ── Morse ───────────────────────────────────────────────────────────────────
double Morse::eval_impl(double r) const {
  const auto [De, a, re] = params;
  const double e = std::exp(-a * (r - re));
  return De * (1.0 - e) * (1.0 - e) - De;
}
double Morse::deriv_impl(double r) const {
  const auto [De, a, re] = params;
  const double e = std::exp(-a * (r - re));
  return 2.0 * De * a * e * (1.0 - e);
}

// ── Buckingham ──────────────────────────────────────────────────────────────
double Buckingham::eval_impl(double r) const {
  const auto [A, rho, C] = params;
  const double x = (rho * rho) / (r * r);
  return A * std::exp(-r / rho) - C * x * x * x; // C·(ρ²/r²)³ = C·ρ⁶/r⁶
}
double Buckingham::deriv_impl(double r) const {
  const auto [A, rho, C] = params;
  const double rho6 = std::pow(rho, 6);
  return -(A / rho) * std::exp(-r / rho) + 6.0 * C * rho6 / std::pow(r, 7);
}

// ── Born ────────────────────────────────────────────────────────────────────
double Born::eval_impl(double r) const {
  const auto [A, B, C, D, E] = params;
  const double r2 = r * r, r6 = r2 * r2 * r2, r8 = r6 * r2;
  return A * std::exp((C - r) / B) - D / r6 + E / r8;
}
double Born::deriv_impl(double r) const {
  const auto [A, B, C, D, E] = params;
  return -(A / B) * std::exp((C - r) / B) + 6.0 * D / std::pow(r, 7) -
         8.0 * E / std::pow(r, 9);
}

// ── PowerDecay ──────────────────────────────────────────────────────────────
double PowerDecay::eval_impl(double r) const {
  const auto [A, n] = params;
  return A / std::pow(r, n);
}
double PowerDecay::deriv_impl(double r) const {
  const auto [A, n] = params;
  return -A * n / std::pow(r, n + 1.0);
}

// ── ExpDecay ────────────────────────────────────────────────────────────────
double ExpDecay::eval_impl(double r) const {
  const auto [A, B] = params;
  return A * std::exp(-B * r);
}
double ExpDecay::deriv_impl(double r) const {
  const auto [A, B] = params;
  return -A * B * std::exp(-B * r);
}

// ── MexpDecay ───────────────────────────────────────────────────────────────
double MexpDecay::eval_impl(double r) const {
  const auto [A, B, r0] = params;
  return A * std::exp(-B * (r - r0));
}
double MexpDecay::deriv_impl(double r) const {
  const auto [A, B, r0] = params;
  return -A * B * std::exp(-B * (r - r0));
}

// ── Harmonic ────────────────────────────────────────────────────────────────
double Harmonic::eval_impl(double r) const {
  const auto [k, r0] = params;
  return k * (r - r0) * (r - r0);
}
double Harmonic::deriv_impl(double r) const {
  const auto [k, r0] = params;
  return 2.0 * k * (r - r0);
}

// ── Universal ───────────────────────────────────────────────────────────────
double Universal::eval_impl(double r) const {
  const auto [E0, a, b, c] = params;
  return E0 * (b / (b - a) * std::pow(r, a) - a / (b - a) * std::pow(r, b)) +
         c * r;
}
double Universal::deriv_impl(double r) const {
  const auto [E0, a, b, c] = params;
  return E0 * (a * b / (b - a)) * (std::pow(r, a - 1.0) - std::pow(r, b - 1.0)) +
         c;
}

// ── Eopp ────────────────────────────────────────────────────────────────────
double Eopp::eval_impl(double r) const {
  const auto [A, n, B, m, k, phi] = params;
  return A / std::pow(r, n) + (B / std::pow(r, m)) * std::cos(k * r + phi);
}
double Eopp::deriv_impl(double r) const {
  const auto [A, n, B, m, k, phi] = params;
  const double c = std::cos(k * r + phi), s = std::sin(k * r + phi);
  return -A * n / std::pow(r, n + 1.0) - B * m / std::pow(r, m + 1.0) * c -
         B * k / std::pow(r, m) * s;
}

// ── EoppExp ─────────────────────────────────────────────────────────────────
double EoppExp::eval_impl(double r) const {
  const auto [A, B, C, m, k, phi] = params;
  return A * std::exp(-B * r) + (C / std::pow(r, m)) * std::cos(k * r + phi);
}
double EoppExp::deriv_impl(double r) const {
  const auto [A, B, C, m, k, phi] = params;
  const double cc = std::cos(k * r + phi), s = std::sin(k * r + phi);
  return -A * B * std::exp(-B * r) - C * m / std::pow(r, m + 1.0) * cc -
         C * k / std::pow(r, m) * s;
}

// ── Meopp ───────────────────────────────────────────────────────────────────
double Meopp::eval_impl(double r) const {
  const auto [A, n, B, m, k, phi, r0] = params;
  return A / std::pow(r - r0, n) + (B / std::pow(r, m)) * std::cos(k * r + phi);
}
double Meopp::deriv_impl(double r) const {
  const auto [A, n, B, m, k, phi, r0] = params;
  const double c = std::cos(k * r + phi), s = std::sin(k * r + phi);
  return -A * n / std::pow(r - r0, n + 1.0) - B * m / std::pow(r, m + 1.0) * c -
         B * k / std::pow(r, m) * s;
}

// ── GenLJ ───────────────────────────────────────────────────────────────────
double GenLJ::eval_impl(double r) const {
  const auto [A, n, m, r0, B] = params;
  const double x = r / r0;
  return A / (m - n) * (m * std::pow(x, -n) - n * std::pow(x, -m)) + B;
}
double GenLJ::deriv_impl(double r) const {
  const auto [A, n, m, r0, B] = params;
  const double x = r / r0;
  // dV/dr = A/(m−n)·(1/r0)·(−mn·x^{−n−1} + nm·x^{−m−1})
  return A / (m - n) / r0 * m * n *
         (std::pow(x, -m - 1.0) - std::pow(x, -n - 1.0));
}

// ── DoubleMorse ─────────────────────────────────────────────────────────────
double DoubleMorse::eval_impl(double r) const {
  const auto [D1, a1, r1, D2, a2, r2, C] = params;
  const double e1 = std::exp(-a1 * (r - r1));
  const double e2 = std::exp(-a2 * (r - r2));
  return D1 * ((1.0 - e1) * (1.0 - e1) - 1.0) +
         D2 * ((1.0 - e2) * (1.0 - e2) - 1.0) + C;
}
double DoubleMorse::deriv_impl(double r) const {
  const auto [D1, a1, r1, D2, a2, r2, C] = params;
  const double e1 = std::exp(-a1 * (r - r1));
  const double e2 = std::exp(-a2 * (r - r2));
  return 2.0 * D1 * a1 * e1 * (1.0 - e1) + 2.0 * D2 * a2 * e2 * (1.0 - e2);
}

// ── DoubleExp ───────────────────────────────────────────────────────────────
double DoubleExp::eval_impl(double r) const {
  const auto [A, B, r1, C, r2] = params;
  const double dr = r - r1;
  return A * std::exp(-B * dr * dr) + std::exp(-C * (r - r2));
}
double DoubleExp::deriv_impl(double r) const {
  const auto [A, B, r1, C, r2] = params;
  const double dr = r - r1;
  return -2.0 * A * B * dr * std::exp(-B * dr * dr) -
         C * std::exp(-C * (r - r2));
}

// ── Mishin ──────────────────────────────────────────────────────────────────
double Mishin::eval_impl(double r) const {
  const auto [A, B, C, r0, n, d] = params;
  const double z = r - r0;
  const double e = std::exp(-d * z);
  return A * std::pow(z, n) * e * (1.0 + B * e) + C;
}
double Mishin::deriv_impl(double r) const {
  const auto [A, B, C, r0, n, d] = params;
  const double z = r - r0;
  const double e = std::exp(-d * z);
  const double zn1 = std::pow(z, n - 1.0);
  // d/dr[A z^n e]      = A z^{n-1} e (n − d z)
  // d/dr[A B z^n e^2]  = A B z^{n-1} e^2 (n − 2 d z)
  return A * zn1 * e * (n - d * z) + A * B * zn1 * e * e * (n - 2.0 * d * z);
}

// ── SqrtFunc ────────────────────────────────────────────────────────────────
double SqrtFunc::eval_impl(double r) const {
  const auto [A, B] = params;
  return A * std::sqrt(r / B);
}
double SqrtFunc::deriv_impl(double r) const {
  const auto [A, B] = params;
  return A / (2.0 * B * std::sqrt(r / B));
}

// ── ConstFunc ───────────────────────────────────────────────────────────────
double ConstFunc::eval_impl(double) const { return params[0]; }
double ConstFunc::deriv_impl(double) const { return 0.0; }

// ── Parabola ────────────────────────────────────────────────────────────────
double Parabola::eval_impl(double r) const {
  const auto [A, B, C] = params;
  return A * r * r + B * r + C;
}
double Parabola::deriv_impl(double r) const {
  const auto [A, B, C] = params;
  return 2.0 * A * r + B;
}

// ── Poly5 ───────────────────────────────────────────────────────────────────
double Poly5::eval_impl(double r) const {
  const auto [a0, a1, a2, a3, a4] = params;
  const double s = r - 1.0, s2 = s * s;
  return a0 + 0.5 * a1 * s2 + a2 * s * s2 + a3 * s2 * s2 + a4 * s2 * s2 * s;
}
double Poly5::deriv_impl(double r) const {
  const auto [a0, a1, a2, a3, a4] = params;
  const double s = r - 1.0, s2 = s * s;
  return a1 * s + 3.0 * a2 * s2 + 4.0 * a3 * s2 * s + 5.0 * a4 * s2 * s2;
}

// ── StiwWeb2 ────────────────────────────────────────────────────────────────
double StiwWeb2::eval_impl(double r) const {
  const auto [A, B, p, q, delta, rc] = params;
  if (r >= rc)
    return 0.0; // exp pole at r = rc; SW pair vanishes beyond
  const double poly = A * std::pow(r, -p) - B * std::pow(r, -q);
  return poly * std::exp(delta / (r - rc));
}
double StiwWeb2::deriv_impl(double r) const {
  const auto [A, B, p, q, delta, rc] = params;
  if (r >= rc)
    return 0.0;
  const double d = r - rc;
  const double g = std::exp(delta / d);
  const double poly = A * std::pow(r, -p) - B * std::pow(r, -q);
  const double dpoly =
      -A * p * std::pow(r, -p - 1.0) + B * q * std::pow(r, -q - 1.0);
  return dpoly * g + poly * g * (-delta / (d * d));
}

// ── StiwWeb3 ────────────────────────────────────────────────────────────────
double StiwWeb3::eval_impl(double r) const {
  const auto [gamma, a] = params;
  if (r >= a)
    return 0.0;
  return std::exp(gamma / (r - a));
}
double StiwWeb3::deriv_impl(double r) const {
  const auto [gamma, a] = params;
  if (r >= a)
    return 0.0;
  const double d = r - a;
  const double h = std::exp(gamma / d);
  if (h == 0.0)
    return 0.0;
  return h * (-gamma / (d * d));
}

// ── TersoffPot ──────────────────────────────────────────────────────────────
double TersoffPot::eval_impl(double r) const {
  const double A = params[0], B = params[1], lam = params[2], mu = params[3];
  const double R = params[9], S = params[10];
  return fc(r, R, S) * (A * std::exp(-lam * r) - B * std::exp(-mu * r));
}
double TersoffPot::deriv_impl(double r) const {
  const double A = params[0], B = params[1], lam = params[2], mu = params[3];
  const double R = params[9], S = params[10];
  const double f = fc(r, R, S);
  const double df = dfc(r, R, S);
  const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
  const double dpair =
      -lam * A * std::exp(-lam * r) + mu * B * std::exp(-mu * r);
  return df * pair + f * dpair;
}

// ── TersoffMix ──────────────────────────────────────────────────────────────
double TersoffMix::eval_impl(double r) const {
  const auto [chi, omega] = params;
  return chi * std::exp(-omega * r);
}
double TersoffMix::deriv_impl(double r) const {
  const auto [chi, omega] = params;
  return -chi * omega * std::exp(-omega * r);
}

// ── TersoffModPot ───────────────────────────────────────────────────────────
double TersoffModPot::eval_impl(double r) const {
  const double A = params[0], B = params[1];
  const double lam = params[2], mu = params[3];
  const double R = params[9], S = params[10];
  const double c1 = params[11], c2 = params[12];
  const double c3 = params[13], c4 = params[14], c5 = params[15];
  const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
  const double corr = 1.0 + c1 * r + c2 * r * r + c3 * r * r * r +
                      c4 * r * r * r * r + c5 * r * r * r * r * r;
  return fc(r, R, S) * pair * corr;
}
double TersoffModPot::deriv_impl(double r) const {
  const double A = params[0], B = params[1];
  const double lam = params[2], mu = params[3];
  const double R = params[9], S = params[10];
  const double c1 = params[11], c2 = params[12];
  const double c3 = params[13], c4 = params[14], c5 = params[15];
  const double f = fc(r, R, S);
  const double df = dfc(r, R, S);
  const double pair = A * std::exp(-lam * r) - B * std::exp(-mu * r);
  const double dpair =
      -lam * A * std::exp(-lam * r) + mu * B * std::exp(-mu * r);
  const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
  const double corr = 1.0 + c1 * r + c2 * r2 + c3 * r3 + c4 * r4 + c5 * r5;
  const double dcorr =
      c1 + 2.0 * c2 * r + 3.0 * c3 * r2 + 4.0 * c4 * r3 + 5.0 * c5 * r4;
  return (df * pair + f * dpair) * corr + f * pair * dcorr;
}

// ── Kawamura ────────────────────────────────────────────────────────────────
double Kawamura::eval_impl(double r) const {
  const auto [p0, p1, p2, p3, p4, p5, p6, p7, p8] = params;
  const double s = p5 + p6, t = p3 + p4;
  const double r6 = std::pow(r, 6);
  return p0 * p1 / r + p2 * s * std::exp((t - r) / s) - p7 * p8 / r6;
}
double Kawamura::deriv_impl(double r) const {
  const auto [p0, p1, p2, p3, p4, p5, p6, p7, p8] = params;
  const double s = p5 + p6, t = p3 + p4;
  return -p0 * p1 / (r * r) - p2 * std::exp((t - r) / s) +
         6.0 * p7 * p8 / std::pow(r, 7);
}

// ── KawamuraMix ─────────────────────────────────────────────────────────────
double KawamuraMix::eval_impl(double r) const {
  const double p0 = params[0], p1 = params[1], p2 = params[2], p3 = params[3];
  const double p4 = params[4], p5 = params[5], p6 = params[6], p7 = params[7],
               p8 = params[8];
  const double p9 = params[9], p10 = params[10], p11 = params[11];
  const double s = p5 + p6, t = p3 + p4, r6 = std::pow(r, 6), w = r - p11;
  return p0 * p1 / r + p2 * s * std::exp((t - r) / s) - p7 * p8 / r6 +
         p2 * p9 * (std::exp(-2.0 * p10 * w) - 2.0 * std::exp(-p10 * w));
}
double KawamuraMix::deriv_impl(double r) const {
  const double p0 = params[0], p1 = params[1], p2 = params[2], p3 = params[3];
  const double p4 = params[4], p5 = params[5], p6 = params[6], p7 = params[7],
               p8 = params[8];
  const double p9 = params[9], p10 = params[10], p11 = params[11];
  const double s = p5 + p6, t = p3 + p4, w = r - p11;
  return -p0 * p1 / (r * r) - p2 * std::exp((t - r) / s) +
         6.0 * p7 * p8 / std::pow(r, 7) +
         2.0 * p2 * p9 * p10 * (std::exp(-p10 * w) - std::exp(-2.0 * p10 * w));
}

// ── Softshell ───────────────────────────────────────────────────────────────
double Softshell::eval_impl(double r) const {
  const auto [A, n] = params;
  return std::pow(A / r, n);
}
double Softshell::deriv_impl(double r) const {
  const auto [A, n] = params;
  return -n * std::pow(A / r, n) / r; // d/dr (A/r)^n = -(n/r)(A/r)^n
}

// ── ExpPlus ─────────────────────────────────────────────────────────────────
double ExpPlus::eval_impl(double r) const {
  const auto [A, B, C] = params;
  return A * std::exp(-B * r) + C;
}
double ExpPlus::deriv_impl(double r) const {
  const auto [A, B, C] = params;
  return -A * B * std::exp(-B * r);
}

// ── Strmm ───────────────────────────────────────────────────────────────────
double Strmm::eval_impl(double r) const {
  const auto [A, B, C, D, r0] = params;
  const double s = r - r0;
  return 2.0 * A * std::exp(-B / 2.0 * s) - C * (1.0 + D * s) * std::exp(-D * s);
}
double Strmm::deriv_impl(double r) const {
  const auto [A, B, C, D, r0] = params;
  const double s = r - r0;
  return -A * B * std::exp(-B / 2.0 * s) + C * D * D * s * std::exp(-D * s);
}

} // namespace potfit
