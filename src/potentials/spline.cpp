#include "potfit/potentials/spline.hpp"

#include <algorithm>
#include <cassert>

namespace potfit {

SplinePotential::SplinePotential(std::vector<double> x, std::vector<double> y) {
    assert(x.size() == y.size() && x.size() >= 2);
    const int n = static_cast<int>(x.size());
    x_ = Eigen::Map<const Eigen::VectorXd>(x.data(), n);
    y_ = Eigen::Map<const Eigen::VectorXd>(y.data(), n);
    d2y_ = Eigen::VectorXd::Zero(n);
    compute_d2y();
}

// Natural cubic spline: d2y_[0] = d2y_[n-1] = 0.
// Thomas algorithm solves the interior tridiagonal system.
void SplinePotential::compute_d2y() {
    const int n = static_cast<int>(x_.size());
    d2y_.setZero();

    if (n == 2)
        return;  // linear: second derivatives remain zero

    Eigen::VectorXd h(n - 1);
    for (int i = 0; i < n - 1; ++i)
        h[i] = x_[i + 1] - x_[i];

    Eigen::VectorXd diag(n - 2), rhs(n - 2), upper(n - 3);
    for (int i = 0; i < n - 2; ++i) {
        diag[i] = 2.0 * (h[i] + h[i + 1]);
        rhs[i]  = 6.0 * ((y_[i + 2] - y_[i + 1]) / h[i + 1] -
                          (y_[i + 1] - y_[i])     / h[i]);
    }
    for (int i = 0; i < n - 3; ++i)
        upper[i] = h[i + 1];

    for (int i = 1; i < n - 2; ++i) {
        const double factor = h[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i]  -= factor * rhs[i - 1];
    }

    d2y_[n - 2] = rhs[n - 3] / diag[n - 3];
    for (int i = n - 3; i >= 1; --i)
        d2y_[i] = (rhs[i - 1] - upper[i - 1] * d2y_[i + 1]) / diag[i - 1];
    // d2y_[0] and d2y_[n-1] remain 0 (natural BCs)
}

double SplinePotential::eval(double r) const {
    const int n = static_cast<int>(x_.size());
    if (r <= x_[0])   return y_[0];
    if (r >= x_[n-1]) return y_[n-1];

    const int i = static_cast<int>(
        std::upper_bound(x_.data(), x_.data() + n, r) - x_.data()) - 1;
    const double h = x_[i+1] - x_[i];
    const double a = (x_[i+1] - r) / h;
    const double b = (r - x_[i])   / h;
    return a * y_[i] + b * y_[i+1] +
           ((a*a*a - a) * d2y_[i] + (b*b*b - b) * d2y_[i+1]) * (h*h) / 6.0;
}

double SplinePotential::deriv(double r) const {
    const int n = static_cast<int>(x_.size());
    if (r <= x_[0])   return (y_[1]   - y_[0])   / (x_[1]   - x_[0]);
    if (r >= x_[n-1]) return (y_[n-1] - y_[n-2]) / (x_[n-1] - x_[n-2]);

    const int i = static_cast<int>(
        std::upper_bound(x_.data(), x_.data() + n, r) - x_.data()) - 1;
    const double h = x_[i+1] - x_[i];
    const double a = (x_[i+1] - r) / h;
    const double b = (r - x_[i])   / h;
    return (y_[i+1] - y_[i]) / h +
           (-(3.0*a*a - 1.0) * d2y_[i] + (3.0*b*b - 1.0) * d2y_[i+1]) * h / 6.0;
}

std::pair<double, double> SplinePotential::span() const {
    return {x_[0], x_[x_.size() - 1]};
}

void SplinePotential::gather_params(Eigen::VectorXd& dst, int offset) const {
    dst.segment(offset, y_.size()) = y_;
}

void SplinePotential::scatter_params(const Eigen::VectorXd& src, int offset) {
    y_ = src.segment(offset, y_.size());
    compute_d2y();
}

}  // namespace potfit
