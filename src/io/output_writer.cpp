#include "potfit/io/output_writer.hpp"

#include <Eigen/Core>
#include <fstream>
#include <stdexcept>

namespace potfit::io {

void write_native(const std::filesystem::path& path,
                  const std::vector<Potential>& potentials)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());

    const int n = static_cast<int>(potentials.size());
    f << "#F 3 " << n << "\n";
    f << "#E\n";

    // Distance-block: rmin rmax nknots per potential.
    for (const auto& p : potentials) {
        auto [lo, hi] = p.span();
        f << lo << " " << hi << " " << p.param_count() << "\n";
    }

    // Value-block: one y-value per line, per potential.
    for (const auto& p : potentials) {
        Eigen::VectorXd y(p.param_count());
        p.gather_params(y, 0);
        for (int k = 0; k < p.param_count(); ++k)
            f << y[k] << "\n";
    }
}

void write_lammps(const std::filesystem::path& path,
                  const std::vector<Potential>& potentials)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());

    constexpr int N = 1000;

    for (std::size_t idx = 0; idx < potentials.size(); ++idx) {
        const auto& p = potentials[idx];
        auto [lo, hi] = p.span();
        const double step = (hi - lo) / (N - 1);

        f << "# pair potential " << idx << "\n";
        f << "POT_" << idx << "\n";
        f << "N " << N << " R " << lo << " " << hi << "\n\n";

        for (int k = 0; k < N; ++k) {
            const double r = lo + k * step;
            f << (k + 1) << " " << r << " " << p.eval(r) << " " << p.deriv(r) << "\n";
        }
        f << "\n";
    }
}

void write_imd(const std::filesystem::path& path,
               const std::vector<Potential>& potentials)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());

    const int n = static_cast<int>(potentials.size());
    f << "#F 3 " << n << "\n";
    f << "#T IMD\n";
    f << "#E\n";

    for (const auto& p : potentials) {
        auto [lo, hi] = p.span();
        f << lo << " " << hi << " " << p.param_count() << "\n";
    }

    for (const auto& p : potentials) {
        Eigen::VectorXd y(p.param_count());
        p.gather_params(y, 0);
        for (int k = 0; k < p.param_count(); ++k)
            f << y[k] << "\n";
    }
}

}  // namespace potfit::io
