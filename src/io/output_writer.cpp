#include "potfit/io/output_writer.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace potfit::io {

using json = nlohmann::json;

// Sample each potential on a uniform grid and write JSON tabulated format.
// Works correctly for both SplinePotential and analytic types (LJ, Morse, etc.).
static constexpr int kDefaultKnots = 500;

void write_native(const std::filesystem::path& path,
                  const std::vector<Potential>& potentials,
                  int nknots)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());

    json j;
    j["format"] = "tabulated";
    j["potentials"] = json::array();

    for (const auto& p : potentials) {
        auto [lo, hi] = p.span();
        const double step = (hi - lo) / (nknots - 1);
        json pot;
        pot["rmin"] = lo;
        pot["rmax"] = hi;
        pot["knots"] = json::array();
        for (int k = 0; k < nknots; ++k)
            pot["knots"].push_back(p.eval(lo + k * step));
        j["potentials"].push_back(std::move(pot));
    }

    f << j.dump(2) << "\n";
}

void write_native(const std::filesystem::path& path,
                  const std::vector<Potential>& potentials)
{
    write_native(path, potentials, kDefaultKnots);
}

void write_lammps(const std::filesystem::path& path,
                  const std::vector<Potential>& potentials)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());

    for (std::size_t idx = 0; idx < potentials.size(); ++idx) {
        const auto& p = potentials[idx];
        auto [lo, hi] = p.span();
        const double step = (hi - lo) / (kDefaultKnots - 1);

        f << "# pair potential " << idx << "\n";
        f << "POT_" << idx << "\n";
        f << "N " << kDefaultKnots << " R " << lo << " " << hi << "\n\n";

        for (int k = 0; k < kDefaultKnots; ++k) {
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
        f << lo << " " << hi << " " << kDefaultKnots << "\n";
    }

    for (const auto& p : potentials) {
        auto [lo, hi] = p.span();
        const double step = (hi - lo) / (kDefaultKnots - 1);
        for (int k = 0; k < kDefaultKnots; ++k)
            f << p.eval(lo + k * step) << "\n";
    }
}

}  // namespace potfit::io
