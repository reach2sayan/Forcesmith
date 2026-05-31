#include "potfit/io/output_writer.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <ranges>
#include <stdexcept>

namespace potfit::io {

using json = nlohmann::json;

// Sample one potential on a uniform grid over its span → {rmin,rmax,knots}.
static json sample_one(const Potential &p, int nknots) {
  auto [lo, hi] = p.span();
  const double step = (hi - lo) / (nknots - 1);
  json pot;
  pot["rmin"] = lo;
  pot["rmax"] = hi;
  pot["knots"] = json::array();
  for (int k : std::views::iota(0, nknots))
    pot["knots"].push_back(p.eval(lo + k * step));
  return pot;
}

// Build a {format:"tabulated", potentials:[...]} section from a range of
// Potentials (works for the flat vector and for PotentialPair/PotentialArray).
template <typename Range>
static json sample_section(const Range &pots, int nknots) {
  json sec;
  sec["format"] = "tabulated";
  sec["potentials"] = json::array();
  for (const auto &p : pots)
    sec["potentials"].push_back(sample_one(p, nknots));
  return sec;
}

void write_native(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials, int nknots) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("cannot open " + path.string());

  json j = sample_section(potentials, nknots);
  f << j.dump(2) << "\n";
}

void write_native_eam(const std::filesystem::path &path,
                      const EAMForceCalculator &eam, int nknots) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("cannot open " + path.string());

  // ADP/angular/tersoff/stiweb can reuse sample_section the same way when their
  // output is added; only EAM is supported here.
  json j;
  j["model"] = "eam";
  j["ntypes"] = eam.density.size();
  j["pair"] = sample_section(eam.pair, nknots);
  j["density"] = sample_section(eam.density, nknots);
  j["embedding"] = sample_section(eam.embedding, nknots);

  f << j.dump(2) << "\n";
}

void write_lammps(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("cannot open " + path.string());

  for (auto [idx, p] : std::views::enumerate(potentials)) {
    auto [lo, hi] = p.span();
    const double step = (hi - lo) / (kDefaultKnots - 1);

    f << "# pair potential " << idx << "\n";
    f << "POT_" << idx << "\n";
    f << "N " << kDefaultKnots << " R " << lo << " " << hi << "\n\n";

    for (int k : std::views::iota(0, kDefaultKnots)) {
      const double r = lo + k * step;
      f << (k + 1) << " " << r << " " << p.eval(r) << " " << p.deriv(r) << "\n";
    }
    f << "\n";
  }
}

void write_imd(const std::filesystem::path &path,
               const std::vector<Potential> &potentials) {
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("cannot open " + path.string());
  }

  const int n = static_cast<int>(potentials.size());
  f << "#F 3 " << n << "\n";
  f << "#T IMD\n";
  f << "#E\n";

  for (const auto &p : potentials) {
    auto [lo, hi] = p.span();
    f << lo << " " << hi << " " << kDefaultKnots << "\n";
  }

  for (const auto &p : potentials) {
    auto [lo, hi] = p.span();
    const double step = (hi - lo) / (kDefaultKnots - 1);
    for (int k : std::views::iota(0, kDefaultKnots))
      f << p.eval(lo + k * step) << "\n";
  }
}

} // namespace potfit::io
