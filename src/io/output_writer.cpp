#include "potfit/io/output_writer.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <ranges>
#include <stdexcept>

namespace potfit::io {

// Open `path` for writing into a fresh std::ofstream named `var`, throwing if
// the stream fails to open. Used by every writer below.
#define OPEN_FILE_WITH_HANDLE(var, path)                                       \
  std::ofstream var(path);                                                     \
  if (!(var)) {                                                                \
    throw std::runtime_error("cannot open " + (path).string());               \
  }

using json = nlohmann::json;

// Sample one potential on a uniform grid over its span → {rmin,rmax,knots}.
static json sample_one(const Potential &p, int nknots) {
  auto [lo, hi] = p.span();
  const double step = (hi - lo) / (nknots - 1);
  json pot;
  pot["rmin"] = lo;
  pot["rmax"] = hi;
  pot["knots"] = json::array();
  for (int k : std::views::iota(0, nknots)) {
    const double r = lo + k * step;
    const double v = p.eval(r);
    if (!std::isfinite(v)) {
      throw std::runtime_error("non-finite potential value (" +
                               std::to_string(v) + ") at r=" +
                               std::to_string(r) +
                               "; cannot tabulate potential for output");
    }
    pot["knots"].push_back(v);
  }
  return pot;
}

template <typename Range>
static json sample_section(const Range &pots, int nknots) {
  json sec;
  sec["format"] = "tabulated";
  sec["potentials"] = json::array();
  for (const auto &p : pots) {
    sec["potentials"].push_back(sample_one(p, nknots));
  }
  return sec;
}

void write_native(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials, int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);
  json j = sample_section(potentials, nknots);
  f << j.dump(2) << "\n";
}

void write_native_eam(const std::filesystem::path &path,
                      const EAMForceCalculator &eam, int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);

  json j;
  j["model"] = "eam";
  j["ntypes"] = eam.density.size();
  j["pair"] = sample_section(eam.pair, nknots);
  j["density"] = sample_section(eam.density, nknots);
  j["embedding"] = sample_section(eam.embedding, nknots);

  f << j.dump(2) << "\n";
}

void write_native_adp(const std::filesystem::path &path,
                      const ADPForceCalculator &adp, int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);
  json j;
  j["model"] = "adp";
  j["ntypes"] = adp.density.size();
  j["pair"] = sample_section(adp.pair, nknots);
  j["density"] = sample_section(adp.density, nknots);
  j["embedding"] = sample_section(adp.embedding, nknots);
  j["dipole"] = sample_section(adp.dipole, nknots);
  j["quadrupole"] = sample_section(adp.quadrupole, nknots);

  f << j.dump(2) << "\n";
}

void write_native_angular(const std::filesystem::path &path,
                          const AngularForceCalculator &ang, int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);
  json j;
  j["model"] = "angular";
  j["ntypes"] = ang.angular.size();
  j["pair"] = sample_section(ang.pair, nknots);
  j["radial"] = sample_section(ang.radial, nknots);
  // angular g(cosθ) is sampled over its own [-1,1] span by sample_one.
  j["angular"] = sample_section(ang.angular, nknots);

  f << j.dump(2) << "\n";
}

void write_native_tersoff(const std::filesystem::path &path,
                          const TersoffForceCalculator &ters) {
  OPEN_FILE_WITH_HANDLE(f, path);
  json j;
  j["model"] = "tersoff";
  j["ntypes"] = ters.ntypes;
  j["potentials"] = json::array();
  // params iterate in SymmetricMatrix slot order — the same order the reader
  // fills from the input "potentials" array (force_model_reader.cpp).
  for (const TersoffParams &p : ters.params) {
    json o;
    o["A"] = p.A.value;
    o["B"] = p.B.value;
    o["lambda"] = p.lambda.value;
    o["mu"] = p.mu.value;
    o["beta"] = p.beta.value;
    o["n"] = p.n.value;
    o["c"] = p.c.value;
    o["d"] = p.d.value;
    o["h"] = p.h.value;
    o["R"] = p.R.value;
    o["S"] = p.S.value;
    o["omega"] = p.omega.value;
    j["potentials"].push_back(std::move(o));
  }

  f << j.dump(2) << "\n";
}

void write_native_stiweb(const std::filesystem::path &path,
                         const StiwebForceCalculator &sw) {
  OPEN_FILE_WITH_HANDLE(f, path);

  json j;
  j["model"] = "stiweb";
  j["ntypes"] = sw.ntypes;
  j["potentials"] = json::array();
  for (const SWParams &p : sw.params) {
    json o;
    o["A"] = p.A.value;
    o["B"] = p.B.value;
    o["p"] = p.p.value;
    o["q"] = p.q.value;
    o["delta"] = p.delta.value;
    o["a1"] = p.a1.value;
    o["gamma"] = p.gamma.value;
    o["a2"] = p.a2.value;
    j["potentials"].push_back(std::move(o));
  }
  // Per-triplet λ: flat ntypes·paircol array in stored order.
  j["lambda"] = json::array();
  for (const Param &l : sw.lambda)
    j["lambda"].push_back(l.value);

  f << j.dump(2) << "\n";
}

void write_lammps(const std::filesystem::path &path,
                  const std::vector<Potential> &potentials) {
  OPEN_FILE_WITH_HANDLE(f, path);

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
  OPEN_FILE_WITH_HANDLE(f, path);

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
