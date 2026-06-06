#include "forcesmith/io/output_writer.hpp"

#include "forcesmith/io/config_reader.hpp" // ParseError

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <fstream>
#include <ranges>

namespace forcesmith::io {

namespace leaf = boost::leaf;

// Open `path` for writing into a fresh std::ofstream named `var`, returning a
// leaf error from the enclosing function if the stream fails to open. Used by
// every writer below (each returns leaf::result<void>).
#define OPEN_FILE_WITH_HANDLE(var, path)                                       \
  std::ofstream var(path);                                                     \
  if (!(var)) {                                                                \
    return leaf::new_error(ParseError{"cannot open " + (path).string(), 0});   \
  }

using json = nlohmann::json;

// Sample one potential on a uniform grid over its span → {rmin,rmax,knots}.
static leaf::result<json> sample_one(const RadialPotential &p, int nknots) {
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
      return leaf::new_error(ParseError{
          "non-finite potential value (" + std::to_string(v) + ") at r=" +
              std::to_string(r) + "; cannot tabulate potential for output",
          0});
    }
    pot["knots"].push_back(v);
  }
  return pot;
}

template <typename Range>
static leaf::result<json> sample_section(const Range &pots, int nknots) {
  json sec;
  sec["format"] = "tabulated";
  sec["potentials"] = json::array();
  for (const auto &p : pots) {
    BOOST_LEAF_AUTO(one, sample_one(p, nknots));
    sec["potentials"].push_back(std::move(one));
  }
  return sec;
}

leaf::result<void> write_native(const std::filesystem::path &path,
                                const std::vector<RadialPotential> &potentials,
                                int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);
  BOOST_LEAF_AUTO(j, sample_section(potentials, nknots));
  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native_eam(const std::filesystem::path &path,
                                    const EAMForceCalculator &eam, int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);

  json j;
  j["model"] = "eam";
  j["ntypes"] = eam.density.size();
  BOOST_LEAF_ASSIGN(j["pair"], sample_section(eam.pair, nknots));
  BOOST_LEAF_ASSIGN(j["density"], sample_section(eam.density, nknots));
  BOOST_LEAF_ASSIGN(j["embedding"], sample_section(eam.embedding, nknots));

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native_adp(const std::filesystem::path &path,
                                    const ADPForceCalculator &adp, int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);
  json j;
  j["model"] = "adp";
  j["ntypes"] = adp.density.size();
  BOOST_LEAF_ASSIGN(j["pair"], sample_section(adp.pair, nknots));
  BOOST_LEAF_ASSIGN(j["density"], sample_section(adp.density, nknots));
  BOOST_LEAF_ASSIGN(j["embedding"], sample_section(adp.embedding, nknots));
  BOOST_LEAF_ASSIGN(j["dipole"], sample_section(adp.dipole, nknots));
  BOOST_LEAF_ASSIGN(j["quadrupole"], sample_section(adp.quadrupole, nknots));

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native_angular(const std::filesystem::path &path,
                                        const AngularForceCalculator &ang,
                                        int nknots) {
  OPEN_FILE_WITH_HANDLE(f, path);
  json j;
  j["model"] = "angular";
  j["ntypes"] = ang.angular.size();
  BOOST_LEAF_ASSIGN(j["pair"], sample_section(ang.pair, nknots));
  BOOST_LEAF_ASSIGN(j["radial"], sample_section(ang.radial, nknots));
  // angular g(cosθ) is sampled over its own [-1,1] span by sample_one.
  BOOST_LEAF_ASSIGN(j["angular"], sample_section(ang.angular, nknots));

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native_tersoff(const std::filesystem::path &path,
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
  return {};
}

leaf::result<void> write_native_stiweb(const std::filesystem::path &path,
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
  return {};
}

// Serialize one value-erased head. EnergyHead::to_json() routes to the
// build_json_from_head ADL CPO of the held concrete head, so each head type
// (including external ones) owns its own JSON shape.
static json head_to_json(const EnergyHead &h) { return h.to_json(); }

static json heads_to_json(const TypeArray<EnergyHead> &heads) {
  json arr = json::array();
  for (const auto &h : heads) {
    arr.push_back(head_to_json(h));
  }
  return arr;
}

// Per-feature descriptor standardization (one {mean, inv_std} per element
// type), emitted as a sibling of "heads" so a reloaded model predicts
// identically. Lives on MLBase, not the head, so it is written separately from
// heads_to_json. Omitted entirely when disabled or never computed (e.g. a model
// that was never fit).
template <class Model>
static void add_standardization(json &j, const Model &ml) {
  if (!ml.standardize_features || ml.mean_.size() == 0) {
    return;
  }
  json arr = json::array();
  for (std::size_t t = 0; t < ml.mean_.size(); ++t) {
    const Eigen::VectorXd &mu = ml.mean_[t];
    const Eigen::VectorXd &iv = ml.inv_std_[t];
    arr.push_back(
        {{"mean", std::vector<double>(mu.data(), mu.data() + mu.size())},
         {"inv_std", std::vector<double>(iv.data(), iv.data() + iv.size())}});
  }
  j["standardization"] = std::move(arr);
}

leaf::result<void> write_native_acsf(const std::filesystem::path &path,
                                     const ACSF &ml) {
  OPEN_FILE_WITH_HANDLE(f, path);

  json j;
  j["model"] = "ml";
  j["ntypes"] = ml.ntypes;

  json desc;
  desc["type"] = "acsf";
  desc["rcut"] = ml.rcut;
  if (ml.g1 > 0) {
    desc["g1"] = ml.g1;
  }
  desc["g2"] = json::array();
  for (const auto &g : ml.radial) {
    desc["g2"].push_back({{"eta", g.eta}, {"rs", g.rs}});
  }
  if (!ml.g3.empty()) {
    desc["g3"] = json::array();
    for (const auto &g : ml.g3) {
      desc["g3"].push_back({{"kappa", g.kappa}});
    }
  }
  if (!ml.g4.empty()) {
    desc["g4"] = json::array();
    for (const auto &g : ml.g4) {
      desc["g4"].push_back(
          {{"eta", g.eta}, {"zeta", g.zeta}, {"lambda", g.lambda}});
    }
  }
  if (!ml.g5.empty()) {
    desc["g5"] = json::array();
    for (const auto &g : ml.g5) {
      desc["g5"].push_back(
          {{"eta", g.eta}, {"zeta", g.zeta}, {"lambda", g.lambda}});
    }
  }
  j["descriptor"] = std::move(desc);
  j["heads"] = heads_to_json(ml.heads);
  add_standardization(j, ml);

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native_soap(const std::filesystem::path &path,
                                     const SoapModel &soap) {
  OPEN_FILE_WITH_HANDLE(f, path);

  json j;
  j["model"] = "ml";
  j["ntypes"] = soap.ntypes;

  json desc;
  desc["type"] = "soap";
  desc["n_max"] = soap.n_max;
  desc["l_max"] = soap.l_max;
  desc["rcut"] = soap.rcut;
  desc["sigma"] = soap.sigma;
  j["descriptor"] = std::move(desc);
  j["heads"] = heads_to_json(soap.heads);
  add_standardization(j, soap);

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native_lmbtr(const std::filesystem::path &path,
                                      const LMBTR &ml) {
  OPEN_FILE_WITH_HANDLE(f, path);

  json j;
  j["model"] = "ml";
  j["ntypes"] = ml.ntypes;

  auto grid = [](const LMBTR::Grid &g) {
    return json{{"min", g.min}, {"max", g.max}, {"n", g.n}, {"sigma", g.sigma}};
  };

  json desc;
  desc["type"] = "lmbtr";
  desc["rcut"] = ml.rcut;
  desc["weight_scale"] = ml.weight_scale;
  desc["normalize"] = ml.normalize_l2;
  if (ml.k2) {
    desc["k2"] = grid(*ml.k2);
  }
  if (ml.k3) {
    desc["k3"] = grid(*ml.k3);
  }
  j["descriptor"] = std::move(desc);
  j["heads"] = heads_to_json(ml.heads);
  add_standardization(j, ml);

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void>
write_lammps(const std::filesystem::path &path,
             const std::vector<RadialPotential> &potentials) {
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
  return {};
}

leaf::result<void> write_imd(const std::filesystem::path &path,
                             const std::vector<RadialPotential> &potentials) {
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
  return {};
}

} // namespace forcesmith::io
