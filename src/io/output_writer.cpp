#include "forcesmith/io/output_writer.hpp"

#include "forcesmith/core/families.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/io/file.hpp"
#include "forcesmith/io/schema.hpp"

#include "forcesmith/io/config_reader.hpp" // ParseError

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <ranges>

namespace forcesmith::io {

namespace leaf = boost::leaf;

using json = nlohmann::json;

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

template <std::ranges::input_range Range>
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
  BOOST_LEAF_AUTO(f, open_out(path));
  BOOST_LEAF_AUTO(j, sample_section(potentials, nknots));
  f << j.dump(2) << "\n";
  return {};
}

template <CTabulated T>
leaf::result<void> write_tabulated(const std::filesystem::path &path,
                                   const T &calc, int nknots) {
  BOOST_LEAF_AUTO(f, open_out(path));
  json j;
  j["model"] = family_name<T>;
  j["ntypes"] = calc.ntypes;

  leaf::result<void> status{};
  for_each_table(calc, [&](const auto &table, std::string_view key) {
    if (!status) {
      return;
    }
    auto section = sample_section(table, nknots);
    if (!section) {
      status = leaf::result<void>{section.error()};
      return;
    }
    if (key.empty()) {
      j = std::move(*section);
    } else {
      j[std::string(key)] = std::move(*section);
    }
  });
  BOOST_LEAF_CHECK(std::move(status));

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native(const std::filesystem::path &path,
                                const EAMForceCalculator &eam, int nknots) {
  return write_tabulated(path, eam, nknots);
}

leaf::result<void> write_native(const std::filesystem::path &path,
                                const ADPForceCalculator &adp, int nknots) {
  return write_tabulated(path, adp, nknots);
}

leaf::result<void> write_native(const std::filesystem::path &path,
                                const AngularForceCalculator &ang, int nknots) {
  return write_tabulated(path, ang, nknots);
}

leaf::result<void> write_native(const std::filesystem::path &path,
                                const TersoffForceCalculator &ters) {
  BOOST_LEAF_AUTO(f, open_out(path));
  json j;
  j["model"] = "tersoff";
  j["ntypes"] = ters.ntypes;
  j["potentials"] = json(ters.params | std::views::all);

  f << j.dump(2) << "\n";
  return {};
}

leaf::result<void> write_native(const std::filesystem::path &path,
                                const StiwebForceCalculator &sw) {
  BOOST_LEAF_AUTO(f, open_out(path));
  json j;
  j["model"] = "stiweb";
  j["ntypes"] = sw.ntypes;
  j["potentials"] = json(sw.params | std::views::all);
  j["lambda"] = json(sw.lambda);

  f << j.dump(2) << "\n";
  return {};
}

static json head_to_json(const EnergyHead &h) { return h.to_json(); }

static json heads_to_json(const TypeArray<EnergyHead> &heads) {
  json arr = json::array();
  for (const auto &h : heads) {
    arr.push_back(head_to_json(h));
  }
  return arr;
}

template <class M>
concept CStandardizable = requires(const M &m) {
  m.standardize_features;
  m.mean_;
  m.inv_std_;
};

static void add_standardization(json &j, const CStandardizable auto &ml) {
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

leaf::result<void> write_native(const std::filesystem::path &path,
                                const ACSF &ml) {
  BOOST_LEAF_AUTO(f, open_out(path));

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

leaf::result<void> write_native(const std::filesystem::path &path,
                                const SoapModel &soap) {
  BOOST_LEAF_AUTO(f, open_out(path));

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

leaf::result<void> write_native(const std::filesystem::path &path,
                                const LMBTR &ml) {
  BOOST_LEAF_AUTO(f, open_out(path));

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
  BOOST_LEAF_AUTO(f, open_out(path));

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
  BOOST_LEAF_AUTO(f, open_out(path));

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
    for (int k : std::views::iota(0, kDefaultKnots)) {
      f << p.eval(lo + k * step) << "\n";
    }
  }
  return {};
}

} // namespace forcesmith::io
