#include "forcesmith/cli/evaluate_report.hpp"

#include "forcesmith/core/json.hpp"
#include "forcesmith/core/voigt.hpp"
#include "forcesmith/io/config_reader.hpp" // ParseError

#include <ranges>
#include <boost/leaf/error.hpp>
#include <boost/leaf/handle_errors.hpp>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace leaf = boost::leaf;
using json = nlohmann::json;

namespace forcesmith::cli {

leaf::result<void> write_evaluate_report(forcesmith::Forcesmith &session,
                                         const CliOptions &o) {
  const std::string ev_path = *o.evaluate;
  const double ew = o.energy_weight;
  const double sw = o.stress_weight;

  BOOST_LEAF_AUTO(configs, session.configurations());
  BOOST_LEAF_AUTO(results, session.evaluate_all());

  std::ofstream out(ev_path);
  if (!out) {
    return leaf::new_error(forcesmith::io::ParseError{
        "cannot open evaluate output: " + ev_path, 0});
  }

  const auto stress6 = [](const forcesmith::SymTens &s) {
    json a = json::array();
    for (const auto &[i, j] : kVoigt6) {
      a.push_back(s(i, j));
    }
    return a;
  };

  double total_sumsq = 0.0;
  json report;
  report["energy_weight"] = ew;
  report["stress_weight"] = sw;
  report["nconf"] = configs.size();
  report["configs"] = json::array();

  for (const auto &[i, pair_] : std::views::zip(configs, results) |
                                    std::views::enumerate) {
    const auto &[cfg, r] = pair_;
    double csq = 0.0;
    json atoms = json::array();
    for (const auto &[cf, atom] : std::views::zip(r.forces, cfg.atoms)) {
      const Vec3 &rf = atom.ref.force;
      csq += (cf - rf).squaredNorm();
      atoms.push_back(json{{"calc_force", Vec3(cf)}, {"ref_force", Vec3(rf)}});
    }
    const double de = ew * (r.energy - cfg.ref.energy);
    csq += de * de;
    if (sw > 0.0) {
      const forcesmith::SymTens ds = r.stress - cfg.ref.stress;
      for (const auto &[vi, vj] : kVoigt6) {
        csq += (sw * ds(vi, vj)) * (sw * ds(vi, vj));
      }
    }
    csq += r.limit * r.limit;
    total_sumsq += csq;

    report["configs"].push_back(json{{"index", static_cast<std::size_t>(i)},
                                     {"name", cfg.name},
                                     {"natoms", cfg.atoms.size()},
                                     {"calc_energy", r.energy},
                                     {"ref_energy", cfg.ref.energy},
                                     {"calc_stress", stress6(r.stress)},
                                     {"ref_stress", stress6(cfg.ref.stress)},
                                     {"atoms", std::move(atoms)},
                                     {"limit", r.limit},
                                     {"sumsq", csq}});
  }
  report["total_sumsq"] = total_sumsq;

  out << report.dump(2) << "\n";
  std::cout << "evaluated " << configs.size()
            << " configurations, total error sum = " << std::setprecision(10)
            << total_sumsq << "\nwrote evaluation report to " << ev_path
            << "\n";
  return {};
}

} // namespace forcesmith::cli
