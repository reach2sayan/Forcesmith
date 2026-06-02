#include "potfit/cli/evaluate_report.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace leaf = boost::leaf;

namespace potfit::cli {

leaf::result<void> write_evaluate_report(potfit::PotFit &session,
                                         const CliOptions &o, int &exit_code) {
  const std::string ev_path = *o.evaluate;
  const double ew = o.energy_weight;
  const double sw = o.stress_weight;

  BOOST_LEAF_AUTO(configs, session.configurations());

  std::ofstream out(ev_path);
  if (!out) {
    std::cerr << "error: cannot open evaluate output: " << ev_path << "\n";
    exit_code = 1;
    return {};
  }
  out << std::setprecision(17);

  auto stress6 = [](const potfit::SymTens &s, std::ostream &oo) {
    oo << s(0, 0) << ", " << s(1, 1) << ", " << s(2, 2) << ", " << s(0, 1)
       << ", " << s(1, 2) << ", " << s(0, 2);
  };

  double total_sumsq = 0.0;
  out << "{\n  \"energy_weight\": " << ew << ",\n  \"stress_weight\": " << sw
      << ",\n  \"nconf\": " << configs.size() << ",\n  \"configs\": [\n";

  for (std::size_t i = 0; i < configs.size(); ++i) {
    const potfit::Configuration &cfg = configs[i];
    BOOST_LEAF_AUTO(r, session.evaluate(i));

    double csq = 0.0;
    out << "    {\n      \"index\": " << i << ",\n      \"name\": \""
        << cfg.name << "\""
        << ",\n      \"natoms\": " << cfg.atoms.size()
        << ",\n      \"calc_energy\": " << r.energy
        << ",\n      \"ref_energy\": " << cfg.ref.energy
        << ",\n      \"calc_stress\": [";
    stress6(r.stress, out);
    out << "],\n      \"ref_stress\": [";
    stress6(cfg.ref.stress, out);
    out << "],\n      \"atoms\": [\n";
    for (std::size_t a = 0; a < cfg.atoms.size(); ++a) {
      const auto &cf = r.forces[a];
      const auto &rf = cfg.atoms[a].ref.force;
      for (int k = 0; k < 3; ++k) {
        const double d = cf[k] - rf[k];
        csq += d * d;
      }
      out << "        {\"calc_force\": [" << cf[0] << ", " << cf[1] << ", "
          << cf[2] << "], \"ref_force\": [" << rf[0] << ", " << rf[1] << ", "
          << rf[2] << "]}" << (a + 1 < cfg.atoms.size() ? "," : "") << "\n";
    }
    const double de = ew * (r.energy - cfg.ref.energy);
    csq += de * de;
    if (sw > 0.0) {
      const potfit::SymTens ds = r.stress - cfg.ref.stress;
      const double comps[6] = {ds(0, 0), ds(1, 1), ds(2, 2),
                               ds(0, 1), ds(1, 2), ds(0, 2)};
      for (double c : comps) {
        csq += (sw * c) * (sw * c);
      }
    }
    csq += r.limit * r.limit;
    total_sumsq += csq;

    out << "      ],\n      \"limit\": " << r.limit
        << ",\n      \"sumsq\": " << csq << "\n    }"
        << (i + 1 < configs.size() ? "," : "") << "\n";
  }

  out << "  ],\n  \"total_sumsq\": " << total_sumsq << "\n}\n";
  std::cout << "evaluated " << configs.size()
            << " configurations, total error sum = " << std::setprecision(10)
            << total_sumsq << "\nwrote evaluation report to " << ev_path
            << "\n";
  return {};
}

} // namespace potfit::cli
