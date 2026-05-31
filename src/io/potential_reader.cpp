#include "potfit/io/potential_reader.hpp"

#include "potfit/potentials/analytic_potential.hpp"
#include "potfit/potentials/spline.hpp"

#include <boost/leaf/error.hpp>
#include <nlohmann/json.hpp>

#include <functional>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace potfit::io {

using json = nlohmann::json;
namespace leaf = boost::leaf;

namespace {

using Maker = std::function<Potential(std::span<const double>, double, double)>;
struct Entry {
  int nparams;
  std::vector<std::string> param_names;
  Maker maker;
};
using Registry = std::unordered_map<std::string_view, Entry>;

template <typename... Names>
  requires(std::convertible_to<Names, std::string_view> && ...)
void add(Registry &reg, int nparams, std::vector<std::string> pnames,
         Maker maker, Names... names) {
  Entry e{nparams, std::move(pnames), std::move(maker)};
  (reg.try_emplace(std::string_view{names}, e), ...);
}

template <std::size_t N>
std::array<double, N> to_arr(std::span<const double> p) {
  std::array<double, N> a{};
  std::ranges::copy_n(p.begin(), static_cast<std::ptrdiff_t>(N), a.begin());
  return a;
}

const Registry &registry() {
  static const Registry reg = [] {
    Registry m;
    m.reserve(64);

    // clang-format off
    add(m,  2, {"epsilon","sigma"},                                                          [](auto p, auto lo, auto hi) { return Potential(LennardJones(p[0], p[1],                                      lo, hi)); }, "pair_lj",     "lj"         );
    add(m,  3, {"De","a","re"},                                                              [](auto p, auto lo, auto hi) { return Potential(Morse(p[0], p[1], p[2],                                       lo, hi)); }, "morse"                      );
    add(m,  3, {"A","rho","C"},                                                              [](auto p, auto lo, auto hi) { return Potential(Buckingham(p[0], p[1], p[2],                                  lo, hi)); }, "buckingham",  "buck"        );
    add(m,  5, {"A","B","C","D","E"},                                                        [](auto p, auto lo, auto hi) { return Potential(Born(p[0], p[1], p[2], p[3], p[4],                           lo, hi)); }, "born"                       );
    add(m,  2, {"A","n"},                                                                    [](auto p, auto lo, auto hi) { return Potential(PowerDecay(p[0], p[1],                                        lo, hi)); }, "power_decay", "power"       );
    add(m,  2, {"A","B"},                                                                    [](auto p, auto lo, auto hi) { return Potential(ExpDecay(p[0], p[1],                                          lo, hi)); }, "exp_decay",   "exp"         );
    add(m,  3, {"A","B","r0"},                                                               [](auto p, auto lo, auto hi) { return Potential(MexpDecay(p[0], p[1], p[2],                                   lo, hi)); }, "mexp_decay",  "mexp"        );
    add(m,  2, {"k","r0"},                                                                   [](auto p, auto lo, auto hi) { return Potential(Harmonic(p[0], p[1],                                          lo, hi)); }, "harmonic"                   );
    add(m,  4, {"E0","a","b","c"},                                                           [](auto p, auto lo, auto hi) { return Potential(Universal(p[0], p[1], p[2], p[3],                             lo, hi)); }, "universal"                  );
    add(m,  6, {"A","n","B","m","k","phi"},                                                  [](auto p, auto lo, auto hi) { return Potential(Eopp(p[0], p[1], p[2], p[3], p[4], p[5],                     lo, hi)); }, "eopp"                       );
    add(m,  6, {"A","B","C","m","k","phi"},                                                  [](auto p, auto lo, auto hi) { return Potential(EoppExp(p[0], p[1], p[2], p[3], p[4], p[5],                  lo, hi)); }, "eopp_exp",    "eopp_exp_"   );
    add(m,  7, {"A","n","B","m","k","phi","r0"},                                             [](auto p, auto lo, auto hi) { return Potential(Meopp(p[0], p[1], p[2], p[3], p[4], p[5], p[6],              lo, hi)); }, "meopp"                      );
    add(m,  5, {"A","n","m","r0","B"},                                                       [](auto p, auto lo, auto hi) { return Potential(GenLJ(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "gen_lj",      "genlj"       );
    add(m,  7, {"D1","a1","r1","D2","a2","r2","C"},                                          [](auto p, auto lo, auto hi) { return Potential(DoubleMorse(p[0], p[1], p[2], p[3], p[4], p[5], p[6],        lo, hi)); }, "double_morse","dbl_morse"   );
    add(m,  5, {"A","B","r1","C","r2"},                                                      [](auto p, auto lo, auto hi) { return Potential(DoubleExp(p[0], p[1], p[2], p[3], p[4],                      lo, hi)); }, "double_exp",  "dbl_exp"     );
    add(m,  6, {"A","B","C","r0","n","d"},                                                   [](auto p, auto lo, auto hi) { return Potential(Mishin(p[0], p[1], p[2], p[3], p[4], p[5],                   lo, hi)); }, "mishin"                     );
    add(m,  2, {"A","B"},                                                                    [](auto p, auto lo, auto hi) { return Potential(SqrtFunc(p[0], p[1],                                          lo, hi)); }, "sqrt"                       );
    add(m,  1, {"C"},                                                                        [](auto p, auto lo, auto hi) { return Potential(ConstFunc(p[0],                                               lo, hi)); }, "const"                      );
    add(m,  3, {"A","B","C"},                                                                [](auto p, auto lo, auto hi) { return Potential(Parabola(p[0], p[1], p[2],                                   lo, hi)); }, "parabola"                   );
    add(m,  5, {"a0","a1","a2","a3","a4"},                                                   [](auto p, auto lo, auto hi) { return Potential(Poly5(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "poly5"                      );
    add(m,  6, {"A","B","p","q","delta","rc"},                                               [](auto p, auto lo, auto hi) { return Potential(StiwWeb2(p[0], p[1], p[2], p[3], p[4], p[5],                 lo, hi)); }, "stiweb_2",    "sw2"         );
    add(m,  2, {"gamma","a"},                                                                [](auto p, auto lo, auto hi) { return Potential(StiwWeb3(p[0], p[1],                                          lo, hi)); }, "stiweb_3",    "sw3"         );
    add(m, 11, {"A","B","lambda","mu","beta","n","c","d","h","R","S"},                        [](auto p, auto lo, auto hi) { return Potential(TersoffPot(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], lo, hi)); }, "tersoff", "tersoff_pot");
    add(m,  2, {"chi","omega"},                                                              [](auto p, auto lo, auto hi) { return Potential(TersoffMix(p[0], p[1],                                        lo, hi)); }, "tersoff_mix"                );
    add(m, 16, {"A","B","lambda","mu","beta","n","c","d","h","R","S","c1","c2","c3","c4","c5"}, [](auto p, auto lo, auto hi) { return Potential(TersoffModPot(to_arr<16>(p),                               lo, hi)); }, "tersoff_mod", "tmod"        );
    add(m,  9, {"A","B","C","D","E","F","G","H","I"},                                        [](auto p, auto lo, auto hi) { return Potential(Kawamura(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], lo, hi)); }, "kawamura"                 );
    add(m, 12, {"A","B","C","D","E","F","G","H","I","J","K","L"},                            [](auto p, auto lo, auto hi) { return Potential(KawamuraMix(to_arr<12>(p),                                    lo, hi)); }, "kawamura_mix"               );
    add(m,  2, {"A","n"},                                                                    [](auto p, auto lo, auto hi) { return Potential(Softshell(p[0], p[1],                                         lo, hi)); }, "softshell",   "soft"        );
    add(m,  3, {"A","B","C"},                                                                [](auto p, auto lo, auto hi) { return Potential(ExpPlus(p[0], p[1], p[2],                                    lo, hi)); }, "exp_plus",    "expplus"     );
    add(m,  5, {"A","B","C","D","E"},                                                        [](auto p, auto lo, auto hi) { return Potential(Strmm(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "strmm"                      );
    // clang-format on

    return m;
  }();
  return reg;
}

} // anonymous namespace

leaf::result<std::vector<Potential>> parse_potential(std::string_view input) {
  auto fail = [](std::string msg) -> leaf::result<std::vector<Potential>> {
    return leaf::new_error(ParseError{std::move(msg), 0});
  };

  json j;
  try {
    j = json::parse(input);
  } catch (const json::parse_error &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }

  try {
    if (!j.contains("format"))
      return fail("missing 'format' key");
    const std::string fmt = j["format"].get<std::string>();

    if (!j.contains("potentials") || !j["potentials"].is_array())
      return fail("missing or invalid 'potentials' array");
    const auto &pots_arr = j["potentials"];

    std::vector<Potential> potentials;
    potentials.reserve(pots_arr.size());

    if (fmt == "tabulated") {
      for (const auto &p : pots_arr) {
        const double rmin = p.at("rmin").get<double>();
        const double rmax = p.at("rmax").get<double>();
        const auto &knots_arr = p.at("knots");
        if (!knots_arr.is_array() || knots_arr.size() < 2)
          return fail(
              "tabulated potential: 'knots' must be an array of >= 2 values");

        const int n = static_cast<int>(knots_arr.size());
        const double h = (rmax - rmin) / (n - 1);
        std::vector<double> x(static_cast<std::size_t>(n));
        std::vector<double> y(static_cast<std::size_t>(n));
        for (auto [k, knot] : std::views::enumerate(knots_arr)) {
          x[k] = rmin + static_cast<double>(k) * h;
          y[k] = knot.get<double>();
        }
        SplinePotential sp(std::move(x), std::move(y));
        // Optional "fixed": true freezes every knot, so this potential
        // contributes no free parameters to the optimizer (held constant).
        if (p.value("fixed", false))
          for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k)
            sp.set_fixed(k, true);
        potentials.emplace_back(std::move(sp));
      }
      return potentials;
    }

    if (fmt == "analytic") {
      const Registry &reg = registry();
      for (const auto &p : pots_arr) {
        if (!p.contains("type"))
          return fail("analytic potential missing 'type'");
        const std::string type_name = p["type"].get<std::string>();

        auto it = reg.find(type_name);
        if (it == reg.end())
          return fail("unknown analytic function: " + type_name);
        const auto &[nparams, param_names, make] = it->second;

        const double rmin = p.at("rmin").get<double>();
        const double rmax = p.at("rmax").get<double>();

        std::vector<double> params;
        params.reserve(static_cast<std::size_t>(nparams));
        for (const auto &name : param_names) {
          if (!p.contains(name))
            return fail("missing parameter '" + name + "' for type " +
                        type_name);
          params.push_back(p[name].get<double>());
        }

        potentials.emplace_back(make(params, rmin, rmax));
      }
      return potentials;
    }

    return fail("unsupported potential format '" + fmt + "'");

  } catch (const json::exception &e) {
    return leaf::new_error(ParseError{e.what(), 0});
  }
}

} // namespace potfit::io
