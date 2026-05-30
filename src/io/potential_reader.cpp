#include "potfit/io/potential_reader.hpp"

#include "potfit/potentials/analytic_potential.hpp"
#include "potfit/potentials/spline.hpp"

#include <boost/leaf/error.hpp>
#include <boost/parser/parser.hpp>

#include <functional>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>

namespace bp = boost::parser;
namespace leaf = boost::leaf;

namespace potfit::io {

namespace {

// ── Line-format structs for boost::parser tuples ─────────────────────────────

struct FLine {
  int fmt;
  int count;
}; // #F <fmt> <count>
struct DistLine {
  double rmin, rmax;
  int nknots;
}; // <rmin> <rmax> <nknots>
struct ParamVals {
  double val, lo, hi;
}; // <val> <lo> <hi>
struct RngLine {
  double rmin, rmax;
};

inline auto const f_p = bp::lit("#F") >> bp::int_ >> bp::int_;
inline auto const dist_p = bp::double_ >> bp::double_ >> bp::int_;
inline auto const param_p = bp::double_ >> bp::double_ >> bp::double_;
inline auto const type_rng_p = bp::double_ >> bp::double_;

// Strip the leading "param <name> " token and return the rest for numeric
// parsing.
std::string_view strip_param_prefix(std::string_view sv) {
  if (sv.starts_with("param"))
    sv.remove_prefix(5);
  while (!sv.empty() && (sv[0] == ' ' || sv[0] == '\t'))
    sv.remove_prefix(1);
  while (!sv.empty() && sv[0] != ' ' && sv[0] != '\t')
    sv.remove_prefix(1);
  return sv;
}

// ── Alexandrescu-style object factory ────────────────────────────────────────
//
// Maker    — type-erased factory: (params, lo, hi) → Potential.
// Entry    — owns both the expected parameter count and the maker.
// Registry — map from every accepted name/alias to its Entry.
//
// Registration is done once at program start via the static registry()
// function. Adding a new function type requires a single add() call; aliases
// are free.

using Maker = std::function<Potential(std::span<const double>, double, double)>;
using Entry = std::pair<int, Maker>; // {nparams, maker}
using Registry = std::unordered_map<std::string_view, Entry>;

// Insert one Entry under each of the supplied name aliases.
// Uses a fold over a variadic pack so the caller never touches the map
// directly.
template <typename... Names>
  requires(std::convertible_to<Names, std::string_view> && ...)
void add(Registry &reg, int nparams, Maker maker, Names... names) {
  Entry e{nparams, std::move(maker)};
  (reg.try_emplace(std::string_view{names}, e), ...);
}

// Helper: build an std::array<double,N> from the first N elements of a span.
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
    // ── pair ─────────────────────────────────────────────────────────────────
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(LennardJones(p[0], p[1],                                      lo, hi)); }, "pair_lj",     "lj"         );
    add(m,  3, [](auto p, auto lo, auto hi) { return Potential(Morse(p[0], p[1], p[2],                                       lo, hi)); }, "morse"                      );
    // ── analytic ─────────────────────────────────────────────────────────────
    add(m,  3, [](auto p, auto lo, auto hi) { return Potential(Buckingham(p[0], p[1], p[2],                                  lo, hi)); }, "buckingham",  "buck"        );
    add(m,  5, [](auto p, auto lo, auto hi) { return Potential(Born(p[0], p[1], p[2], p[3], p[4],                           lo, hi)); }, "born"                       );
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(PowerDecay(p[0], p[1],                                        lo, hi)); }, "power_decay", "power"       );
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(ExpDecay(p[0], p[1],                                          lo, hi)); }, "exp_decay",   "exp"         );
    add(m,  3, [](auto p, auto lo, auto hi) { return Potential(MexpDecay(p[0], p[1], p[2],                                   lo, hi)); }, "mexp_decay",  "mexp"        );
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(Harmonic(p[0], p[1],                                          lo, hi)); }, "harmonic"                   );
    add(m,  4, [](auto p, auto lo, auto hi) { return Potential(Universal(p[0], p[1], p[2], p[3],                             lo, hi)); }, "universal"                  );
    add(m,  6, [](auto p, auto lo, auto hi) { return Potential(Eopp(p[0], p[1], p[2], p[3], p[4], p[5],                     lo, hi)); }, "eopp"                       );
    add(m,  6, [](auto p, auto lo, auto hi) { return Potential(EoppExp(p[0], p[1], p[2], p[3], p[4], p[5],                  lo, hi)); }, "eopp_exp",    "eopp_exp_"   );
    add(m,  7, [](auto p, auto lo, auto hi) { return Potential(Meopp(p[0], p[1], p[2], p[3], p[4], p[5], p[6],              lo, hi)); }, "meopp"                      );
    add(m,  5, [](auto p, auto lo, auto hi) { return Potential(GenLJ(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "gen_lj",      "genlj"       );
    add(m,  7, [](auto p, auto lo, auto hi) { return Potential(DoubleMorse(p[0], p[1], p[2], p[3], p[4], p[5], p[6],        lo, hi)); }, "double_morse","dbl_morse"   );
    add(m,  5, [](auto p, auto lo, auto hi) { return Potential(DoubleExp(p[0], p[1], p[2], p[3], p[4],                      lo, hi)); }, "double_exp",  "dbl_exp"     );
    add(m,  6, [](auto p, auto lo, auto hi) { return Potential(Mishin(p[0], p[1], p[2], p[3], p[4], p[5],                   lo, hi)); }, "mishin"                     );
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(SqrtFunc(p[0], p[1],                                          lo, hi)); }, "sqrt"                       );
    add(m,  1, [](auto p, auto lo, auto hi) { return Potential(ConstFunc(p[0],                                               lo, hi)); }, "const"                      );
    add(m,  3, [](auto p, auto lo, auto hi) { return Potential(Parabola(p[0], p[1], p[2],                                   lo, hi)); }, "parabola"                   );
    add(m,  5, [](auto p, auto lo, auto hi) { return Potential(Poly5(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "poly5"                      );
    // ── Stillinger–Weber ─────────────────────────────────────────────────────
    add(m,  6, [](auto p, auto lo, auto hi) { return Potential(StiwWeb2(p[0], p[1], p[2], p[3], p[4], p[5],                 lo, hi)); }, "stiweb_2",    "sw2"         );
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(StiwWeb3(p[0], p[1],                                          lo, hi)); }, "stiweb_3",    "sw3"         );
    // ── Tersoff ──────────────────────────────────────────────────────────────
    add(m, 11, [](auto p, auto lo, auto hi) { return Potential(TersoffPot(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], lo, hi)); }, "tersoff", "tersoff_pot");
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(TersoffMix(p[0], p[1],                                        lo, hi)); }, "tersoff_mix"                );
    add(m, 16, [](auto p, auto lo, auto hi) { return Potential(TersoffModPot(to_arr<16>(p),                                  lo, hi)); }, "tersoff_mod", "tmod"        );
    // ── Kawamura ─────────────────────────────────────────────────────────────
    add(m,  9, [](auto p, auto lo, auto hi) { return Potential(Kawamura(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], lo, hi)); }, "kawamura"                 );
    add(m, 12, [](auto p, auto lo, auto hi) { return Potential(KawamuraMix(to_arr<12>(p),                                    lo, hi)); }, "kawamura_mix"               );
    // ── misc ─────────────────────────────────────────────────────────────────
    add(m,  2, [](auto p, auto lo, auto hi) { return Potential(Softshell(p[0], p[1],                                         lo, hi)); }, "softshell",   "soft"        );
    add(m,  3, [](auto p, auto lo, auto hi) { return Potential(ExpPlus(p[0], p[1], p[2],                                    lo, hi)); }, "exp_plus",    "expplus"     );
    add(m,  5, [](auto p, auto lo, auto hi) { return Potential(Strmm(p[0], p[1], p[2], p[3], p[4],                          lo, hi)); }, "strmm"                      );
    // clang-format on

    return m;
  }();
  return reg;
}

} // anonymous namespace

// ── Public parser
// ─────────────────────────────────────────────────────────────

leaf::result<std::vector<Potential>> parse_potential(std::string_view input) {
  std::size_t line_num = 0;
  auto fail = [&](std::string msg) -> leaf::result<std::vector<Potential>> {
    return leaf::new_error(ParseError{std::move(msg), line_num});
  };

  // ── Split into non-empty trimmed lines ────────────────────────────────────
  std::vector<std::string_view> lines;
  for (auto const lr : input | std::views::split('\n')) {
    std::string_view sv{lr.begin(), lr.end()};
    if (!sv.empty() && sv.back() == '\r')
      sv.remove_suffix(1);
    lines.push_back(sv);
  }

  std::size_t idx = 0;
  auto next_line = [&]() -> std::string_view {
    while (idx < lines.size()) {
      ++line_num;
      std::string_view sv = lines[idx++];
      if (sv.find_first_not_of(" \t") != std::string_view::npos)
        return sv;
    }
    return {};
  };

  // ── Parse header (#F … #E) ────────────────────────────────────────────────
  int fmt = -1, num_funcs = 0;
  for (bool done = false; !done;) {
    auto line = next_line();
    if (line.empty())
      return fail("unexpected end of file before #E header terminator");
    if (line.starts_with("#F")) {
      FLine fl{};
      if (!bp::parse(line, f_p, bp::ws, fl))
        return fail("malformed #F directive");
      fmt = fl.fmt;
      num_funcs = fl.count;
    } else if (line == "#E") {
      if (fmt < 0)
        return fail("#E reached without a preceding #F directive");
      done = true;
    } else if (line[0] != '#') {
      return fail("non-directive line before #E header terminator");
    }
    // #T, #C, #I, #G — skip silently
  }

  if (fmt != 3 && fmt != 0)
    return fail("unsupported potential format " + std::to_string(fmt));

  std::vector<Potential> potentials;
  potentials.reserve(static_cast<std::size_t>(num_funcs));

  // ── Format 3: tabulated (equal-spaced) ───────────────────────────────────
  if (fmt == 3) {
    std::vector<DistLine> dists;
    dists.reserve(static_cast<std::size_t>(num_funcs));
    for (int f = 0; f < num_funcs; ++f) {
      auto line = next_line();
      if (line.empty())
        return fail("unexpected end of file in distance block");
      DistLine dl{};
      if (!bp::parse(line, dist_p, bp::ws, dl))
        return fail("malformed distance line (expected: rmin rmax nknots)");
      if (dl.nknots < 2)
        return fail("num_knots must be >= 2");
      dists.push_back(dl);
    }
    for (int f = 0; f < num_funcs; ++f) {
      const auto &d = dists[static_cast<std::size_t>(f)];
      const double h = (d.rmax - d.rmin) / (d.nknots - 1);
      std::vector<double> x(static_cast<std::size_t>(d.nknots));
      std::vector<double> y(static_cast<std::size_t>(d.nknots));
      for (int k = 0; k < d.nknots; ++k) {
        x[static_cast<std::size_t>(k)] = d.rmin + k * h;
        auto line = next_line();
        if (line.empty())
          return fail("unexpected end of file: expected " +
                      std::to_string(d.nknots - k) + " more knot value(s)");
        double v{};
        if (!bp::parse(line, bp::double_, bp::ws, v))
          return fail("malformed knot value line");
        y[static_cast<std::size_t>(k)] = v;
      }
      potentials.emplace_back(SplinePotential(std::move(x), std::move(y)));
    }
    return potentials;
  }

  // ── Format 0: analytic — single registry lookup per function ─────────────
  const Registry &reg = registry();

  for (int f = 0; f < num_funcs; ++f) {
    // 1. type line
    auto type_line = next_line();
    if (type_line.empty())
      return fail("unexpected end of file: expected 'type <name>' line");
    if (!type_line.starts_with("type "))
      return fail("expected 'type <name>', got: " + std::string(type_line));
    std::string_view name = type_line.substr(5);
    if (auto pos = name.find_first_not_of(' '); pos != std::string_view::npos)
      name = name.substr(pos);

    // 2. registry lookup — gives us nparams + maker in one shot
    auto it = reg.find(name);
    if (it == reg.end())
      return fail("unknown analytic function: " + std::string(name));
    const auto &[nparams, make] = it->second;

    // 3. rmin / rmax
    auto rng_line = next_line();
    if (rng_line.empty())
      return fail("unexpected end of file: expected rmin rmax line");
    RngLine rng{};
    if (!bp::parse(rng_line, type_rng_p, bp::ws, rng))
      return fail("malformed rmin/rmax line");

    // 4. parameter values
    std::vector<double> params;
    params.reserve(static_cast<std::size_t>(nparams));
    for (int i = 0; i < nparams; ++i) {
      auto pl = next_line();
      if (pl.empty())
        return fail("unexpected end of file: expected param line");
      ParamVals pd{};
      if (!bp::parse(strip_param_prefix(pl), param_p, bp::ws, pd))
        return fail("malformed param line");
      params.push_back(pd.val);
    }

    // 5. construct via the factory
    potentials.emplace_back(make(params, rng.rmin, rng.rmax));
  }
  return potentials;
}

} // namespace potfit::io
