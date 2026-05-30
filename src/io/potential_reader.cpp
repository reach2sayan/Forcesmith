#include "potfit/io/potential_reader.hpp"

#include "potfit/potentials/analytic_potential.hpp"
#include "potfit/potentials/spline.hpp"

#include <boost/parser/parser.hpp>
#include <boost/leaf/error.hpp>

#include <ranges>
#include <string>

namespace bp   = boost::parser;
namespace leaf = boost::leaf;

namespace potfit::io {

namespace {

struct FLine    { int fmt; int count; };           // #F <fmt> <count>
struct DistLine { double rmin, rmax; int nknots; }; // <rmin> <rmax> <nknots>
struct ParamVals{ double val, lo, hi; };            // <val> <lo> <hi>  (after stripping "param <name>")

inline auto const f_p        = bp::lit("#F")  >> bp::int_    >> bp::int_;
inline auto const dist_p     = bp::double_    >> bp::double_ >> bp::int_;
inline auto const param_p    = bp::double_    >> bp::double_ >> bp::double_;
inline auto const type_rng_p = bp::double_    >> bp::double_;

// Strip "param <name> " prefix and return the remainder for numeric parsing.
std::string_view strip_param_prefix(std::string_view sv) {
    if (sv.starts_with("param")) sv.remove_prefix(5);
    while (!sv.empty() && (sv[0] == ' ' || sv[0] == '\t')) sv.remove_prefix(1);
    while (!sv.empty() &&  sv[0] != ' ' && sv[0] != '\t')  sv.remove_prefix(1);
    return sv;
}

} // namespace

leaf::result<std::vector<Potential>>
parse_potential(std::string_view input)
{
    std::size_t line_num = 0;
    auto fail = [&](std::string msg) -> leaf::result<std::vector<Potential>> {
        return leaf::new_error(ParseError{std::move(msg), line_num});
    };

    // ── Split into trimmed lines ───────────────────────────────────────────────
    std::vector<std::string_view> lines;
    for (auto const lr : input | std::views::split('\n')) {
        std::string_view sv{lr.begin(), lr.end()};
        if (!sv.empty() && sv.back() == '\r') sv.remove_suffix(1);
        lines.push_back(sv);
    }

    // ── Phase 1: parse header block (up to and including #E) ──────────────────
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

    int fmt = -1, num_funcs = 0;
    bool header_done = false;

    while (!header_done) {
        auto line = next_line();
        if (line.empty())
            return fail("unexpected end of file before #E header terminator");

        if (line.starts_with("#F")) {
            FLine fl{};
            if (!bp::parse(line, f_p, bp::ws, fl))
                return fail("malformed #F directive");
            fmt        = fl.fmt;
            num_funcs  = fl.count;
        } else if (line == "#E") {
            if (fmt < 0)
                return fail("#E reached without a preceding #F directive");
            header_done = true;
        } else if (line[0] == '#') {
            // #T, #C, #I, #G — skip silently
        } else {
            return fail("non-directive line before #E header terminator");
        }
    }

    if (fmt != 3 && fmt != 0)
        return fail("unsupported potential format " + std::to_string(fmt));

    std::vector<Potential> potentials;
    potentials.reserve(static_cast<std::size_t>(num_funcs));

    // ── Format 3: tabulated equal-spaced ──────────────────────────────────────
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
            const auto& d = dists[static_cast<std::size_t>(f)];
            std::vector<double> x(static_cast<std::size_t>(d.nknots));
            std::vector<double> y(static_cast<std::size_t>(d.nknots));
            const double step = (d.rmax - d.rmin) / (d.nknots - 1);
            for (int k = 0; k < d.nknots; ++k) {
                x[static_cast<std::size_t>(k)] = d.rmin + k * step;
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

    // ── Format 0: analytic ────────────────────────────────────────────────────
    for (int f = 0; f < num_funcs; ++f) {
        auto type_line = next_line();
        if (type_line.empty())
            return fail("unexpected end of file: expected 'type <name>' line");
        if (!type_line.starts_with("type "))
            return fail("expected 'type <name>', got: " + std::string(type_line));
        std::string_view func_name = type_line.substr(5);
        if (auto p = func_name.find_first_not_of(' '); p != std::string_view::npos)
            func_name = func_name.substr(p);

        int nparams = 0;
        if (func_name == "pair_lj")
            nparams = 2;
        else if (func_name == "morse")
            nparams = 3;
        else
            return fail("unknown analytic function: " + std::string(func_name));

        auto rng_line = next_line();
        if (rng_line.empty())
            return fail("unexpected end of file: expected rmin rmax line");
        struct RngLine { double rmin, rmax; };
        RngLine rng{};
        if (!bp::parse(rng_line, type_rng_p, bp::ws, rng))
            return fail("malformed rmin/rmax line");

        std::vector<double> params;
        params.reserve(static_cast<std::size_t>(nparams));
        for (int p = 0; p < nparams; ++p) {
            auto pl = next_line();
            if (pl.empty())
                return fail("unexpected end of file: expected param line");
            ParamVals pd{};
            if (!bp::parse(strip_param_prefix(pl), param_p, bp::ws, pd))
                return fail("malformed param line");
            params.push_back(pd.val);
        }

        if (func_name == "pair_lj") {
            potentials.emplace_back(
                LennardJones(params[0], params[1], rng.rmin, rng.rmax));
        } else {
            potentials.emplace_back(
                Morse(params[0], params[1], params[2], rng.rmin, rng.rmax));
        }
    }
    return potentials;
}

}  // namespace potfit::io
