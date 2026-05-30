#include "potfit/io/config_reader.hpp"

#include <boost/parser/parser.hpp>
#include <boost/leaf/error.hpp>

#include <ranges>
#include <string>

namespace bp   = boost::parser;
namespace leaf = boost::leaf;

namespace potfit::io {

namespace {

// ── Aggregate structs: field count must match non-void parser attributes ──────
// boost::parser uses Boost.PFR to fill these directly from sequence attributes.

struct NLine    { int    natoms; int useforce; };  // 2 non-void attrs
struct Vec3Line { double x, y, z; };               // 3 non-void attrs
struct SLine    { double xx, yy, zz, xy, yz, zx; };// 6 non-void attrs
struct AtomFull { int type; double px, py, pz, fx, fy, fz; }; // 7
struct AtomPos  { int type; double px, py, pz; };              // 4

// ── Parsers ───────────────────────────────────────────────────────────────────

inline auto const n_p = bp::lit("#N") >> bp::int_    >> bp::int_;
inline auto const x_p = bp::lit("#X") >> bp::double_ >> bp::double_ >> bp::double_;
inline auto const y_p = bp::lit("#Y") >> bp::double_ >> bp::double_ >> bp::double_;
inline auto const z_p = bp::lit("#Z") >> bp::double_ >> bp::double_ >> bp::double_;
inline auto const e_p = bp::lit("#E") >> bp::double_;
inline auto const w_p = bp::lit("#W") >> bp::double_;
inline auto const s_p = bp::lit("#S") >> bp::double_ >> bp::double_ >> bp::double_
                                      >> bp::double_ >> bp::double_ >> bp::double_;
inline auto const atom_full_p = bp::int_    >> bp::double_ >> bp::double_ >> bp::double_
                                            >> bp::double_ >> bp::double_ >> bp::double_;
inline auto const atom_pos_p  = bp::int_    >> bp::double_ >> bp::double_ >> bp::double_;

// ── Incremental builder ───────────────────────────────────────────────────────

struct Builder {
    Configuration cfg;
    Mat3 pending_box    = Mat3::Zero();  // accumulated from #X/#Y/#Z
    int  expected_atoms = 0;
    bool use_force      = false;
    bool active         = false;
};

void finalize(Builder& b, std::vector<Configuration>& out) {
    b.cfg.bc = PeriodicBC(b.pending_box);
    out.push_back(std::move(b.cfg));
    b = Builder{};
}

} // namespace

leaf::result<std::vector<Configuration>> parse_config(std::string_view input)
{
    std::vector<Configuration> configs;
    Builder b;
    std::size_t line_num = 0;

    auto fail = [&](std::string msg) -> leaf::result<std::vector<Configuration>> {
        return leaf::new_error(ParseError{std::move(msg), line_num});
    };

    for (auto const lr : input | std::views::split('\n')) {
        ++line_num;
        std::string_view line{lr.begin(), lr.end()};
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        if (line.find_first_not_of(" \t") == std::string_view::npos) continue;

        if (line.starts_with("#N")) {
            if (b.active) {
                if (static_cast<int>(b.cfg.atoms.size()) != b.expected_atoms)
                    return fail("incomplete configuration before next #N");
                finalize(b, configs);
            }
            NLine nd{};
            if (!bp::parse(line, n_p, bp::ws, nd))
                return fail("malformed #N directive");
            b.expected_atoms = nd.natoms;
            b.use_force      = nd.useforce != 0;
            b.active         = true;

        } else if (line.starts_with("#X")) {
            Vec3Line v{};
            if (!bp::parse(line, x_p, bp::ws, v)) return fail("malformed #X directive");
            b.pending_box.col(0) = Vec3{v.x, v.y, v.z};

        } else if (line.starts_with("#Y")) {
            Vec3Line v{};
            if (!bp::parse(line, y_p, bp::ws, v)) return fail("malformed #Y directive");
            b.pending_box.col(1) = Vec3{v.x, v.y, v.z};

        } else if (line.starts_with("#Z")) {
            Vec3Line v{};
            if (!bp::parse(line, z_p, bp::ws, v)) return fail("malformed #Z directive");
            b.pending_box.col(2) = Vec3{v.x, v.y, v.z};

        } else if (line.starts_with("#E")) {
            double e{};
            if (!bp::parse(line, e_p, bp::ws, e)) return fail("malformed #E directive");
            b.cfg.energy = e;

        } else if (line.starts_with("#W")) {
            double w{};
            if (!bp::parse(line, w_p, bp::ws, w)) return fail("malformed #W directive");
            b.cfg.weight = w;

        } else if (line.starts_with("#S")) {
            SLine s{};
            if (!bp::parse(line, s_p, bp::ws, s)) return fail("malformed #S directive");
            b.cfg.stress(0,0) = s.xx;  b.cfg.stress(1,1) = s.yy;  b.cfg.stress(2,2) = s.zz;
            b.cfg.stress(0,1) = b.cfg.stress(1,0) = s.xy;
            b.cfg.stress(1,2) = b.cfg.stress(2,1) = s.yz;
            b.cfg.stress(0,2) = b.cfg.stress(2,0) = s.zx;

        } else if (line[0] == '#') {
            continue; // #C element names, #F/#G markers, any unknown directive

        } else {
            if (!b.active)
                return fail("atom line before any #N directive");
            if (static_cast<int>(b.cfg.atoms.size()) >= b.expected_atoms)
                return fail("more atom lines than declared in #N");

            Atom a;
            if (b.use_force) {
                AtomFull af{};
                if (!bp::parse(line, atom_full_p, bp::ws, af))
                    return fail("malformed atom line (expected: type x y z fx fy fz)");
                a.type  = af.type;
                a.pos   = Vec3{af.px, af.py, af.pz};
                a.force = Vec3{af.fx, af.fy, af.fz};
            } else {
                AtomPos ap{};
                if (!bp::parse(line, atom_pos_p, bp::ws, ap))
                    return fail("malformed atom line (expected: type x y z)");
                a.type = ap.type;
                a.pos  = Vec3{ap.px, ap.py, ap.pz};
            }
            b.cfg.atoms.push_back(std::move(a));
        }
    }

    if (b.active) {
        if (static_cast<int>(b.cfg.atoms.size()) != b.expected_atoms)
            return fail("file ended with incomplete configuration");
        finalize(b, configs);
    }

    return configs;
}

} // namespace potfit::io
