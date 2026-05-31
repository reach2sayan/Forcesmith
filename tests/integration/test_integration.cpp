#include "potfit/io/config_reader.hpp"
#include "potfit/io/force_model_reader.hpp"
#include "potfit/io/potential_reader.hpp"
#include "potfit/core/neighbor_list.hpp"
#include "potfit/force/eam_force.hpp"
#include "potfit/force/pair_force.hpp"
#include "potfit/force/stiweb_force.hpp"
#include "potfit/force/tersoff_force.hpp"
#include "potfit/force/potential_table.hpp"
#include "potfit/optimization/optimizer.hpp"

#include <boost/leaf/handle_errors.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace leaf = boost::leaf;
using namespace potfit;
using namespace potfit::io;

// ── Helper: build a minimal 2-atom Configuration ─────────────────────────────

static Configuration make_dimer(Vec3 pos0, Vec3 pos1, double box = 100.0) {
    Configuration cfg;
    cfg.bc = PeriodicBC(box * Mat3::Identity());
    Atom a0, a1;
    a0.type = 0; a0.pos = pos0;
    a1.type = 0; a1.pos = pos1;
    cfg.atoms = {a0, a1};
    return cfg;
}

static Configuration make_trimer(Vec3 pos0, Vec3 pos1, Vec3 pos2, double box = 100.0) {
    Configuration cfg;
    cfg.bc = PeriodicBC(box * Mat3::Identity());
    Atom a0, a1, a2;
    a0.type = 0; a0.pos = pos0;
    a1.type = 0; a1.pos = pos1;
    a2.type = 0; a2.pos = pos2;
    cfg.atoms = {a0, a1, a2};
    return cfg;
}

// ── Test 1: LJ pair energy at equilibrium ─────────────────────────────────────

TEST(Integration, LJPair_EnergyAtEquilibrium) {
    // LJ(ε=1,σ=1) equilibrium at r = 2^(1/6), energy = -1.0
    const double r_eq = std::pow(2.0, 1.0 / 6.0);

    double calc_e = 0.0;
    std::string err;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(pots, parse_potential(R"({
              "format": "analytic",
              "potentials": [
                {"type":"pair_lj","rmin":0.5,"rmax":4.0,"epsilon":1.0,"sigma":1.0}
              ]
            })"));

            PairForceCalculator calc;
            calc.pair.reserve(1);
            calc.pair.emplace_back(pots[0]);

            auto cfg = make_dimer({0.0, 0.0, 0.0}, {r_eq, 0.0, 0.0});
            calc.eval_forces(cfg);
            calc_e = cfg.calc_energy;
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(calc_e, -1.0, 1e-10);
}

// ── Test 2: Morse pair force at equilibrium ────────────────────────────────────

TEST(Integration, MorsePair_ForceAtEquilibrium) {
    // Morse(De=1, a=1.5, re=2.5): deriv(re) = 0 → zero force at equilibrium
    double f0_norm = 0.0, f1_norm = 0.0;
    std::string err;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(pots, parse_potential(R"({
              "format": "analytic",
              "potentials": [
                {"type":"morse","rmin":0.5,"rmax":6.0,"De":1.0,"a":1.5,"re":2.5}
              ]
            })"));

            PairForceCalculator calc;
            calc.pair.reserve(1);
            calc.pair.emplace_back(pots[0]);

            auto cfg = make_dimer({0.0, 0.0, 0.0}, {2.5, 0.0, 0.0});
            calc.eval_forces(cfg);
            f0_norm = cfg.atoms[0].calc_force.norm();
            f1_norm = cfg.atoms[1].calc_force.norm();
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(f0_norm, 0.0, 1e-10);
    EXPECT_NEAR(f1_norm, 0.0, 1e-10);
}

// ── Test 3: Tabulated potential evaluates at knot ─────────────────────────────

TEST(Integration, TabulatedPair_EnergyAtKnot) {
    // 3 knots: r=1.0→0.0, r=2.0→LJ(2.0), r=3.0→LJ(3.0)
    // LJ(ε=1,σ=1) at r=2.0: 4*(2^-12 - 2^-6) = 4*(1/4096 - 1/64) ≈ -0.061523
    // Test at r=2.0 (exact knot point) → energy = -0.061523
    const double lj_at_2 = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0); // ≈ -0.061523

    double calc_e = 0.0;
    std::string err;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(pots, parse_potential(R"({
              "format": "tabulated",
              "potentials": [
                {"rmin": 1.0, "rmax": 3.0,
                 "knots": [0.0, -0.061523438, -0.005487361]}
              ]
            })"));

            PairForceCalculator calc;
            calc.pair.reserve(1);
            calc.pair.emplace_back(pots[0]);

            auto cfg = make_dimer({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0});
            calc.eval_forces(cfg);
            calc_e = cfg.calc_energy;
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    // At an exact knot point the spline reproduces the value exactly
    EXPECT_NEAR(calc_e, lj_at_2, 1e-6);
}

// ── Test 4: EAM dimer gives negative total energy ─────────────────────────────

TEST(Integration, EAMDimer_EnergyIsNegative) {
    // Simple EAM: repulsive pair + exponential density + sqrt embedding
    // The embedding energy −√ρ dominates → total E < 0
    double calc_e = 0.0;
    std::string err;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(fm, parse_force_model(R"({
              "model": "eam",
              "ntypes": 1,
              "pair": {
                "format": "tabulated",
                "potentials": [{"rmin":1.5,"rmax":6.0,"knots":[2.0,1.0,0.0,1.0,2.0]}]
              },
              "density": {
                "format": "tabulated",
                "potentials": [{"rmin":1.5,"rmax":6.0,"knots":[1.0,0.5,0.0,0.0,0.0]}]
              },
              "embedding": {
                "format": "analytic",
                "potentials": [{"type":"sqrt","rmin":0.0,"rmax":5.0,"A":-1.0,"B":0.0}]
              }
            })"));

            auto &eam = std::get<EAMForceCalculator>(fm);
            auto cfg = make_dimer({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0});
            build_neighbor_list(cfg, 6.0);
            eam.eval_forces(cfg);
            calc_e = cfg.calc_energy;
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_TRUE(std::isfinite(calc_e));
    EXPECT_LT(calc_e, 0.0);
}

// ── Test 5: Tersoff Si — force/energy FD consistency ─────────────────────────

TEST(Integration, TersoffSi_ForceEnergyFDConsistency) {
    // Tersoff Si 1988 parameters; dimer at r=2.5 Å (well within cutoff R=2.7)
    std::string err;
    double fx_calc = 0.0, fx_fd = 0.0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(fm, parse_force_model(R"({
              "model": "tersoff",
              "ntypes": 1,
              "potentials": [{
                "A":1830.8,"B":471.18,
                "lambda":2.4799,"mu":1.7322,
                "beta":1.1e-6,"n":0.78734,
                "c":100390.0,"d":16.218,"h":-0.59825,
                "R":2.7,"S":3.0
              }]
            })"));

            auto &tc = std::get<TersoffForceCalculator>(fm);
            const double rcut = 3.5;

            auto energy_at = [&](double x0) -> double {
                auto c = make_dimer({x0, 0.0, 0.0}, {2.5, 0.0, 0.0});
                build_neighbor_list(c, rcut);
                tc.eval_forces(c);
                return c.calc_energy;
            };

            const double dr = 1e-5;
            fx_fd = -(energy_at(dr / 2.0) - energy_at(-dr / 2.0)) / dr;

            auto cfg = make_dimer({0.0, 0.0, 0.0}, {2.5, 0.0, 0.0});
            build_neighbor_list(cfg, rcut);
            tc.eval_forces(cfg);
            fx_calc = cfg.atoms[0].calc_force[0];
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(fx_calc, fx_fd, 1e-5 * std::abs(fx_calc) + 1e-8);
}

// ── Test 6: Stillinger-Weber Si — tetrahedral angle minimizes energy ──────────

TEST(Integration, StiwebSi_TetAngleMinimizesEnergy) {
    // SW Si: 3-body term λ(cos θ + 1/3)² vanishes exactly at θ=109.47°.
    // With r = 2.35 Å (< cutoff = a*σ = 3.771 Å) and atoms 1,2 separated
    // beyond cutoff at θ=109.47° (|12| ≈ 3.84 > 3.771) → no 12 pair interaction.
    // E(109.47°) = 2*v2(2.35) + 0   vs   E(90°) = 2*v2(2.35) + v2(r12) + v3
    // Since 3-body at 90° is positive, E(109.47°) < E(90°).
    const double d = 2.35;
    const double cos_tet = -1.0 / 3.0;
    const double sin_tet = std::sqrt(1.0 - cos_tet * cos_tet); // 2√2/3
    const Vec3 pos_tet = {d * cos_tet, d * sin_tet, 0.0};
    const Vec3 pos_90  = {0.0, d, 0.0};

    std::string err;
    double e_tet = 0.0, e_90 = 0.0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            // potfit SW form: textbook Si params converted to
            //   A' = A·B·σ^p, B' = A·σ^q, delta = σ, a1 = a·σ,
            //   gamma' = γ·σ, a2 = a·σ;  lambda per-triplet.
            BOOST_LEAF_AUTO(fm, parse_force_model(R"({
              "model": "stiweb",
              "ntypes": 1,
              "potentials": [{
                "A":81.795,"B":7.0496,
                "p":4.0,"q":0.0,
                "delta":2.0951,"a1":3.77118,
                "gamma":2.51412,"a2":3.77118
              }],
              "lambda":[21.0]
            })"));

            auto &sw = std::get<StiwebForceCalculator>(fm);
            const double rcut = 4.0;

            auto cfg_tet = make_trimer({0.0,0.0,0.0}, {d,0.0,0.0}, pos_tet);
            build_neighbor_list(cfg_tet, rcut);
            sw.eval_forces(cfg_tet);
            e_tet = cfg_tet.calc_energy;

            auto cfg_90 = make_trimer({0.0,0.0,0.0}, {d,0.0,0.0}, pos_90);
            build_neighbor_list(cfg_90, rcut);
            sw.eval_forces(cfg_90);
            e_90 = cfg_90.calc_energy;
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_LT(e_tet, e_90)
        << "tetrahedral energy " << e_tet
        << " should be less than 90-degree energy " << e_90;
}

// ── Helper: 5-atom tetrahedral cluster (central + 4 at tet positions) ────────

static Configuration make_tetrahedron(double r, double box = 100.0) {
    Configuration cfg;
    cfg.bc = PeriodicBC(box * Mat3::Identity());
    // Tetrahedral unit vectors: body diagonals of a cube, normalised.
    const std::array<Vec3, 4> dirs = {{
        Vec3( 1.0,  1.0,  1.0).normalized(),
        Vec3( 1.0, -1.0, -1.0).normalized(),
        Vec3(-1.0,  1.0, -1.0).normalized(),
        Vec3(-1.0, -1.0,  1.0).normalized(),
    }};
    Atom center;
    center.type = 0; center.pos = Vec3::Zero();
    cfg.atoms.push_back(center);
    for (const auto &d : dirs) {
        Atom a; a.type = 0; a.pos = r * d;
        cfg.atoms.push_back(a);
    }
    return cfg;
}

// ── Test 7: LJ optimizer convergence ─────────────────────────────────────────

TEST(Integration, OptimizerLJ_ConvergesFromWrongParams) {
    // 3 reference configs with energies/forces from LJ(ε=1, σ=1).
    // Start from LJ(ε=1.5, σ=1.2) and check that optimizer recovers ε≈1, σ≈1.
    //
    // LJ(1,1) values:
    //   r=2^(1/6) ≈ 1.12246: E=-1.0,  F=0
    //   r=1.5:                E=-0.32034, F_atom0=(+1.15800,0,0)
    //   r=2.0:                E=-0.06152, F_atom0=(+0.18164,0,0)
    const char *cfg_json = R"([
      {
        "X":[30,0,0],"Y":[0,30,0],"Z":[0,0,30],"E":-1.0,
        "atoms":[
          {"element":"Cu","position":[0.0,0.0,0.0],"force":[0.0,0.0,0.0]},
          {"element":"Cu","position":[1.12246,0.0,0.0],"force":[0.0,0.0,0.0]}
        ]
      },
      {
        "X":[30,0,0],"Y":[0,30,0],"Z":[0,0,30],"E":-0.32034,
        "atoms":[
          {"element":"Cu","position":[0.0,0.0,0.0],"force":[1.15800,0.0,0.0]},
          {"element":"Cu","position":[1.5,0.0,0.0],"force":[-1.15800,0.0,0.0]}
        ]
      },
      {
        "X":[30,0,0],"Y":[0,30,0],"Z":[0,0,30],"E":-0.06152,
        "atoms":[
          {"element":"Cu","position":[0.0,0.0,0.0],"force":[0.18164,0.0,0.0]},
          {"element":"Cu","position":[2.0,0.0,0.0],"force":[-0.18164,0.0,0.0]}
        ]
      }
    ])";

    // Initial potential: LJ(ε=1.5, σ=1.2) — wrong parameters
    const char *pot_json = R"({
      "format": "analytic",
      "potentials": [
        {"type":"pair_lj","rmin":0.5,"rmax":4.0,"epsilon":1.5,"sigma":1.2}
      ]
    })";

    std::string err;
    double eps_final = 0.0, sig_final = 0.0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(configs, parse_config(cfg_json));
            BOOST_LEAF_AUTO(pots, parse_potential(pot_json));
            ForceCalculator model = make_pair_force_calculator(std::move(pots));

            OptimizerOptions opts;
            opts.max_iter = 500;
            opts.energy_weight = 1.0;

            run_optimizer(configs, model, opts);

            const auto& pair_calc = std::get<PairForceCalculator>(model);
            Eigen::VectorXd x(pair_calc.pair[std::size_t{0}, std::size_t{0}].param_count());
            pair_calc.pair[std::size_t{0}, std::size_t{0}].gather_params(x, 0);
            eps_final = x[0]; // epsilon
            sig_final = x[1]; // sigma
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(eps_final, 1.0, 0.1) << "epsilon should converge to 1.0";
    EXPECT_NEAR(sig_final, 1.0, 0.1) << "sigma should converge to 1.0";
}

// ── Test 8: EAM dimer force/energy FD consistency ─────────────────────────────

TEST(Integration, EAMDimer_ForceConsistency) {
    std::string err;
    double fx_calc = 0.0, fx_fd = 0.0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(fm, parse_force_model(R"({
              "model":"eam","ntypes":1,
              "pair":    {"format":"tabulated","potentials":[
                           {"rmin":1.5,"rmax":6.0,
                            "knots":[2.0,1.5,1.0,0.5,0.1,0.0]}]},
              "density": {"format":"tabulated","potentials":[
                           {"rmin":1.5,"rmax":6.0,
                            "knots":[1.0,0.8,0.5,0.2,0.05,0.0]}]},
              "embedding":{"format":"analytic","potentials":[
                           {"type":"sqrt","rmin":0.0,"rmax":5.0,
                            "A":-1.0,"B":0.0}]}
            })"));
            auto &eam = std::get<EAMForceCalculator>(fm);

            const double h = 1e-5;
            auto energy_at = [&](double x0) {
                auto cfg = make_dimer({x0, 0.0, 0.0}, {2.5, 0.0, 0.0});
                eam.eval_forces(cfg);
                return cfg.calc_energy;
            };
            fx_fd = -(energy_at(h / 2.0) - energy_at(-h / 2.0)) / h;

            auto cfg = make_dimer({0.0, 0.0, 0.0}, {2.5, 0.0, 0.0});
            eam.eval_forces(cfg);
            fx_calc = cfg.atoms[0].calc_force[0];
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(fx_calc, fx_fd, 1e-5 * std::abs(fx_calc) + 1e-8);
}

// ── Test 9: EAM 3-atom config — Newton's 3rd law ──────────────────────────────

TEST(Integration, EAM_ThreeAtom_NewtonThirdLaw) {
    std::string err;
    Vec3 sum_force = Vec3::Zero();
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(fm, parse_force_model(R"({
              "model":"eam","ntypes":1,
              "pair":    {"format":"tabulated","potentials":[
                           {"rmin":1.5,"rmax":6.0,
                            "knots":[2.0,1.5,1.0,0.5,0.1,0.0]}]},
              "density": {"format":"tabulated","potentials":[
                           {"rmin":1.5,"rmax":6.0,
                            "knots":[1.0,0.8,0.5,0.2,0.05,0.0]}]},
              "embedding":{"format":"analytic","potentials":[
                           {"type":"sqrt","rmin":0.0,"rmax":5.0,
                            "A":-1.0,"B":0.0}]}
            })"));
            auto &eam = std::get<EAMForceCalculator>(fm);

            // 3 distinct pairwise distances — embedding gradient couples all atoms
            auto cfg = make_trimer({0.0,0.0,0.0}, {2.0,0.0,0.0}, {0.0,2.5,0.0});
            eam.eval_forces(cfg);
            for (const auto &a : cfg.atoms)
                sum_force += a.calc_force;
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(sum_force.norm(), 0.0, 1e-10);
}

// ── Test 10: Tersoff Si — tetrahedral cluster more bound than dimer ────────────

TEST(Integration, TersoffSi_TetCluster_MoreBoundThanDimer) {
    std::string err;
    double e_dimer_per_atom = 0.0, e_tet_per_atom = 0.0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            BOOST_LEAF_AUTO(fm, parse_force_model(R"({
              "model": "tersoff",
              "ntypes": 1,
              "potentials": [{
                "A":1830.8,"B":471.18,
                "lambda":2.4799,"mu":1.7322,
                "beta":1.1e-6,"n":0.78734,
                "c":100390.0,"d":16.218,"h":-0.59825,
                "R":2.7,"S":3.0
              }]
            })"));
            auto &tc = std::get<TersoffForceCalculator>(fm);
            const double r = 2.35;

            auto cfg_dimer = make_dimer({0.0,0.0,0.0}, {r,0.0,0.0});
            tc.eval_forces(cfg_dimer);
            e_dimer_per_atom = cfg_dimer.calc_energy
                               / static_cast<double>(cfg_dimer.atoms.size());

            auto cfg_tet = make_tetrahedron(r);
            tc.eval_forces(cfg_tet);
            e_tet_per_atom = cfg_tet.calc_energy
                             / static_cast<double>(cfg_tet.atoms.size());
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_LT(e_tet_per_atom, e_dimer_per_atom)
        << "tet E/atom=" << e_tet_per_atom
        << " should be < dimer E/atom=" << e_dimer_per_atom;
}

// ── Test 11: EAM optimizer convergence from wrong embedding parameter ──────────

TEST(Integration, OptimizerEAM_ConvergesFromWrongParams) {
    // True model: power_decay pair + exp_decay density + sqrt embedding
    static constexpr const char* kEAMTrue = R"({
      "model":"eam","ntypes":1,
      "pair":    {"format":"analytic","potentials":[
                   {"type":"power_decay","rmin":1.5,"rmax":6.0,"A":100.0,"n":12.0}]},
      "density": {"format":"analytic","potentials":[
                   {"type":"exp_decay","rmin":1.5,"rmax":6.0,"A":1.0,"B":1.0}]},
      "embedding":{"format":"analytic","potentials":[
                   {"type":"sqrt","rmin":0.0,"rmax":5.0,"A":-1.0,"B":0.0}]}
    })";
    static constexpr const char* kEAMPert = R"({
      "model":"eam","ntypes":1,
      "pair":    {"format":"analytic","potentials":[
                   {"type":"power_decay","rmin":1.5,"rmax":6.0,"A":100.0,"n":12.0}]},
      "density": {"format":"analytic","potentials":[
                   {"type":"exp_decay","rmin":1.5,"rmax":6.0,"A":1.0,"B":1.0}]},
      "embedding":{"format":"analytic","potentials":[
                   {"type":"sqrt","rmin":0.0,"rmax":5.0,"A":-1.4,"B":0.0}]}
    })";

    std::string err;
    double fx_ref = 0.0, e_ref = 0.0;
    double fx_opt = 0.0, e_opt = 0.0;
    leaf::try_handle_all(
        [&]() -> leaf::result<void> {
            // Build reference data from true model
            BOOST_LEAF_AUTO(fm_true, parse_force_model(kEAMTrue));
            auto &eam_true = std::get<EAMForceCalculator>(fm_true);

            std::vector<Configuration> configs = {
                make_dimer({0.0,0.0,0.0}, {2.0,0.0,0.0}),
                make_dimer({0.0,0.0,0.0}, {2.5,0.0,0.0}),
                make_dimer({0.0,0.0,0.0}, {3.0,0.0,0.0}),
            };
            for (auto &cfg : configs) {
                eam_true.eval_forces(cfg);
                for (auto &a : cfg.atoms) a.force = a.calc_force;
                cfg.energy = cfg.calc_energy;
            }

            // Reference at r=2.5 for comparison
            {
                auto cfg = make_dimer({0.0,0.0,0.0}, {2.5,0.0,0.0});
                eam_true.eval_forces(cfg);
                fx_ref = cfg.atoms[0].calc_force[0];
                e_ref  = cfg.calc_energy;
            }

            // Optimise the perturbed model
            BOOST_LEAF_AUTO(fm_pert, parse_force_model(kEAMPert));
            OptimizerOptions opts;
            opts.max_iter      = 300;
            opts.energy_weight = 1.0;
            run_optimizer(configs, fm_pert, opts);

            // Evaluate with optimised params
            auto &eam_opt = std::get<EAMForceCalculator>(fm_pert);
            auto cfg_opt = make_dimer({0.0,0.0,0.0}, {2.5,0.0,0.0});
            eam_opt.eval_forces(cfg_opt);
            fx_opt = cfg_opt.atoms[0].calc_force[0];
            e_opt  = cfg_opt.calc_energy;
            return {};
        },
        [&](const ParseError &e) { err = e.message; },
        [&]() { err = "unknown error"; }
    );
    ASSERT_TRUE(err.empty()) << err;
    EXPECT_NEAR(fx_opt, fx_ref, 0.1 * std::abs(fx_ref) + 1e-6)
        << "optimised force should match reference";
    EXPECT_NEAR(e_opt, e_ref, 0.1 * std::abs(e_ref) + 1e-6)
        << "optimised energy should match reference";
}

// ── Test 12: Tersoff Si optimizer — recover A and B with others fixed ──────────

TEST(Integration, OptimizerTersoff_ConvergesFromWrongParams) {
    // Helper: build a Si 1988 TersoffForceCalculator with A and B free,
    // all other params optionally fixed.
    auto make_si_calc = [](double A, double B, bool fix_others) {
        TersoffParams p;
        p.A      = {A,        false};
        p.B      = {B,        false};
        p.lambda = {2.4799,   fix_others};
        p.mu     = {1.7322,   fix_others};
        p.beta   = {1.1e-6,   fix_others};
        p.n      = {0.78734,  fix_others};
        p.c      = {100390.0, fix_others};
        p.d      = {16.218,   fix_others};
        p.h      = {-0.59825, fix_others};
        p.R      = {2.7,      fix_others};
        p.S      = {3.0,      fix_others};
        TersoffForceCalculator tc;
        tc.params.reserve(1);
        tc.params.emplace_back(std::move(p));
        return tc;
    };

    // Reference: true Si params, nothing fixed
    auto tc_true = make_si_calc(1830.8, 471.18, false);

    std::vector<Configuration> configs = {
        make_dimer({0.0,0.0,0.0}, {2.35,0.0,0.0}),
        make_dimer({0.0,0.0,0.0}, {2.5, 0.0,0.0}),
    };
    for (auto &cfg : configs) {
        tc_true.eval_forces(cfg);
        for (auto &a : cfg.atoms) a.force = a.calc_force;
        cfg.energy = cfg.calc_energy;
    }

    // Perturbed: A=2100, B=580; all other 9 params fixed
    ForceCalculator fc = make_si_calc(2100.0, 580.0, true);
    // param_count() == 2: only A and B are free

    OptimizerOptions opts;
    opts.max_iter      = 400;
    opts.energy_weight = 1.0;
    run_optimizer(configs, fc, opts);

    // Recover A and B
    Eigen::VectorXd x(2);
    std::visit([&](const auto &m) { m.gather_params(x, std::size_t{0}); }, fc);
    const double A_rec = x[0];
    const double B_rec = x[1];

    EXPECT_NEAR(A_rec, 1830.8, 200.0) << "recovered A should be near 1830.8";
    EXPECT_NEAR(B_rec, 471.18,  60.0) << "recovered B should be near 471.18";
}
