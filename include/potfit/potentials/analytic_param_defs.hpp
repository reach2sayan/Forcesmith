#pragma once

// Single source of truth for analytic-potential parameter NAMES + scaffolding
// DEFAULTS. Each function lists its parameters once as an X-macro row
//
//     X(token, default_value, min, max)
//
// in the SAME order the registry maker passes them to the constructor
// (src/io/potential_reader.cpp). `BOOST_PP_STRINGIZE(token)` turns the token into
// the JSON key string, so the reader registry's `param_names` and the
// `potfit init` scaffolder's default table both expand from this one
// definition — they are identical by construction, not merely tested to agree.
//
// Defaults are ported from upstream makeapot (util/potfit/functions.py),
// mapped positionally onto the port's parameter names (the port renames a few:
// makeapot D_e/alpha/beta → port De/A/B, etc.).

#include <boost/preprocessor/stringize.hpp>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace potfit {

// One scaffolding default: the parameter's JSON key plus its start value and
// box constraints, ready to emit as {"value","min","max"}.
struct AnalyticParamDef {
  std::string_view name;
  double value;
  double min;
  double max;
};

// ── The per-function parameter tables ───────────────────────────────────────
// Order MUST match the registry maker's constructor-argument order. Adding a
// function here + a row in POTFIT_ANALYTIC_FUNCTIONS below is all it takes to
// teach both the reader and the scaffolder a new function.
//                                token        value     min       max
#define POTFIT_APD_lj(X)          X(epsilon,    0.1,      0.0,      1.0)        \
                                  X(sigma,      2.5,      1.0,      4.0)
#define POTFIT_APD_morse(X)       X(De,         0.1,      0.0,      1.0)        \
                                  X(a,          2.0,      1.0,      5.0)        \
                                  X(re,         2.5,      1.0,      5.0)
#define POTFIT_APD_buckingham(X)  X(A,          1.0,    -10.0,     10.0)        \
                                  X(rho,        1.0,    -10.0,     10.0)        \
                                  X(C,          1.0,    -10.0,     10.0)
#define POTFIT_APD_born(X)        X(A,          0.1,      0.0,     10.0)        \
                                  X(B,          2.0,      0.0,     10.0)        \
                                  X(C,          3.0,      0.0,     10.0)        \
                                  X(D,          2.0,      0.0,     10.0)        \
                                  X(E,          2.5,      1.0,      8.0)
#define POTFIT_APD_power_decay(X) X(A,          1.0,      0.1,     10.0)        \
                                  X(n,          2.0,      1.0,      5.0)
#define POTFIT_APD_exp_decay(X)   X(A,          4.0,      1.0,     20.0)        \
                                  X(B,          1.0,      0.5,      5.0)
#define POTFIT_APD_mexp_decay(X)  X(A,          0.1,      0.0,     10.0)        \
                                  X(B,          0.1,      0.0,     10.0)        \
                                  X(r0,         2.0,      0.0,     10.0)
#define POTFIT_APD_harmonic(X)    X(k,          0.1,      0.0,     10.0)        \
                                  X(r0,         2.0,      0.0,      8.0)
#define POTFIT_APD_universal(X)   X(E0,         1.0,    -10.0,     10.0)        \
                                  X(a,          1.0,      0.0,     20.0)        \
                                  X(b,          2.0,      0.0,     20.0)        \
                                  X(c,          0.0,     -1.0,      1.0)
#define POTFIT_APD_eopp(X)        X(A,         15.0,      1.0,  10000.0)        \
                                  X(n,          6.0,      1.0,     20.0)        \
                                  X(B,          5.0,   -100.0,    100.0)        \
                                  X(m,          3.0,      1.0,     10.0)        \
                                  X(k,          2.5,      0.0,      6.0)        \
                                  X(phi,        3.0,      0.0,      6.3)
#define POTFIT_APD_sqrt(X)        X(A,          0.1,      0.0,     10.0)        \
                                  X(B,          2.0,      0.0,     10.0)
#define POTFIT_APD_const(X)       X(C,          1.0,      0.0,      2.0)
#define POTFIT_APD_parabola(X)    X(A,          1.0,    -10.0,     10.0)        \
                                  X(B,          1.0,    -10.0,     10.0)        \
                                  X(C,          1.0,    -10.0,     10.0)
#define POTFIT_APD_softshell(X)   X(A,          1.0,      0.1,     10.0)        \
                                  X(n,          2.0,      1.0,      5.0)

// Smooth-cutoff (`_sc`) variants: the base parameters plus the appended
// switching width `h` (makeapot default 1 ∈ [0.5, 2]).
#define POTFIT_APD_SC_H(X)        X(h,          1.0,      0.5,      2.0)
#define POTFIT_APD_lj_sc(X)       POTFIT_APD_lj(X)        POTFIT_APD_SC_H(X)
#define POTFIT_APD_morse_sc(X)    POTFIT_APD_morse(X)     POTFIT_APD_SC_H(X)
#define POTFIT_APD_exp_decay_sc(X) POTFIT_APD_exp_decay(X) POTFIT_APD_SC_H(X)
#define POTFIT_APD_eopp_sc(X)     POTFIT_APD_eopp(X)      POTFIT_APD_SC_H(X)

// The master list: F(registry_name, table_macro). registry_name is the
// canonical "type" string the reader/scaffolder use.
#define POTFIT_ANALYTIC_FUNCTIONS(F)                                           \
  F(lj, POTFIT_APD_lj)                                                         \
  F(morse, POTFIT_APD_morse)                                                   \
  F(buckingham, POTFIT_APD_buckingham)                                         \
  F(born, POTFIT_APD_born)                                                     \
  F(power_decay, POTFIT_APD_power_decay)                                       \
  F(exp_decay, POTFIT_APD_exp_decay)                                           \
  F(mexp_decay, POTFIT_APD_mexp_decay)                                         \
  F(harmonic, POTFIT_APD_harmonic)                                             \
  F(universal, POTFIT_APD_universal)                                           \
  F(eopp, POTFIT_APD_eopp)                                                     \
  F(sqrt, POTFIT_APD_sqrt)                                                     \
  F(const, POTFIT_APD_const)                                                   \
  F(parabola, POTFIT_APD_parabola)                                             \
  F(softshell, POTFIT_APD_softshell)                                           \
  F(lj_sc, POTFIT_APD_lj_sc)                                                   \
  F(morse_sc, POTFIT_APD_morse_sc)                                             \
  F(exp_decay_sc, POTFIT_APD_exp_decay_sc)                                     \
  F(eopp_sc, POTFIT_APD_eopp_sc)

// ── Expansion helpers ───────────────────────────────────────────────────────

// Expand a table macro into a brace-init list of parameter-name strings, e.g.
//   POTFIT_PARAM_NAME_LIST(POTFIT_APD_lj)  →  std::vector<std::string>{ "epsilon", "sigma", }
// Use this in the reader registry so its names ARE these tokens, by definition.
#define POTFIT_APD_AS_NAME(tok, v, lo, hi) std::string(BOOST_PP_STRINGIZE(tok)),
#define POTFIT_PARAM_NAME_LIST(TABLE)                                          \
  std::vector<std::string> { TABLE(POTFIT_APD_AS_NAME) }

// Span of AnalyticParamDef for a function name; empty span if unknown. Lookup
// over the master list, so it covers exactly the functions defined above.
[[nodiscard]] std::span<const AnalyticParamDef>
analytic_defaults(std::string_view function);

// All function names with a default table (the master list), for tests that
// cross-check the macro names/order against the reader registry.
[[nodiscard]] std::span<const std::string_view> analytic_default_functions();

} // namespace potfit
