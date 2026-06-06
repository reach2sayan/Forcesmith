#pragma once

// Single source of truth for analytic-potential parameter NAMES + scaffolding
// DEFAULTS. Each function lists its parameters once as an X-macro row
//
//     X(token, default_value, min, max)
//
// in the SAME order the registry maker passes them to the constructor
// (src/io/potential_reader.cpp). `BOOST_PP_STRINGIZE(token)` turns the token
// into the JSON key string, so the reader registry's `param_names` and the
// `forcesmith init` scaffolder's default table both expand from this one
// definition — they are identical by construction, not merely tested to agree.
//
// Defaults are ported from upstream makeapot (util/forcesmith/functions.py),
// mapped positionally onto the port's parameter names (the port renames a few:
// makeapot D_e/alpha/beta → port De/A/B, etc.).

#include <boost/preprocessor/stringize.hpp>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace forcesmith {

struct AnalyticParamDef {
  std::string_view name;
  double value;
  double min;
  double max;
};

// The per-function parameter tables
// Order MUST match the registry maker's constructor-argument order. Adding a
// function here + a row in FORCESMITH_ANALYTIC_FUNCTIONS below is all it takes
// to teach both the reader and the scaffolder a new function.
//                                token        value     min       max
#define FORCESMITH_APD_lj(X)                                                   \
  X(epsilon, 0.1, 0.0, 1.0)                                                    \
  X(sigma, 2.5, 1.0, 4.0)
#define FORCESMITH_APD_morse(X)                                                \
  X(De, 0.1, 0.0, 1.0)                                                         \
  X(a, 2.0, 1.0, 5.0)                                                          \
  X(re, 2.5, 1.0, 5.0)
#define FORCESMITH_APD_buckingham(X)                                           \
  X(A, 1.0, -10.0, 10.0)                                                       \
  X(rho, 1.0, -10.0, 10.0)                                                     \
  X(C, 1.0, -10.0, 10.0)
#define FORCESMITH_APD_born(X)                                                 \
  X(A, 0.1, 0.0, 10.0)                                                         \
  X(B, 2.0, 0.0, 10.0)                                                         \
  X(C, 3.0, 0.0, 10.0)                                                         \
  X(D, 2.0, 0.0, 10.0)                                                         \
  X(E, 2.5, 1.0, 8.0)
#define FORCESMITH_APD_power_decay(X)                                          \
  X(A, 1.0, 0.1, 10.0)                                                         \
  X(n, 2.0, 1.0, 5.0)
#define FORCESMITH_APD_exp_decay(X)                                            \
  X(A, 4.0, 1.0, 20.0)                                                         \
  X(B, 1.0, 0.5, 5.0)
#define FORCESMITH_APD_mexp_decay(X)                                           \
  X(A, 0.1, 0.0, 10.0)                                                         \
  X(B, 0.1, 0.0, 10.0)                                                         \
  X(r0, 2.0, 0.0, 10.0)
#define FORCESMITH_APD_harmonic(X)                                             \
  X(k, 0.1, 0.0, 10.0)                                                         \
  X(r0, 2.0, 0.0, 8.0)
#define FORCESMITH_APD_universal(X)                                            \
  X(E0, 1.0, -10.0, 10.0)                                                      \
  X(a, 1.0, 0.0, 20.0)                                                         \
  X(b, 2.0, 0.0, 20.0)                                                         \
  X(c, 0.0, -1.0, 1.0)
#define FORCESMITH_APD_eopp(X)                                                 \
  X(A, 15.0, 1.0, 10000.0)                                                     \
  X(n, 6.0, 1.0, 20.0)                                                         \
  X(B, 5.0, -100.0, 100.0)                                                     \
  X(m, 3.0, 1.0, 10.0)                                                         \
  X(k, 2.5, 0.0, 6.0)                                                          \
  X(phi, 3.0, 0.0, 6.3)
#define FORCESMITH_APD_sqrt(X)                                                 \
  X(A, 0.1, 0.0, 10.0)                                                         \
  X(B, 2.0, 0.0, 10.0)
#define FORCESMITH_APD_const(X) X(C, 1.0, 0.0, 2.0)
#define FORCESMITH_APD_parabola(X)                                             \
  X(A, 1.0, -10.0, 10.0)                                                       \
  X(B, 1.0, -10.0, 10.0)                                                       \
  X(C, 1.0, -10.0, 10.0)
#define FORCESMITH_APD_softshell(X)                                            \
  X(A, 1.0, 0.1, 10.0)                                                         \
  X(n, 2.0, 1.0, 5.0)

// Smooth-cutoff (`_sc`) variants: the base parameters plus the appended
// switching width `h` (makeapot default 1 ∈ [0.5, 2]).
#define FORCESMITH_APD_SC_H(X) X(h, 1.0, 0.5, 2.0)
#define FORCESMITH_APD_lj_sc(X) FORCESMITH_APD_lj(X) FORCESMITH_APD_SC_H(X)
#define FORCESMITH_APD_morse_sc(X)                                             \
  FORCESMITH_APD_morse(X) FORCESMITH_APD_SC_H(X)
#define FORCESMITH_APD_exp_decay_sc(X)                                         \
  FORCESMITH_APD_exp_decay(X) FORCESMITH_APD_SC_H(X)
#define FORCESMITH_APD_eopp_sc(X) FORCESMITH_APD_eopp(X) FORCESMITH_APD_SC_H(X)

// The master list: F(registry_name, table_macro). registry_name is the
// canonical "type" string the reader/scaffolder use.
#define FORCESMITH_ANALYTIC_FUNCTIONS(F)                                       \
  F(lj, FORCESMITH_APD_lj)                                                     \
  F(morse, FORCESMITH_APD_morse)                                               \
  F(buckingham, FORCESMITH_APD_buckingham)                                     \
  F(born, FORCESMITH_APD_born)                                                 \
  F(power_decay, FORCESMITH_APD_power_decay)                                   \
  F(exp_decay, FORCESMITH_APD_exp_decay)                                       \
  F(mexp_decay, FORCESMITH_APD_mexp_decay)                                     \
  F(harmonic, FORCESMITH_APD_harmonic)                                         \
  F(universal, FORCESMITH_APD_universal)                                       \
  F(eopp, FORCESMITH_APD_eopp)                                                 \
  F(sqrt, FORCESMITH_APD_sqrt)                                                 \
  F(const, FORCESMITH_APD_const)                                               \
  F(parabola, FORCESMITH_APD_parabola)                                         \
  F(softshell, FORCESMITH_APD_softshell)                                       \
  F(lj_sc, FORCESMITH_APD_lj_sc)                                               \
  F(morse_sc, FORCESMITH_APD_morse_sc)                                         \
  F(exp_decay_sc, FORCESMITH_APD_exp_decay_sc)                                 \
  F(eopp_sc, FORCESMITH_APD_eopp_sc)

// ── Expansion helpers ───────────────────────────────────────────────────────

// Expand a table macro into a brace-init list of parameter-name strings, e.g.
//   FORCESMITH_PARAM_NAME_LIST(FORCESMITH_APD_lj)  →  std::vector<std::string>{
//   "epsilon", "sigma", }
// Use this in the reader registry so its names ARE these tokens, by definition.
#define FORCESMITH_APD_AS_NAME(tok, v, lo, hi)                                 \
  std::string(BOOST_PP_STRINGIZE(tok)),
#define FORCESMITH_PARAM_NAME_LIST(TABLE)                                      \
  std::vector<std::string> { TABLE(FORCESMITH_APD_AS_NAME) }

// Span of AnalyticParamDef for a function name; empty span if unknown. Lookup
// over the master list, so it covers exactly the functions defined above.
[[nodiscard]] std::span<const AnalyticParamDef>
analytic_defaults(std::string_view function);

// All function names with a default table (the master list), for tests that
// cross-check the macro names/order against the reader registry.
[[nodiscard]] std::span<const std::string_view> analytic_default_functions();

} // namespace forcesmith
