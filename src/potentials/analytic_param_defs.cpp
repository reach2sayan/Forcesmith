#include "forcesmith/potentials/analytic_param_defs.hpp"

#include <array>

namespace forcesmith {

namespace {

// One AnalyticParamDef per row of a table macro.
#define FORCESMITH_APD_AS_DEF(tok, v, lo, hi)                                  \
  AnalyticParamDef{BOOST_PP_STRINGIZE(tok), (v), (lo), (hi)},

// One entry per function: its canonical "type" name and the static array of
// its parameter defaults — both generated from the single macro table, so they
// stay in lockstep with the reader registry's names.
struct Entry {
  std::string_view name;
  std::span<const AnalyticParamDef> defaults;
};

#define FORCESMITH_APD_DEFINE_ARRAY(fn, TABLE)                                 \
  inline constexpr std::array fn##_defs{TABLE(FORCESMITH_APD_AS_DEF)};

FORCESMITH_ANALYTIC_FUNCTIONS(FORCESMITH_APD_DEFINE_ARRAY)

#define FORCESMITH_APD_TABLE_ENTRY(fn, TABLE)                                  \
  Entry{BOOST_PP_STRINGIZE(fn), fn##_defs},

inline constexpr std::array kTable{
    FORCESMITH_ANALYTIC_FUNCTIONS(FORCESMITH_APD_TABLE_ENTRY)};
} // namespace

std::span<const AnalyticParamDef> analytic_defaults(std::string_view function) {
  for (const Entry &e : kTable) {
    if (e.name == function) {
      return e.defaults;
    }
  }
  return {};
}

std::span<const std::string_view> analytic_default_functions() {
#define FORCESMITH_APD_TABLE_NAME(fn, TABLE)                                   \
  std::string_view{BOOST_PP_STRINGIZE(fn)},
  static constexpr std::array kNames{
      FORCESMITH_ANALYTIC_FUNCTIONS(FORCESMITH_APD_TABLE_NAME)};
  return kNames;
}

} // namespace forcesmith
