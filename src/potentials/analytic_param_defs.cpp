#include "forcesmith/potentials/analytic_param_defs.hpp"

#include "forcesmith/potentials/analytic_potential.hpp"

#include <boost/mp11/algorithm.hpp>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace forcesmith {

namespace {

template <CAnalyticForm T> const std::vector<AnalyticParamDef> &defs_of() {
  static const std::vector<AnalyticParamDef> v = [] {
    std::vector<AnalyticParamDef> out;
    out.reserve(T::num_params);
    for (std::size_t i = 0; i < T::num_params; ++i) {
      out.push_back(AnalyticParamDef{T::param_names[i], T::defaults[i].value,
                                     T::defaults[i].min, T::defaults[i].max});
    }
    return out;
  }();
  return v;
}

struct Row {
  std::string_view name; // canonical name or alias
  std::span<const AnalyticParamDef> defaults;
};

const std::vector<Row> &table() {
  static const std::vector<Row> rows = [] {
    std::vector<Row> out;
    boost::mp11::mp_for_each<
        boost::mp11::mp_transform<boost::mp11::mp_identity, AnalyticForms>>(
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          for (std::string_view n : T::names) {
            if (!n.empty()) {
              out.push_back(Row{n, defs_of<T>()});
            }
          }
        });
    return out;
  }();
  return rows;
}

} // namespace

std::span<const AnalyticParamDef> analytic_defaults(std::string_view function) {
  const auto &tab = table();
  auto iter = std::ranges::find(tab, function, &Row::name);
  return iter == tab.end() ? std::span<const AnalyticParamDef>{}
                           : iter->defaults;
}

std::span<const std::string_view> analytic_default_functions() {
  static const std::vector<std::string_view> names = [] {
    std::vector<std::string_view> out;
    boost::mp11::mp_for_each<
        boost::mp11::mp_transform<boost::mp11::mp_identity, AnalyticForms>>(
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          out.push_back(T::names.front()); // the canonical name
        });
    return out;
  }();
  return names;
}

} // namespace forcesmith
