#pragma once

#include <span>
#include <string_view>

namespace forcesmith {

struct AnalyticParamDef {
  std::string_view name;
  double value;
  double min;
  double max;
};

[[nodiscard]] std::span<const AnalyticParamDef>
analytic_defaults(std::string_view function);

[[nodiscard]] std::span<const std::string_view> analytic_default_functions();

} // namespace forcesmith
