#pragma once

#include <boost/leaf/result.hpp>
#include <boost/optional.hpp>
#include <string_view>

namespace potfit::elements {

struct Element {
  const std::string_view symbol;
  const int Z;
  const double mass_amu;
};

[[nodiscard]] boost::optional<const Element &>
find_by_symbol(std::string_view symbol) noexcept;

[[nodiscard]] boost::optional<const Element &> find_by_Z(int Z) noexcept;

[[nodiscard]] boost::leaf::result<Element> lookup(std::string_view symbol);

[[nodiscard]] boost::leaf::result<int> atomic_number(std::string_view symbol);

[[nodiscard]] boost::leaf::result<double> atomic_mass(std::string_view symbol);

} // namespace potfit::elements
