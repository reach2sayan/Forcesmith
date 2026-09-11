#pragma once

// By-name ddx symbol lookup: indexing by position would silently re-bind if a
// symbol were renamed or added.

#include "ddx.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <type_traits>

#include <concepts>
#include <cstddef>
#include <string_view>

namespace forcesmith::symbolic {

template <class S>
concept CSymbol = requires {
  { S::name.data() } -> std::convertible_to<const char *>;
  { S::name.size() } -> std::convertible_to<std::size_t>;
};

template <class Eq>
concept CEquation = requires { typename std::remove_cvref_t<Eq>::symbols; };

template <CSymbol S> consteval std::string_view name_of() {
  return std::string_view(S::name.data(), S::name.size());
}

inline constexpr std::size_t npos = static_cast<std::size_t>(-1);

template <class Symbols> consteval std::size_t index_in(std::string_view want) {
  std::size_t i = 0;
  std::size_t found = npos;
  boost::mp11::mp_for_each<
      boost::mp11::mp_transform<boost::mp11::mp_identity, Symbols>>(
      [&](auto tag) {
        if (name_of<typename decltype(tag)::type>() == want) {
          found = i;
        }
        ++i;
      });
  return found;
}

template <CEquation Eq> consteval std::size_t slot_of(std::string_view want) {
  return index_in<typename std::remove_cvref_t<Eq>::symbols>(want);
}

template <ddx::impl::FixedString Want, CEquation Eq>
consteval auto derivative_of(const Eq &eq) {
  constexpr std::size_t k = slot_of<Eq>(Want.view());
  static_assert(k != npos, "differentiating by a symbol the equation lacks");
  return ddx::Equation{eq[ddx::idx<k + 1>()]};
}

} // namespace forcesmith::symbolic
