#pragma once

#include "forcesmith/core/param.hpp"
#include "forcesmith/core/types.hpp"
#include "forcesmith/io/config_reader.hpp" // for io::ParseError

#include <boost/describe/members.hpp>
#include <boost/describe/modifiers.hpp>
#include <boost/leaf/result.hpp>
#include <boost/mp11/algorithm.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>
#include <concepts>
#include <type_traits>
#include <utility>

namespace forcesmith {

template <class T>
concept CDescribed = boost::describe::has_describe_members<T>::value;

namespace detail {
template <class T>
using described_public_members =
    boost::describe::describe_members<T, boost::describe::mod_public>;
} // namespace detail

template <class T>
inline constexpr std::array<std::string_view, 0> optional_members{};

} // namespace forcesmith

namespace nlohmann {

template <class T>
struct adl_serializer<T, std::enable_if_t<forcesmith::CDescribed<T>>> {
  static void to_json(json &j, const T &v) {
    j = json::object();
    boost::mp11::mp_for_each<forcesmith::detail::described_public_members<T>>(
        [&](auto D) { j[D.name] = v.*D.pointer; });
  }
  static void from_json(const json &j, T &v) {
    boost::mp11::mp_for_each<forcesmith::detail::described_public_members<T>>(
        [&](auto D) {
          const bool optional = std::ranges::contains(
              forcesmith::optional_members<T>, std::string_view(D.name));
          if (optional && !j.contains(D.name)) {
            return;
          }
          j.at(D.name).get_to(v.*D.pointer);
        });
  }
};

template <> struct adl_serializer<forcesmith::Param> {
  static void to_json(json &j, const forcesmith::Param &p) { j = p.value; }
  static void from_json(const json &j, forcesmith::Param &p) {
    if (j.is_object()) {
      p.value = j.at("value").get<double>();
      p.min = j.value("min", p.min);
      p.max = j.value("max", p.max);
      p.fixed = j.value("fixed", false);
    } else {
      p.value = j.get<double>();
    }
  }
};

template <> struct adl_serializer<forcesmith::Vec3> {
  static void to_json(json &j, const forcesmith::Vec3 &v) {
    j = json::array({v[0], v[1], v[2]});
  }
  static void from_json(const json &j, forcesmith::Vec3 &v) {
    for (int i = 0; i < 3; ++i) {
      v[i] = j.at(static_cast<std::size_t>(i)).get<double>();
    }
  }
};

template <> struct adl_serializer<forcesmith::Mat3> {
  static void to_json(json &j, const forcesmith::Mat3 &m) {
    j = json::array();
    for (int r = 0; r < 3; ++r) {
      j.push_back(json::array({m(r, 0), m(r, 1), m(r, 2)}));
    }
  }
  static void from_json(const json &j, forcesmith::Mat3 &m) {
    for (int r = 0; r < 3; ++r) {
      const json &row = j.at(static_cast<std::size_t>(r));
      for (int c = 0; c < 3; ++c) {
        m(r, c) = row.at(static_cast<std::size_t>(c)).get<double>();
      }
    }
  }
};

} // namespace nlohmann

namespace forcesmith::io {
namespace detail {
template <class T> struct as_result {
  using type = boost::leaf::result<T>;
};
template <class T> struct as_result<boost::leaf::result<T>> {
  using type = boost::leaf::result<T>;
};
} // namespace detail

template <std::invocable F>
[[nodiscard]] auto catch_json(F &&f) ->
    typename detail::as_result<std::invoke_result_t<F &&>>::type {
  try {
    return std::forward<F>(f)();
  } catch (const nlohmann::json::exception &e) {
    return boost::leaf::new_error(ParseError{e.what(), 0});
  }
}
} // namespace forcesmith::io
