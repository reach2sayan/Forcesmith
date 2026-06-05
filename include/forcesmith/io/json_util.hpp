#pragma once

#include "forcesmith/io/config_reader.hpp" // for io::ParseError

#include <boost/leaf/result.hpp>
#include <nlohmann/json.hpp>

#include <type_traits>
#include <utility>

namespace forcesmith::io {
namespace detail {
template <class T> struct as_result {
  using type = boost::leaf::result<T>;
};
template <class T> struct as_result<boost::leaf::result<T>> {
  using type = boost::leaf::result<T>;
};
} // namespace detail

template <class F>
[[nodiscard]] auto catch_json(F &&f) ->
    typename detail::as_result<std::invoke_result_t<F &&>>::type {
  try {
    return std::forward<F>(f)();
  } catch (const nlohmann::json::exception &e) {
    return boost::leaf::new_error(ParseError{e.what(), 0});
  }
}
} // namespace forcesmith::io
