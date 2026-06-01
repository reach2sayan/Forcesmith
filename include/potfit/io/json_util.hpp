#pragma once

// Combinator that lifts a throwing nlohmann::json call into a leaf::result,
// replacing the repeated `try { parse } catch (parse_error)` + `try { body }
// catch (json::exception)` idiom used across the io readers.

#include "potfit/io/config_reader.hpp" // for io::ParseError

#include <boost/leaf/result.hpp>
#include <nlohmann/json.hpp>

#include <type_traits>
#include <utility>

namespace potfit::io {
namespace detail {
template <class T> struct as_result {
  using type = boost::leaf::result<T>;
};
template <class T> struct as_result<boost::leaf::result<T>> {
  using type = boost::leaf::result<T>;
};
} // namespace detail

// Run `f`; translate any nlohmann json exception (parse_error derives from
// json::exception) into a ParseError leaf error. Flattens: if f() already
// returns leaf::result<T>, that type is returned unchanged; otherwise the plain
// T is wrapped. BOOST_LEAF_AUTO inside the callable therefore preserves its
// early-error semantics.
template <class F>
[[nodiscard]] auto catch_json(F &&f) ->
    typename detail::as_result<std::invoke_result_t<F &&>>::type {
  try {
    return std::forward<F>(f)();
  } catch (const nlohmann::json::exception &e) {
    return boost::leaf::new_error(ParseError{e.what(), 0});
  }
}
} // namespace potfit::io
