#pragma once

#include "forcesmith/io/config_reader.hpp" // ParseError

#include <boost/leaf/error.hpp>
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

namespace forcesmith::io {

namespace bp = boost::parser;

namespace detail {
struct LineCol {
  std::size_t line;
  std::size_t column;
};

[[nodiscard]] inline LineCol line_col_of(std::string_view text,
                                         std::size_t offset) {
  offset = std::min(offset, text.size());
  const std::size_t line = 1 + static_cast<std::size_t>(std::ranges::count(
                                   text.substr(0, offset), '\n'));
  const std::size_t bol = text.rfind('\n', offset == 0 ? 0 : offset - 1);
  const std::size_t col = (bol == std::string_view::npos || offset == 0)
                              ? offset
                              : offset - bol - 1;
  return {line, col + 1};
}
} // namespace detail

template <class Parser>
[[nodiscard]] auto parse_or_error(const Parser &rule, std::string_view text,
                                  std::string_view what) {
  using Attr = decltype(*bp::parse(text, rule, bp::ws));
  using Result = boost::leaf::result<std::remove_cvref_t<Attr>>;

  auto first = text.begin();
  const auto last = text.end();
  if (auto attr = bp::prefix_parse(first, last, rule, bp::ws)) {
    if (first == last) {
      return Result{std::move(*attr)};
    }
  } else {
    first = text.begin();
  }
  const auto pos =
      detail::line_col_of(text, static_cast<std::size_t>(first - text.begin()));
  return Result{boost::leaf::new_error(ParseError{
      std::string(what) + ": cannot parse '" + std::string(text) +
          "' at offset " +
          std::to_string(static_cast<std::size_t>(first - text.begin())),
      pos.line, pos.column})};
}

} // namespace forcesmith::io
