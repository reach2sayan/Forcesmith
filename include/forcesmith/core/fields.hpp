#pragma once

#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace forcesmith {

template <class Model, class Table> struct TableField {
  Table Model::*pointer;
  std::string_view name;
  bool radial = true;
};

template <class Model, class Table>
TableField(Table Model::*, std::string_view) -> TableField<Model, Table>;
template <class Model, class Table>
TableField(Table Model::*, std::string_view, bool) -> TableField<Model, Table>;

template <class M>
concept CTabulated = requires { std::remove_cvref_t<M>::tables; };

template <class Model> constexpr auto table_fields() {
  if constexpr (CTabulated<Model>) {
    return std::remove_cvref_t<Model>::tables;
  } else {
    return std::tuple{};
  }
}

template <class Model, class F>
constexpr void for_each_table(Model &model, F &&f) {
  std::apply([&](auto... field) { (f(model.*field.pointer, field.name), ...); },
             table_fields<Model>());
}

template <class Model, class F>
constexpr void for_each_table_field(Model &model, F &&f) {
  std::apply([&](auto... field) { (f(model.*field.pointer, field), ...); },
             table_fields<Model>());
}

template <class Model>
inline constexpr std::size_t table_count =
    std::tuple_size_v<decltype(table_fields<Model>())>;

} // namespace forcesmith
