#pragma once

#include "forcesmith/core/radial_potential.hpp"

#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace forcesmith {

template <std::integral I> constexpr auto upper_triangle(I n) {
  return std::views::iota(I{0}, n) | std::views::transform([n](I ti) {
           return std::views::iota(ti, n) | std::views::transform([ti](I tj) {
                    return std::pair{ti, tj};
                  });
         }) |
         std::views::join;
}

template <std::integral I> constexpr auto strict_upper_triangle(I n) {
  return std::views::iota(I{0}, n) | std::views::transform([n](I ti) {
           return std::views::iota(ti + 1, n) |
                  std::views::transform(
                      [ti](I tj) { return std::pair{ti, tj}; });
         }) |
         std::views::join;
}

constexpr std::size_t pair_ordinal(std::size_t a, std::size_t b,
                                   std::size_t ntypes) noexcept {
  if (a > b) {
    std::swap(a, b);
  }
  return a * ntypes - a * (a - 1) / 2 + (b - a);
}

template <class A>
concept CTyped = !std::integral<A> && requires(const A &a) { a.type; };

template <typename T> class SymmetricMatrix {
  std::size_t ntypes_ = 0;
  std::vector<T> data_;

public:
  using value_type = T;

  constexpr void reserve(std::size_t ntypes) {
    ntypes_ = ntypes;
    data_.reserve(ntypes * (ntypes + 1) / 2);
  }
  template <typename U>
    requires std::constructible_from<T, U &&>
  constexpr void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
    grow();
  }
  constexpr void push_back(const T &t) {
    data_.push_back(t);
    grow();
  }
  constexpr void push_back(T &&t) {
    data_.push_back(std::move(t));
    grow();
  }

  constexpr const T &operator[](std::size_t ti, std::size_t tj) const {
    return data_[pair_ordinal(ti, tj, ntypes_)];
  }
  constexpr T &operator[](std::size_t ti, std::size_t tj) {
    return data_[pair_ordinal(ti, tj, ntypes_)];
  }
  template <CTyped A, CTyped B>
  constexpr const T &operator[](const A &a, const B &b) const {
    return (*this)[a.type, b.type];
  }
  template <CTyped A, CTyped B>
  constexpr T &operator[](const A &a, const B &b) {
    return (*this)[a.type, b.type];
  }

  constexpr std::size_t ntypes() const noexcept { return ntypes_; }
  constexpr std::size_t size() const noexcept { return data_.size(); }
  constexpr bool empty() const noexcept { return data_.empty(); }
  constexpr auto indices() const { return upper_triangle(ntypes_); }

  using iterator = T *;
  using const_iterator = const T *;
  constexpr iterator begin() noexcept { return data_.data(); }
  constexpr iterator end() noexcept { return data_.data() + data_.size(); }
  constexpr const_iterator begin() const noexcept { return data_.data(); }
  constexpr const_iterator end() const noexcept {
    return data_.data() + data_.size();
  }
  constexpr const_iterator cbegin() const noexcept { return begin(); }
  constexpr const_iterator cend() const noexcept { return end(); }

private:
  constexpr void grow() noexcept {
    if (ntypes_ * (ntypes_ + 1) / 2 < data_.size()) {
      ++ntypes_;
    }
  }
};

template <typename T> class TypeArray {
  std::vector<T> data_;

public:
  using value_type = T;
  using size_type = std::size_t;
  using iterator = T *;
  using const_iterator = const T *;

  TypeArray() = default;
  template <std::input_iterator It, std::sentinel_for<It> S>
  TypeArray(It first, S last) {
    if constexpr (std::sized_sentinel_for<S, It>) {
      data_.reserve(static_cast<std::size_t>(last - first));
    }
    for (; first != last; ++first) {
      data_.push_back(*first);
    }
  }

  constexpr void reserve(std::size_t ntypes) { data_.reserve(ntypes); }
  template <typename U>
    requires std::constructible_from<T, U &&>
  constexpr void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
  }
  constexpr void push_back(const T &t) { data_.push_back(t); }
  constexpr void push_back(T &&t) { data_.push_back(std::move(t)); }

  constexpr const T &operator[](std::size_t ti) const { return data_[ti]; }
  constexpr T &operator[](std::size_t ti) { return data_[ti]; }
  template <CTyped A> constexpr const T &operator[](const A &a) const {
    return data_[a.type];
  }
  template <CTyped A> constexpr T &operator[](const A &a) {
    return data_[a.type];
  }

  constexpr iterator begin() noexcept { return data_.data(); }
  constexpr iterator end() noexcept { return data_.data() + data_.size(); }
  constexpr const_iterator begin() const noexcept { return data_.data(); }
  constexpr const_iterator end() const noexcept {
    return data_.data() + data_.size();
  }
  constexpr const_iterator cbegin() const noexcept { return begin(); }
  constexpr const_iterator cend() const noexcept { return end(); }
  constexpr std::size_t size() const noexcept { return data_.size(); }
  constexpr bool empty() const noexcept { return data_.empty(); }
};

using RadialPotentialPair = SymmetricMatrix<RadialPotential>;
using RadialPotentialArray = TypeArray<RadialPotential>;

} // namespace forcesmith
