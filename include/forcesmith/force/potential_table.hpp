#pragma once

#include "forcesmith/core/potential_base.hpp"

#include <boost/stl_interfaces/iterator_interface.hpp>

#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace forcesmith {

// (ti, tj) pairs with 0 <= ti <= tj < n, row-major
template <std::integral I> constexpr auto upper_triangle(I n) {
  return std::views::iota(I{0}, n) | std::views::transform([n](I ti) {
           return std::views::iota(ti, n) | std::views::transform([ti](I tj) {
                    return std::pair{ti, tj};
                  });
         }) |
         std::views::join;
}

// (ti, tj) pairs with 0 <= ti < tj < n — the strict (off-diagonal) upper
// triangle, i.e. the distinct unordered index pairs.
template <std::integral I> constexpr auto strict_upper_triangle(I n) {
  return std::views::iota(I{0}, n) | std::views::transform([n](I ti) {
           return std::views::iota(ti + 1, n) |
                  std::views::transform(
                      [ti](I tj) { return std::pair{ti, tj}; });
         }) |
         std::views::join;
}

template <typename T>
struct TypeArrayIterator
    : boost::stl_interfaces::iterator_interface<TypeArrayIterator<T>,
                                                std::random_access_iterator_tag,
                                                std::remove_const_t<T>, T &> {
  TypeArrayIterator() = default;
  explicit TypeArrayIterator(T *p) noexcept : ptr_(p) {}

private:
  constexpr T *&base_reference() noexcept { return ptr_; }
  constexpr T *const &base_reference() const noexcept { return ptr_; }
  friend boost::stl_interfaces::access;
  T *ptr_ = nullptr;
};

// Symmetric ntypes×ntypes upper-triangular matrix of T.
template <typename T> class SymmetricMatrix {
  std::size_t ntypes_ = 0;
  std::vector<T> data_;
  constexpr std::size_t slot(std::size_t a, std::size_t b) const noexcept {
    if (a > b) {
      std::swap(a, b);
    }
    return a * ntypes_ - a * (a - 1) / 2 + (b - a);
  }
  constexpr void sync_ntypes() noexcept {
    std::size_t n = 0;
    while (n * (n + 1) / 2 < data_.size()) {
      ++n;
    }
    ntypes_ = n;
  }

public:
  constexpr void reserve(std::size_t ntypes) {
    ntypes_ = ntypes;
    data_.reserve(ntypes * (ntypes + 1) / 2);
  }
  template <typename U> constexpr void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
    sync_ntypes();
  }
  using value_type = T;
  constexpr void push_back(const T &t) {
    data_.push_back(t);
    sync_ntypes();
  }
  constexpr void push_back(T &&t) {
    data_.push_back(std::move(t));
    sync_ntypes();
  }
  constexpr const T &operator[](std::size_t ti, std::size_t tj) const {
    return data_[slot(ti, tj)];
  }
  constexpr T &operator[](std::size_t ti, std::size_t tj) {
    return data_[slot(ti, tj)];
  }
  constexpr std::size_t ntypes() const noexcept { return ntypes_; }
  constexpr std::size_t size() const noexcept { return data_.size(); }
  constexpr auto indices() const { return upper_triangle(ntypes_); }

  using iterator = TypeArrayIterator<T>;
  using const_iterator = TypeArrayIterator<const T>;
  iterator begin() noexcept { return iterator(data_.data()); }
  iterator end() noexcept { return iterator(data_.data() + data_.size()); }
  const_iterator begin() const noexcept { return const_iterator(data_.data()); }
  const_iterator end() const noexcept {
    return const_iterator(data_.data() + data_.size());
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }
  template <typename A, typename B>
    requires requires(A a, B b) {
      a.type;
      b.type;
    }
  constexpr const T &operator[](const A &a, const B &b) const {
    return (*this)[a.type, b.type];
  }
  template <typename A, typename B>
    requires requires(A a, B b) {
      a.type;
      b.type;
    }
  constexpr T &operator[](const A &a, const B &b) {
    return (*this)[a.type, b.type];
  }
};

// Linear array of T, one per element type.
template <typename T> class TypeArray {
  std::vector<T> data_;

public:
  using iterator = TypeArrayIterator<T>;
  using const_iterator = TypeArrayIterator<const T>;
  using value_type = T;
  using size_type = std::size_t;

  TypeArray() = default;
  template <std::input_iterator It, std::sentinel_for<It> S>
  TypeArray(It first, S last) {
    for (; first != last; ++first)
      data_.push_back(*first);
  }

  constexpr void reserve(std::size_t ntypes) { data_.reserve(ntypes); }
  template <typename U> constexpr void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
  }

  constexpr void push_back(const T &t) { data_.push_back(t); }
  constexpr void push_back(T &&t) { data_.push_back(std::move(t)); }

  constexpr const T &operator[](std::size_t ti) const { return data_[ti]; }
  constexpr T &operator[](std::size_t ti) { return data_[ti]; }
  template <typename A>
    requires(!std::integral<A>) && requires(A a) { a.type; }
  constexpr const T &operator[](const A &a) const {
    return data_[a.type];
  }
  template <typename A>
    requires(!std::integral<A>) && requires(A a) { a.type; }
  constexpr T &operator[](const A &a) {
    return data_[a.type];
  }

  iterator begin() noexcept { return iterator(data_.data()); }
  iterator end() noexcept { return iterator(data_.data() + data_.size()); }
  const_iterator begin() const noexcept { return const_iterator(data_.data()); }
  const_iterator end() const noexcept {
    return const_iterator(data_.data() + data_.size());
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }
  std::size_t size() const noexcept { return data_.size(); }
  bool empty() const noexcept { return data_.empty(); }
};

using PotentialPair = SymmetricMatrix<Potential>;
using PotentialArray = TypeArray<Potential>;

} // namespace forcesmith
