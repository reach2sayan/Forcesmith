#pragma once

#include "potfit/core/potential_base.hpp"

#include <boost/stl_interfaces/iterator_interface.hpp>

#include <cstddef>
#include <vector>

namespace potfit {

// Random-access iterator over the contiguous storage of TypeArray<T>.
// boost::stl_interfaces::iterator_interface derives all iterator operations
// from the three primitives: operator*, operator+=, and operator- via
// base_reference().
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
// Stores only the ntypes*(ntypes+1)/2 unique entries.
// Access via operator()(ti, tj) — argument order does not matter.
template <typename T> class SymmetricMatrix {
  int ntypes_ = 0;
  std::vector<T> data_;
  constexpr int slot(int a, int b) const noexcept {
    if (a > b) {
      std::swap(a, b);
    }
    return a * ntypes_ - a * (a - 1) / 2 + (b - a);
  }

public:
  // Reserve capacity for ntypes element types. Populate with emplace_back
  // in slot order (00, 01, 11, 02, 12, 22, …) before calling operator().
  constexpr void reserve(int ntypes) {
    ntypes_ = ntypes;
    data_.reserve(ntypes * (ntypes + 1) / 2);
  }
  template <typename U> constexpr void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
  }
  constexpr const T &operator[](int ti, int tj) const {
    return data_[slot(ti, tj)];
  }
  constexpr T &operator[](int ti, int tj) { return data_[slot(ti, tj)]; }
  constexpr int         ntypes() const noexcept { return ntypes_; }
  constexpr std::size_t size()   const noexcept { return data_.size(); }

  auto begin()        { return data_.begin(); }
  auto end()          { return data_.end(); }
  auto begin()  const { return data_.begin(); }
  auto end()    const { return data_.end(); }
  auto cbegin() const { return data_.cbegin(); }
  auto cend()   const { return data_.cend(); }
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

  constexpr void reserve(int ntypes) { data_.reserve(ntypes); }
  template <typename U> constexpr void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
  }

  constexpr const T &operator[](int ti) const { return data_[ti]; }
  constexpr T &operator[](int ti) { return data_[ti]; }
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

// Convenience aliases for the common Potential case.
using PotentialPair = SymmetricMatrix<Potential>;
using PotentialArray = TypeArray<Potential>;

} // namespace potfit
