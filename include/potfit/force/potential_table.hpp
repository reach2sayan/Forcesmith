#pragma once

#include "potfit/core/potential_base.hpp"

#include <vector>

namespace potfit {

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
};

// Convenience aliases for the common Potential case.
using PotentialPair = SymmetricMatrix<Potential>;
using PotentialArray = TypeArray<Potential>;

} // namespace potfit
