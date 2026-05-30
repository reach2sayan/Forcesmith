#pragma once

#include "potfit/core/potential_base.hpp"

#include <vector>

namespace potfit {

// Symmetric ntypes×ntypes upper-triangular matrix of T.
// Stores only the ntypes*(ntypes+1)/2 unique entries.
// Access via operator()(ti, tj) — argument order does not matter.
template <typename T>
class SymmetricMatrix {
  int ntypes_ = 0;
  std::vector<T> data_;

  int slot(int a, int b) const noexcept {
    if (a > b) std::swap(a, b);
    return a * ntypes_ - a * (a - 1) / 2 + (b - a);
  }

public:
  // Reserve capacity for ntypes element types. Populate with emplace_back
  // in slot order (00, 01, 11, 02, 12, 22, …) before calling operator().
  void reserve(int ntypes) {
    ntypes_ = ntypes;
    data_.reserve(ntypes * (ntypes + 1) / 2);
  }

  template <typename U>
  void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
  }

  const T &operator()(int ti, int tj) const { return data_[slot(ti, tj)]; }
  T &      operator()(int ti, int tj)       { return data_[slot(ti, tj)]; }
};

// Linear array of T, one per element type.
// Access via operator()(ti).
template <typename T>
class TypeArray {
  std::vector<T> data_;

public:
  // Reserve capacity for ntypes element types. Populate with emplace_back
  // in type-index order before calling operator().
  void reserve(int ntypes) { data_.reserve(ntypes); }

  template <typename U>
  void emplace_back(U &&u) {
    data_.emplace_back(std::forward<U>(u));
  }

  const T &operator()(int ti) const { return data_[ti]; }
  T &      operator()(int ti)       { return data_[ti]; }
};

// Convenience aliases for the common Potential case.
using PotentialPairMatrix = SymmetricMatrix<Potential>;
using PotentialTypeArray  = TypeArray<Potential>;

} // namespace potfit
