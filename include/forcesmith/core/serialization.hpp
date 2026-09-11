#pragma once

#include "forcesmith/core/types.hpp"

#include <boost/describe/members.hpp>
#include <boost/describe/modifiers.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/serialization/access.hpp>

#include <concepts>

namespace forcesmith {

template <class A>
concept CArchive = requires(A &ar, double &x) {
  ar & x;
  { A::is_loading::value } -> std::convertible_to<bool>;
};

template <class T>
concept CSerializableDescribed =
    boost::describe::has_describe_members<T>::value;

template <typename T> struct Serializer {
  static void apply(CArchive auto &ar, T &v, unsigned int)
    requires CSerializableDescribed<T>
  {
    boost::mp11::mp_for_each<
        boost::describe::describe_members<T, boost::describe::mod_public>>(
        [&](auto D) { ar &(v.*D.pointer); });
  }
};

template <> struct Serializer<Vec3> {
  static void apply(CArchive auto &ar, Vec3 &v, unsigned int) {
    ar &v(0) & v(1) & v(2);
  }
};

template <> struct Serializer<Mat3> {
  static void apply(CArchive auto &ar, Mat3 &m, unsigned int) {
    for (auto [i, j] : tensor3D_indices) {
      ar &m(i, j);
    }
  }
};

template <typename Derived> class Serializable {
  friend class boost::serialization::access;
  constexpr void serialize(CArchive auto &ar, unsigned int version) {
    Serializer<Derived>::apply(ar, static_cast<Derived &>(*this), version);
  }
};

} // namespace forcesmith
