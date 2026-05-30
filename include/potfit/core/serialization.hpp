#pragma once

// Serialization infrastructure:
//
//   struct Foo : Serializable<Foo> { int x; double y; };
//
//   template<> struct Serializer<Foo> {
//       template<class Archive>
//       static void apply(Archive& ar, Foo& f, unsigned int) {
//           ar & f.x & f.y;
//       }
//   };
//
// Serializer<T> must be specialized before any archive reads/writes Foo.
// All fields accessed by the specialization must be public.

#include <boost/serialization/access.hpp>

namespace potfit {

template <typename T>
struct Serializer; // must be explicitly specialized for each serializable type

// CRTP mixin.  Inherit as:  struct Foo : Serializable<Foo> { ... };
template <typename Derived> class Serializable {
  friend class boost::serialization::access;
  template <class Archive>
  constexpr void serialize(Archive &ar, unsigned int version) {
    Serializer<Derived>::apply(ar, static_cast<Derived &>(*this), version);
  }
};

} // namespace potfit
