#pragma once
#include <boost/serialization/access.hpp>

namespace forcesmith {

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

} // namespace forcesmith
