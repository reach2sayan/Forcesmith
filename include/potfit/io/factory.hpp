#pragma once

#include "potfit/io/config_reader.hpp" // for io::ParseError
#include <boost/container/flat_map.hpp>
#include <boost/leaf/result.hpp>
#include <string>
#include <utility>

namespace potfit::io {

template <class IdentifierType, class AbstractProduct>
struct DefaultFactoryError {
  static boost::leaf::result<AbstractProduct>
  OnUnknownType(const IdentifierType &id) {
    return boost::leaf::new_error(
        ParseError{"factory: unknown identifier '" + std::string(id) + "'", 0});
  }
};

template <class AbstractProduct, class IdentifierType, class ProductCreator,
          template <class, class> class FactoryErrorPolicy =
              DefaultFactoryError>
class PotfitFactory : FactoryErrorPolicy<IdentifierType, AbstractProduct> {
public:
  using FactoryErrorPolicy<IdentifierType, AbstractProduct>::OnUnknownType;
  bool Register(IdentifierType id, ProductCreator creator) {
    return associations_.try_emplace(std::move(id), std::move(creator)).second;
  }
  bool Unregister(const IdentifierType &id) {
    return associations_.erase(id) > 0;
  }
  [[nodiscard]] bool IsRegistered(const IdentifierType &id) const {
    return associations_.contains(id);
  }

  template <class... Args>
  boost::leaf::result<AbstractProduct> CreateObject(const IdentifierType &id,
                                                    Args &&...args) const {
    if (auto it = associations_.find(id); it != associations_.end()) {
      return it->second(std::forward<Args>(args)...);
    }
    return OnUnknownType(id);
  }

private:
  boost::container::flat_map<IdentifierType, ProductCreator> associations_;
};

} // namespace potfit::io
