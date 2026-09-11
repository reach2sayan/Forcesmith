#pragma once

#include "forcesmith/core/strong.hpp"

#include <cstdint>

namespace forcesmith {

// Cache-slot handle: minted by prepare_site (single-threaded), read by the _at
// family (safe in the parallel Jacobian). Default = none: evaluate at r.
struct SiteIdTag;

struct SiteId : Strong<std::int32_t, SiteIdTag, false, -1> {
  using Base = Strong<std::int32_t, SiteIdTag, false, -1>;
  using Base::Base;
  [[nodiscard]] constexpr bool cacheable() const noexcept {
    return index() >= 0;
  }
};

} // namespace forcesmith
