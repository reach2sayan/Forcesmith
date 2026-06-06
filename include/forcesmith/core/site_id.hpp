#pragma once

#include <cstdint>

namespace forcesmith {

// Opaque handle to a fit-time evaluation-cache slot inside a RadialPotential.
//
// prepare_site(r) mints one (single-threaded);
// eval_at/deriv_at/eval_and_deriv_at consume it (read-only, safe under the
// parallel Jacobian). A default-constructed SiteId is "none" — cacheable() ==
// false — meaning the bond was not primed for this table (uncacheable/analytic
// potential, out of range at prime time, or no prepare() pass ran), so the
// caller must evaluate directly at r instead. Stored per radial-table role in
// NeighborEntry::sites.
//
// Zero-overhead: a single int32 with the -1 sentinel folded into the type, so
// it replaces the bare `int site` that used to carry this meaning implicitly.
class SiteId {
public:
  SiteId() = default;
  explicit constexpr SiteId(std::int32_t index) noexcept : index_(index) {}
  // True iff this refers to a real prepared cache slot (was `site >= 0`).
  [[nodiscard]] constexpr bool cacheable() const noexcept {
    return index_ >= 0;
  }
  // The underlying cache index; only meaningful when cacheable().
  [[nodiscard]] constexpr std::int32_t index() const noexcept { return index_; }
private:
  std::int32_t index_ = -1; // -1 == none / unprimed
};

} // namespace forcesmith
