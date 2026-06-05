#pragma once

// Shared flat-layout machinery for local ML descriptors (ACSF, LMBTR, ...).
//
// A per-atom descriptor is a flat vector partitioned into contiguous *blocks*.
// Each block channels `count` values over `nchan` channels — `nchan = ntypes`
// for per-species blocks, `nchan = P = ntypes(ntypes+1)/2` for per-species-pair
// blocks — and a single component lives at `base + chan*count + t`. Every
// descriptor differs only in which blocks it pushes and how it names them, so
// the prefix-sum/index arithmetic lives here once and the concrete descriptors
// wrap it (AcsfLayout, LmbtrLayout) to keep their domain vocabulary.

#include "forcesmith/core/types.hpp"

#include <Eigen/Core>

#include <boost/container/static_vector.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace forcesmith {

// Upper-triangle ordinal of the unordered species pair {a,b} (row-major,
// 0 <= lo <= hi < S) — the companion lookup to upper_triangle() in
// potential_table.hpp, matching SOAP's species-pair ordering.
constexpr FORCE_INLINE std::size_t pair_ordinal(std::size_t a, std::size_t b,
                                                std::size_t S) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return lo * S - lo * (lo - 1) / 2 + (hi - lo);
}

// Sequence of contiguous descriptor blocks. add() appends a block of `count`
// values over `nchan` channels, prefix-summing the base offset, and returns the
// block's id; index() maps (block, channel, t) to its flat position. Blocks
// live in inline storage (a descriptor has only a handful: ACSF 5, LMBTR <= 2),
// so a layout costs no heap allocation.
struct DescriptorLayout {
  struct Block {
    std::size_t base = 0;  // first flat index of the block
    std::size_t count = 0; // values per channel
    std::size_t nchan = 0; // channels (S per-species, P per-pair)
  };
  static constexpr std::size_t max_blocks = 8;
  boost::container::static_vector<Block, max_blocks> blocks_;
  std::size_t size_ = 0;

  std::size_t add(std::size_t count, std::size_t nchan) {
    blocks_.push_back(Block{size_, count, nchan});
    size_ += count * nchan;
    return blocks_.size() - 1;
  }

  [[nodiscard]] constexpr Eigen::Index size() const {
    return static_cast<Eigen::Index>(size_);
  }
  [[nodiscard]] Eigen::Index index(std::size_t block, std::size_t chan,
                                   std::size_t t) const {
    const Block &b = blocks_[block];
    return static_cast<Eigen::Index>(b.base + chan * b.count + t);
  }
};

// Re-rank index map for any DescriptorLayout-based descriptor: for each flat
// index of the NEW layout, the flat index of the OLD layout that feeds it, or
// nullopt for a brand-new block (a channel/pair touching an added species).
// `old_of_new[s]` is the OLD compact slot of the species now at NEW slot s (or
// nullopt if that element did not exist before). old_L and new_L must hold the
// SAME blocks in the SAME order (same descriptor config, only S differs), which
// holds because the hyperparameters are unchanged across a re-rank. A block is
// per-species when nchan==S and per-pair when nchan==P=S(S+1)/2 (distinct for
// S>=2; at S==1 both collapse to the single channel and the per-species path is
// correct either way).
std::vector<std::optional<Eigen::Index>>
remap_layout(const DescriptorLayout &old_L, const DescriptorLayout &new_L,
             const std::vector<std::optional<std::size_t>> &old_of_new,
             std::size_t S_old, std::size_t S_new);

} // namespace forcesmith
