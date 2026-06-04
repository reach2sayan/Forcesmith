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

#include <Eigen/Core>

#include <boost/container/static_vector.hpp>

#include <algorithm>
#include <cstddef>

namespace potfit {

// Upper-triangle ordinal of the unordered species pair {a,b} (row-major,
// 0 <= lo <= hi < S) — the companion lookup to upper_triangle() in
// potential_table.hpp, matching SOAP's species-pair ordering.
inline std::size_t pair_ordinal(std::size_t a, std::size_t b, std::size_t S) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return lo * S - lo * (lo - 1) / 2 + (hi - lo);
}

// Sequence of contiguous descriptor blocks. add() appends a block of `count`
// values over `nchan` channels, prefix-summing the base offset, and returns the
// block's id; index() maps (block, channel, t) to its flat position. Blocks live
// in inline storage (a descriptor has only a handful: ACSF 5, LMBTR <= 2), so a
// layout costs no heap allocation.
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

  [[nodiscard]] Eigen::Index size() const {
    return static_cast<Eigen::Index>(size_);
  }
  [[nodiscard]] Eigen::Index index(std::size_t block, std::size_t chan,
                                   std::size_t t) const {
    const Block &b = blocks_[block];
    return static_cast<Eigen::Index>(b.base + chan * b.count + t);
  }
};

} // namespace potfit
