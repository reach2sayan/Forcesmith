#pragma once

#include "forcesmith/core/types.hpp"
#include "forcesmith/force/potential_table.hpp" // pair_ordinal
#include <Eigen/Core>
#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <optional>
#include <vector>

namespace forcesmith {

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
    blocks_.push_back(Block{.base = size_, .count = count, .nchan = nchan});
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

std::vector<std::optional<Eigen::Index>>
remap_layout(const DescriptorLayout &old_L, const DescriptorLayout &new_L,
             const std::vector<std::optional<std::size_t>> &old_of_new,
             std::size_t S_old, std::size_t S_new);

} // namespace forcesmith
