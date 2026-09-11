#pragma once

#include <memory>

namespace forcesmith::detail {

template <typename Concept> class ErasedValue {
protected:
  std::unique_ptr<Concept> self_;
  explicit ErasedValue(std::unique_ptr<Concept> p) noexcept
      : self_{std::move(p)} {}

public:
  ErasedValue(const ErasedValue &o) : self_{o.self_->clone()} {}
  ErasedValue(ErasedValue &&) noexcept = default;
  ErasedValue &operator=(const ErasedValue &o) {
    if (this != &o) {
      self_ = o.self_->clone();
    }
    return *this;
  }
  ErasedValue &operator=(ErasedValue &&) noexcept = default;
  ~ErasedValue() = default;
};

template <typename Concept> class ErasedMoveOnly {
protected:
  std::unique_ptr<Concept> self_;
  explicit ErasedMoveOnly(std::unique_ptr<Concept> p) noexcept
      : self_{std::move(p)} {}

public:
  ErasedMoveOnly(ErasedMoveOnly &&) noexcept = default;
  ErasedMoveOnly &operator=(ErasedMoveOnly &&) noexcept = default;
  ErasedMoveOnly(const ErasedMoveOnly &) = delete;
  ErasedMoveOnly &operator=(const ErasedMoveOnly &) = delete;
  ~ErasedMoveOnly() = default;
};

} // namespace forcesmith::detail
