#pragma once

#include "forcesmith/core/atom.hpp"
#include "forcesmith/force/force_calculator.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace forcesmith {

struct CheckpointError {
  std::string message;
};

class CheckpointWriter {
public:
  explicit CheckpointWriter(std::filesystem::path prefix)
      : prefix_{std::move(prefix)} {}
  CheckpointWriter &configs(const std::vector<Configuration> &c) {
    configs_ = &c;
    return *this;
  }
  CheckpointWriter &model(const ForceCalculator &m) {
    model_ = &m;
    return *this;
  }
  [[nodiscard]] boost::leaf::result<void> write() const;

private:
  std::filesystem::path prefix_;
  const std::vector<Configuration> *configs_ = nullptr;
  const ForceCalculator *model_ = nullptr;
};

class CheckpointReader {
public:
  explicit CheckpointReader(std::filesystem::path prefix)
      : prefix_{std::move(prefix)} {}
  [[nodiscard]] boost::leaf::result<void>
  read(std::vector<Configuration> &configs, ForceCalculator &model) const;

private:
  std::filesystem::path prefix_;
};

} // namespace forcesmith
