#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/force/force_calculator.hpp"

#include <boost/leaf/result.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace potfit {

struct CheckpointError {
  std::string message;
};

// Fluent builder for saving a checkpoint.
//
//   CheckpointWriter("run42")
//       .configs(cfg_vec)
//       .model(force_calculator)
//       .write();
//
// Files written:
// <prefix>.cfg.bin    — binary-archived vector<Configuration>
// <prefix>.model.json — native JSON force model (any family: pair, EAM, ADP,
//                       angular, tersoff, stiweb), via io::write_model

class CheckpointWriter {
public:
  explicit CheckpointWriter(std::filesystem::path prefix)
      : prefix_(std::move(prefix)) {}

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

// Builder for loading a checkpoint.
// CheckpointReader("run42").read(cfg_vec, model);

class CheckpointReader {
public:
  explicit CheckpointReader(std::filesystem::path prefix)
      : prefix_(std::move(prefix)) {}

  [[nodiscard]] boost::leaf::result<void>
  read(std::vector<Configuration> &configs, ForceCalculator &model) const;

private:
  std::filesystem::path prefix_;
};

} // namespace potfit
