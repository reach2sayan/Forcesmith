#pragma once

#include "potfit/core/atom.hpp"
#include "potfit/core/potential_base.hpp"

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
//       .potentials(pot_vec)
//       .write();
//
// Files written:
// <prefix>.cfg.bin  — binary-archived vector<Configuration>
// <prefix>.pot      — native format-3 tabulated potential file

class CheckpointWriter {
public:
  explicit CheckpointWriter(std::filesystem::path prefix)
      : prefix_(std::move(prefix)) {}

  CheckpointWriter &configs(const std::vector<Configuration> &c) {
    configs_ = &c;
    return *this;
  }

  CheckpointWriter &potentials(const std::vector<Potential> &p) {
    pots_ = &p;
    return *this;
  }

  [[nodiscard]] boost::leaf::result<void> write() const;

private:
  std::filesystem::path prefix_;
  const std::vector<Configuration> *configs_ = nullptr;
  const std::vector<Potential> *pots_ = nullptr;
};

// Builder for loading a checkpoint.
// CheckpointReader("run42").read(cfg_vec, pot_vec);

class CheckpointReader {
public:
  explicit CheckpointReader(std::filesystem::path prefix)
      : prefix_(std::move(prefix)) {}

  [[nodiscard]] boost::leaf::result<void>
  read(std::vector<Configuration> &configs,
       std::vector<Potential> &potentials) const;

private:
  std::filesystem::path prefix_;
};

} // namespace potfit
