#pragma once

#include <boost/signals2/signal.hpp>
#include <cstdint>

namespace potfit::events {

struct IterationStats {
  std::uint64_t iteration = 0;
  double objective = 0.0;
  double grad_norm = 0.0;
};

struct ForceEvalStats {
  std::uint64_t conf_index = 0;
  double rms_force = 0.0;
};

inline boost::signals2::signal<void(const IterationStats &)> on_iteration;
inline boost::signals2::signal<void(const ForceEvalStats &)> on_force_eval;
inline boost::signals2::signal<void()> on_output;

} // namespace potfit::events
