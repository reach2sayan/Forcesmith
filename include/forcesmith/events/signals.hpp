#pragma once

#include <atomic>
#include <boost/signals2/signal.hpp>
#include <cstdint>
#include <utility>

namespace forcesmith {
struct Configuration;
}

namespace forcesmith::events {

struct IterationStats {
  std::uint64_t iteration = 0;
  double objective = 0.0;
  double grad_norm =
      0.0; // ‖Jᵀf‖; lags one residual eval (see ForcesmithFunctor)
};

struct ForceEvalStats {
  std::uint64_t conf_index = 0;
  double rms_force = 0.0;
  const Configuration &cfg;
};

struct SuppressibleSignal {
  std::atomic<bool> suppressed{false};
  void operator()(const ForceEvalStats &s) const {
    if (!suppressed.load(std::memory_order_relaxed)) {
      sig(s);
    }
  }
  auto connect(auto &&f) { return sig.connect(std::forward<decltype(f)>(f)); }

private:
  boost::signals2::signal<void(const ForceEvalStats &)> sig;
};

inline boost::signals2::signal<void(const IterationStats &)> on_iteration;
inline SuppressibleSignal on_force_eval;
inline boost::signals2::signal<void()> on_output;

struct ScopedForceEvalSuppress {
  ScopedForceEvalSuppress() {
    on_force_eval.suppressed.store(true, std::memory_order_relaxed);
  }
  ~ScopedForceEvalSuppress() {
    on_force_eval.suppressed.store(false, std::memory_order_relaxed);
  }
  ScopedForceEvalSuppress(const ScopedForceEvalSuppress &) = delete;
  ScopedForceEvalSuppress &operator=(const ScopedForceEvalSuppress &) = delete;
};

} // namespace forcesmith::events
