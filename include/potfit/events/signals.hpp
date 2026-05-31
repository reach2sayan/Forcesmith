#pragma once

#include <boost/signals2/signal.hpp>
#include <atomic>
#include <cstdint>
#include <utility>

namespace potfit {
struct Configuration; // fwd-decl: enrich ForceEvalStats without pulling in atom.hpp
}

namespace potfit::events {

struct IterationStats {
  std::uint64_t iteration = 0;
  double objective = 0.0;
  double grad_norm = 0.0;
};

struct ForceEvalStats {
  std::uint64_t conf_index = 0;
  double rms_force = 0.0;
  // The configuration just evaluated
  const Configuration &cfg;
};

// on_force_eval fires once per configuration from inside eval_forces, i.e. from
// every worker thread of the parallel config loop. Wrap it so the whole signal
// can be muted for the duration of a parallel region: the per-config diagnostic
// is suppressed (currently no slots are attached anyway), avoiding the signals2
// mutex contention of firing it concurrently on hot paths. Drop-in compatible
// with the raw signal — call as on_force_eval(stats), connect via
// .connect(...).
struct SuppressibleSignal {
  boost::signals2::signal<void(const ForceEvalStats &)> sig;
  std::atomic<bool> suppressed{false};
  void operator()(const ForceEvalStats &s) const {
    if (!suppressed.load(std::memory_order_relaxed)) {
      sig(s);
    }
  }
  template <typename F> auto connect(F &&f) {
    return sig.connect(std::forward<F>(f));
  }
};

inline boost::signals2::signal<void(const IterationStats &)> on_iteration;
inline SuppressibleSignal on_force_eval;
inline boost::signals2::signal<void()> on_output;

// RAII: mute on_force_eval for the lifetime of the guard
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

} // namespace potfit::events
