#include "forcesmith/io/logging.hpp"

#include "forcesmith/events/signals.hpp"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <vector>

namespace forcesmith::log {

void init(const std::string &file) {
  constexpr std::size_t max_size = 5UL * 1024 * 1024; // 5 MiB per file
  constexpr std::size_t max_files = 3;

  auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto rotating = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
      file, max_size, max_files);

  std::vector<spdlog::sink_ptr> sinks{console, rotating};
  auto logger =
      std::make_shared<spdlog::logger>("forcesmith", sinks.begin(), sinks.end());
  logger->set_level(spdlog::level::info);
  logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

  spdlog::set_default_logger(std::move(logger));
}

SignalSinks connect_signals() {
  SignalSinks s;

  s.iter = events::on_iteration.connect([](const events::IterationStats &st) {
    spdlog::info("iter {:>5}  obj={:.6e}  |grad|={:.3e}", st.iteration,
                 st.objective, st.grad_norm);
  });

  s.force_eval =
      events::on_force_eval.connect([](const events::ForceEvalStats &st) {
        spdlog::trace("  conf {:>4}  rms_force={:.4e}", st.conf_index,
                      st.rms_force);
      });

  s.output =
      events::on_output.connect([] { spdlog::default_logger()->flush(); });

  return s;
}

} // namespace forcesmith::log
