#pragma once

// Logging façade: routes the optimizer's event signals (potfit::events) to an
// spdlog logger with a colored console sink and a persistent rotating file
// sink. Intentionally spdlog-free in this header — all spdlog (and its bundled
// fmt) lives in logging.cpp so its include/compile cost never leaks into the
// rest of the C++23 build. Callers only see boost::signals2 connection guards.

#include <boost/signals2/connection.hpp>

#include <string>

namespace potfit::log {

// Configure the default logger: a color console sink plus a rotating file sink
// at `file`. Idempotent enough to call once at program start.
void init(const std::string &file = "potfit.log");

// Connect the event signals (on_iteration / on_force_eval / on_output) to the
// logger. The returned guards own the connections and auto-disconnect on
// destruction, so the result must outlive the optimize() call.
struct SignalSinks {
  boost::signals2::scoped_connection iter;
  boost::signals2::scoped_connection force_eval;
  boost::signals2::scoped_connection output;
};

[[nodiscard]] SignalSinks connect_signals();

} // namespace potfit::log
