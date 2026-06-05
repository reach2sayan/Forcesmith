#pragma once

#include <boost/signals2/connection.hpp>

#include <string>

namespace forcesmith::log {

// Configure the default logger: a colour console sink plus a rotating file sink
// at `file`. Idempotent enough to call once at program start.
void init(const std::string &file = "forcesmith.log");
struct SignalSinks {
  boost::signals2::scoped_connection iter;
  boost::signals2::scoped_connection force_eval;
  boost::signals2::scoped_connection output;
};

[[nodiscard]] SignalSinks connect_signals();

} // namespace forcesmith::log
