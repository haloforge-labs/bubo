#ifndef GUFO_SERVER_LOGGING_SINK_HPP_
#define GUFO_SERVER_LOGGING_SINK_HPP_

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "src/cli/serve/logging.hpp"

namespace gufo::server {

// The same queue implementation is exercised with a CPU-only injected writer.
// A writer consumes the entire batch or throws; it must not call Logger.
class AsyncLogSink {
public:
  using Writer = std::function<void(std::string_view)>;
  struct Stats {
    std::uint64_t accepted, delivered, dropped, reported_drops, pending;
    std::size_t pending_bytes;
  };
  explicit AsyncLogSink(Writer writer, bool synchronous = false);
  ~AsyncLogSink();
  AsyncLogSink(const AsyncLogSink&) = delete;
  AsyncLogSink& operator=(const AsyncLogSink&) = delete;
  void Push(std::string line, LogLevel level);
  // A prefix barrier: concurrent producers cannot extend this wait forever.
  // An unlimited barrier is for explicit durability; exits use a 2 s deadline.
  bool FlushSync(
      std::chrono::milliseconds timeout = std::chrono::milliseconds::max());
  Stats Shutdown();
  Stats GetStats() const;

private:
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace gufo::server
#endif
