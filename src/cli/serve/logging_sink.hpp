#ifndef GUFO_SERVER_LOGGING_SINK_HPP_
#define GUFO_SERVER_LOGGING_SINK_HPP_

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "src/cli/serve/logging.hpp"

namespace gufo::server {

// The same queue implementation is exercised with a CPU-only injected writer.
// A writer returns success, or throws; it must not call Logger.
// Void test writers are adapted to return success unless they throw.
class AsyncLogSink {
public:
  using Writer = std::function<bool(std::string_view)>;
  struct Stats {
    std::uint64_t accepted, delivered, dropped, reported_drops, pending;
    std::size_t pending_bytes;
    std::uint64_t failures, failed_lines;
  };
  explicit AsyncLogSink(Writer writer, bool synchronous = false);
  template<typename Output>
  explicit AsyncLogSink(Output output, bool synchronous = false)
      : AsyncLogSink(Writer([output = std::move(output)](
                                std::string_view batch) mutable {
                       if constexpr (std::is_void_v<std::invoke_result_t<
                                         Output&, std::string_view>>) {
                         output(batch);
                         return true;
                       } else {
                         return output(batch);
                       }
                     }),
                     synchronous) {}
  ~AsyncLogSink();
  AsyncLogSink(const AsyncLogSink&) = delete;
  AsyncLogSink& operator=(const AsyncLogSink&) = delete;
  void Push(std::string line, LogLevel level);
  // A prefix barrier: concurrent producers cannot extend this wait forever.
  // Waits are capped at 2 s; success means the prefix has no lost lines.
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
