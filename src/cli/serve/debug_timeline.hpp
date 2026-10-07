#ifndef GUFO_SERVER_DEBUG_TIMELINE_HPP_
#define GUFO_SERVER_DEBUG_TIMELINE_HPP_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

// Owned by a request only after opt-in; timestamps come from serving work units.
class DebugTimeline {
public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::size_t kMaxDecodeCycles = 4096;

  explicit DebugTimeline(Clock::time_point start) : start_(start) {}

  void Failed(std::exception_ptr failure) noexcept {
    if (!failure_)
      failure_ = std::move(failure);
  }

  void Bind(std::uint64_t request_id) noexcept {
    request_id_ = request_id;
    try {
      for (const auto& event : events_)
        Log(event);
    } catch (...) {
      Failed(std::current_exception());
    }
  }

  void Add(std::string_view name, Clock::time_point when,
           json::Value fields = json::Value::object()) noexcept {
    if (failure_)
      return;
    try {
      fields["event"] = std::string(name);
      fields["t_ms"] =
          std::chrono::duration<double, std::milli>(when - start_).count();
      events_.push_back(std::move(fields));
      if (request_id_ != 0)
        Log(events_.back());
    } catch (...) {
      Failed(std::current_exception());
    }
  }

  void Cycle(Clock::time_point when, double ms, std::size_t verify_width,
             std::size_t drafts, std::size_t accepted) noexcept {
    if (cycles_++ >= kMaxDecodeCycles) {
      truncated_ = true;
      return;
    }
    try {
      auto fields = json::Value::object();
      fields["index"] = cycles_ - 1;
      fields["verify_width"] = verify_width;
      fields["drafts"] = drafts;
      fields["accepted"] = accepted;
      fields["ms"] = ms;
      Add("decode_cycle", when, std::move(fields));
    } catch (...) {
      Failed(std::current_exception());
    }
  }

  [[nodiscard]] bool truncated() const noexcept { return truncated_; }

  [[nodiscard]] json::Value Serialize() const {
    if (failure_)
      std::rethrow_exception(failure_);
    // Snapshot completion can follow the pass that started it; present starts
    // in chronological order without moving work or adding synchronization.
    auto ordered = events_;
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a,
                                                       const auto& b) {
      return a.member_double("t_ms") < b.member_double("t_ms");
    });
    auto result = json::Value::array();
    for (auto& event : ordered)
      result.push_back(std::move(event));
    return result;
  }

private:
  void Log(const json::Value& event) const {
    Logger::Diagnostic("timeline", "req=" + std::to_string(request_id_) + " " +
                                 event.dump());
  }

  Clock::time_point start_;
  std::uint64_t request_id_{0};
  std::size_t cycles_{0};
  bool truncated_{false};
  std::vector<json::Value> events_;
  std::exception_ptr failure_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_TIMELINE_HPP_
