#ifndef GUFO_SERVER_DEBUG_TIMELINE_HPP_
#define GUFO_SERVER_DEBUG_TIMELINE_HPP_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
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

  // Supplied by the same settings builder as hook 1, after request resolution.
  void Header(json::Value bundle) {
    bundle_ = std::move(bundle);
    for (auto& event : events_) {
      if (event.member_str("event") == "admission") {
        event["bundle"] = bundle_;
        return;
      }
    }
  }

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
      // Aggregate before applying the cycle cap so totals include omitted work.
      if (name == "prefill") {
        prefill_ms_ += fields.member_double("ms");
        prefill_passes_.push_back(fields);
      } else if (name == "queue") {
        queue_ms_ += fields.member_double("wait_ms");
      } else if (name == "cache_lookup") {
        auto restore = json::Value::object();
        restore["kind"] = fields.member_str("source");
        restore["ms"] = fields.member_double("restore_ms");
        restore["bytes"] = fields.member_size("bytes");
        restore_ms_ += fields.member_double("restore_ms");
        restores_.push_back(std::move(restore));
      } else if (name == "finish") {
        const double wall = std::chrono::duration<double, std::milli>(
            when - start_).count();
        auto summary = json::Value::object();
        summary["cycle_count"] = cycles_;
        summary["cycle_ms"] = cycle_ms_;
        summary["prefill_ms"] = prefill_ms_;
        summary["restore_ms"] = restore_ms_;
        summary["queue_ms"] = queue_ms_;
        summary["wall_ms"] = wall;
        summary["unattributed_ms"] =
            wall - (cycle_ms_ + prefill_ms_ + restore_ms_ + queue_ms_);
        summary["prefill_passes"] = prefill_passes_;
        summary["restores"] = restores_;
        fields["summary"] = std::move(summary);
      }
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
             std::size_t drafts, std::size_t accepted,
             std::optional<std::size_t> lookup = std::size_t{0},
             std::optional<bool> batched = false, bool sampled = false) noexcept {
    cycle_ms_ += ms;
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
      fields["accepted_length"] = accepted;
      fields["partial_accept"] = accepted < drafts;
      if (drafts == 0)
        lookup = 0;
      if (lookup && *lookup <= drafts) {
        auto sources = json::Value::object();
        sources["mtp_head"] = drafts - *lookup;
        sources["prompt_lookup"] = *lookup;
        fields["draft_sources"] = std::move(sources);
        fields["draft_source"] = drafts == 0 ? "none"
            : *lookup == 0 ? "mtp_head"
            : *lookup == drafts ? "prompt_lookup" : "mixed";
      }
      if (const auto* policy = bundle_.find("mtp_policy")) {
        auto cycle_policy = json::Value::object();
        for (const auto& [key, value] : policy->members()) {
          if (key != "name")
            cycle_policy[key] = value;
        }
        // The batch adapter can retry a peer serially without reporting it.
        // Omit the effective name when that engine decision is unavailable.
        if (batched)
          cycle_policy["name"] = policy->member_str("name") == "survival" &&
                                         sampled && !*batched
                                     ? "survival" : "length";
        fields["policy"] = std::move(cycle_policy);
      }
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
  double cycle_ms_{0}, prefill_ms_{0}, restore_ms_{0}, queue_ms_{0};
  json::Value bundle_ = json::Value::object();
  json::Value prefill_passes_ = json::Value::array();
  json::Value restores_ = json::Value::array();
  std::vector<json::Value> events_;
  std::exception_ptr failure_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_TIMELINE_HPP_
