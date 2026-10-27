#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace bubo::cli {

// Rates belong to one slot, request and instance, never to the server's
// aggregate counters. Cached tokens count as read but not as prefill work.
class StatusWatch {
public:
  using Clock = std::chrono::steady_clock;
  using Value = gufo::json::Value;

  std::vector<std::string> Lines(const Value& snapshot,
                                 Clock::time_point now = Clock::now()) {
    const auto* identity = snapshot.find("identity");
    const auto instance = identity ? identity->member_str("instance_id") : "";
    if (instance != instance_) {
      previous_.clear();
      instance_ = instance;
    }
    std::map<double, Sample> next;
    std::vector<std::string> lines;
    const auto* slots = snapshot.find("slots");
    if (slots && slots->is_array()) {
      for (const auto& slot : slots->items()) {
        const auto id = Count(slot, "id");
        if (!id)
          continue;
        std::ostringstream line;
        line << "Session " << std::fixed << std::setprecision(0) << *id + 1
             << ' ';
        const auto state = slot.member_str("state");
        const auto request = Count(slot, "request_id");
        const auto cached = Count(slot, "cached_prompt_tokens");
        const auto prefilled = Count(slot, "prefilled_tokens");
        const auto generated = Count(slot, "generated_tokens");
        Sample current{request,   prefilled, generated, state, now,
                       prefilled, now,       prefilled, now};
        std::optional<double> speed;
        const auto old = previous_.find(*id);
        if (request && old != previous_.end() &&
            old->second.request == request && old->second.state == state) {
          if (state == "prefilling") {
            const auto& prior = old->second;
            if (prefilled && prior.prefilled &&
                *prefilled >= *prior.prefilled) {
              current.baseline = prior.baseline;
              current.baseline_time = prior.baseline_time;
              current.latest = prior.latest;
              current.latest_time = prior.latest_time;
              if (*prefilled > *prior.prefilled) {
                current.latest = prefilled;
                current.latest_time = now;
              }
              const auto seconds =
                  std::chrono::duration<double>(current.latest_time -
                                                current.baseline_time)
                      .count();
              if (seconds > 0 && current.baseline && current.latest &&
                  *current.latest > *current.baseline)
                speed = (*current.latest - *current.baseline) / seconds;
            }
          } else {
            const auto seconds =
                std::chrono::duration<double>(now - old->second.time).count();
            if (seconds > 0 && old->second.generated && generated &&
                *generated >= *old->second.generated)
              speed = (*generated - *old->second.generated) / seconds;
          }
        }
        next.emplace(*id, current);
        if (state == "idle") {
          line << "idle";
        } else if (state == "prefilling") {
          line << "reading ";
          const auto total = Count(slot, "prompt_tokens");
          if (!total || !cached || !prefilled || *total == 0 ||
              *cached > *total || *prefilled > *total - *cached) {
            line << "progress unavailable";
          } else {
            const double read = *cached + *prefilled;
            const auto filled = static_cast<unsigned>(read / *total * 10);
            line << '[' << std::string(filled, '=')
                 << std::string(10 - filled, ' ') << "] "
                 << std::floor(read / *total * 100) << "% " << read << '/'
                 << *total;
            line << " | " << Rate(speed) << " tok/s | ~ETA ";
            if (speed && *speed > 0)
              line << std::ceil((*total - read) / *speed) << 's';
            else
              line << '-';
            if (*cached > 0)
              line << " (" << *cached << " cached)";
          }
        } else if (state == "generating") {
          line << "writing ";
          if (generated)
            line << *generated << " generated";
          else
            line << "count unavailable";
          line << " | " << Rate(speed) << " tok/s";
          if (cached && *cached > 0)
            line << " (" << *cached << " prompt tokens cached)";
        } else {
          line << "status unavailable";
        }
        lines.push_back(line.str());
      }
    }
    previous_ = std::move(next);
    if (lines.empty()) {
      const auto phase = snapshot.member_str("phase");
      if (phase == "multiple_instances")
        lines.push_back(
            "Several servers are running. Select one with "
            "--instance-id (see bubo status --json).");
      else if (phase == "not_prepared" || phase == "error")
        lines.push_back("Not ready. Run bubo status for the next step.");
      else if (phase == "stopped")
        lines.push_back("Ready. Start the server: bubo serve");
      else if (phase == "ready")
        lines.push_back("Server running. Session progress unavailable.");
      else if (phase == "starting" || phase == "loading" || phase == "stopping")
        lines.push_back("Server " + phase + ". Session progress unavailable.");
      else
        lines.push_back("Server status unavailable.");
    }
    return lines;
  }

private:
  static std::optional<double> Count(const Value& value, const char* key) {
    const auto* field = value.find(key);
    if (!field || !field->is_number())
      return std::nullopt;
    const double number = field->as_double();
    if (!std::isfinite(number) || number < 0 || number > 9007199254740991.0 ||
        std::floor(number) != number)
      return std::nullopt;
    return number;
  }

  static std::string Rate(std::optional<double> rate) {
    if (!rate || !std::isfinite(*rate))
      return "-";
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << *rate;
    return text.str();
  }

  struct Sample {
    std::optional<double> request, prefilled, generated;
    std::string state;
    Clock::time_point time;
    std::optional<double> baseline;
    Clock::time_point baseline_time;
    std::optional<double> latest;
    Clock::time_point latest_time;
  };
  std::string instance_;
  std::map<double, Sample> previous_;
};
}  // namespace bubo::cli
