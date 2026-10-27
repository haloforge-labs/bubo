// SPDX-License-Identifier: MIT
#include "guard.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "guard_config.hpp"

namespace gufo::npu {
namespace {
std::string F(const char* format, double a = 0, double b = 0, double c = 0) {
  char text[256];
  std::snprintf(text, sizeof(text), format, a, b, c);
  return text;
}
constexpr double kNsPerS = 1e9;
}  // namespace

const char* GuardModeName(GuardMode mode) noexcept {
  switch (mode) {
    case GuardMode::kOn:
      return "on";
    case GuardMode::kOff:
      return "off";
    case GuardMode::kLog:
      return "log";
  }
  return "on";
}
bool ParseGuardMode(const std::string& text, GuardMode* mode) noexcept {
  if (text == "on")
    *mode = GuardMode::kOn;
  else if (text == "off")
    *mode = GuardMode::kOff;
  else if (text == "log")
    *mode = GuardMode::kLog;
  else
    return false;
  return true;
}
bool ParseGuardOverrides(const std::string& text, GuardOverrides* out,
                         std::string* error) {
  *out = GuardOverrides{};
  if (text.empty())
    return true;
  std::size_t at = 0;
  for (;;) {
    std::size_t end = text.find(',', at);
    if (end == std::string::npos)
      end = text.size();
    const std::string item = text.substr(at, end - at);
    const bool last = end >= text.size();
    at = end + 1;
    if (item.empty()) {
      *error = "empty item (stray or trailing comma)";
      return false;
    }
    const auto eq = item.find('=');
    if (eq == std::string::npos) {
      *error = "expected key=value, got '" + item + "'";
      return false;
    }
    const std::string key = item.substr(0, eq), value = item.substr(eq + 1);
    char* stop = nullptr;
    const double number = std::strtod(value.c_str(), &stop);
    if (value.empty() || stop != value.c_str() + value.size() ||
        !std::isfinite(number)) {
      *error = "bad value for " + key + ": '" + value + "'";
      return false;
    }
    if (key == "power_w" && number >= 1 && number <= 1000)
      out->power_w = number;
    else if (key == "hold_s" && number >= 1 && number <= 86400)
      out->hold_s = number;
    else {
      *error =
          (key == "power_w" || key == "hold_s")
              ? key + " outside " + (key == "power_w" ? "1..1000" : "1..86400")
              : "unknown key '" + key + "' (power_w, hold_s)";
      return false;
    }
    if (last)
      return true;
  }
}

NpuGuard::NpuGuard(GuardMode mode, GuardOverrides overrides, GuardHooks hooks)
    : mode_(mode),
      enforce_(mode == GuardMode::kOn),
      power_w_(overrides.power_w.value_or(guard::POWER_W)),
      hold_s_(overrides.hold_s.value_or(guard::HOLD_SECONDS)),
      hooks_(std::move(hooks)) {
  if (!hooks_.now_ns)
    hooks_.now_ns = [] { return SteadyNowNs(); };
}

bool NpuGuard::MemoryWindowValid(const MemoryWindow& w) const {
  return w.ok && w.samples >= guard::MEM_MIN_SAMPLES &&
         w.span_s >= guard::MEM_MIN_SPAN_S &&
         w.max_gap_s <= guard::MEM_MAX_GAP_S;
}
bool NpuGuard::PagesCondition(const MemoryWindow& w) const {
  return w.pages_samples >= guard::MEM_MIN_SAMPLES &&
         w.pages_mean >= guard::MEM_PAGES_PER_S &&
         static_cast<double>(w.min_avail_mib) < guard::MEM_AVAIL_MIB;
}

void NpuGuard::Trip(const char* reason, const std::string& detail,
                    std::optional<double> power_w, const MemoryWindow* window) {
  if (tripped_)
    return;  // already held: no extra count, no reason overwrite
  const auto now = hooks_.now_ns();
  // State first, then logging and publishing (they may fail or be slow).
  ++trips_;
  tripped_ = true;
  reason_ = reason;
  tripped_at_ns_ = now;
  sticky_ = trips_ >= guard::STICKY_AFTER_TRIPS;
  hold_until_ns_ = now + static_cast<std::int64_t>(hold_s_ * kNsPerS);
  last_still_held_log_ns_ = now;
  power_run_ = 0;
  run_means_.clear();
  GuardTripInfo info;
  info.held = true;
  info.reason = reason;
  info.trips = trips_;
  info.sticky = sticky_;
  info.power_w_mean = power_w;
  if (window && window->ok) {
    info.min_available_mib = static_cast<double>(window->min_avail_mib);
    info.max_commit_pct = window->max_commit_pct;
  }
  if (enforce_) {
    if (hooks_.set_keepalive_paused && !guard::HEARTBEAT_WHILE_HELD)
      hooks_.set_keepalive_paused(true);
    if (hooks_.publish)
      hooks_.publish(info);
  }
  if (hooks_.log) {
    const std::string head =
        enforce_ ? std::string("npu guard: NPU prefill held (") + reason + "): "
                 : std::string("npu guard: LOG MODE would hold NPU prefill (") +
                       reason + "): ";
    const std::string tail =
        sticky_ ? F("; sticky after %.0f trips, stays held until restart",
                    guard::STICKY_AFTER_TRIPS)
                : F("; hold %.0f s", hold_s_);
    hooks_.log(true, head + detail + tail +
                         F("; trips=%.0f/", static_cast<double>(trips_)) +
                         std::to_string(guard::STICKY_AFTER_TRIPS) +
                         (enforce_ ? "; new chunks use the GPU"
                                   : "; log mode: nothing is held"));
  }
}

void NpuGuard::TryRearm(std::int64_t now) {
  const std::int64_t t0 =
      now - static_cast<std::int64_t>(guard::CLEAR_WINDOW_S * kNsPerS);
  bool power_clear = true, memory_clear = true;
  double power_mean = -1, commit_max = -1;
  if (hooks_.power) {
    const auto w = hooks_.power(t0, now);
    if (w.ok) {
      power_mean = w.mean_w;
      power_clear = w.mean_w < guard::POWER_CLEAR_W;
    }  // no data (or chip power unavailable): treated as clear
  }
  if (hooks_.memory) {
    const auto m = hooks_.memory(
        t0, now, guard::MEM_PAGES_CLAMP_FACTOR * guard::MEM_PAGES_PER_S);
    if (m.ok) {
      commit_max = m.max_commit_pct;
      memory_clear =
          m.max_commit_pct < guard::MEM_COMMIT_CLEAR_PCT && !PagesCondition(m);
    }
  }
  if (!(power_clear && memory_clear)) {
    if (hooks_.log && now - last_still_held_log_ns_ >=
                          static_cast<std::int64_t>(60 * kNsPerS)) {
      last_still_held_log_ns_ = now;
      hooks_.log(
          false,
          "npu guard: hold time is over but the conditions are "
          "not clear (" +
              (power_mean >= 0 ? F("power_30s=%.1f W >= %.1f", power_mean,
                                   guard::POWER_CLEAR_W)
                               : std::string("power_30s=na")) +
              (commit_max >= 0 ? F(", commit_max=%.1f %% (clear < %.1f)",
                                   commit_max, guard::MEM_COMMIT_CLEAR_PCT)
                               : std::string(", commit=na")) +
              "); staying held");
    }
    return;
  }
  if (enforce_) {
    // Cold NPU after a long pause is an untested regime: wake it with one
    // synchronous heartbeat before clearing the hold. A failure takes the
    // normal fault path (faulted), and then nothing is re-armed.
    if (hooks_.warm_up) {
      std::string beat_error;
      const auto beat_start = hooks_.now_ns();
      const bool beat_ok = hooks_.warm_up(&beat_error);
      const double beat_ms =
          static_cast<double>(hooks_.now_ns() - beat_start) / 1e6;
      if (!beat_ok) {
        if (hooks_.log)
          hooks_.log(
              true,
              F("npu guard: re-arm warm-up beat FAILED after %.1f ms",
                beat_ms) +
                  (beat_error.empty() ? std::string() : ": " + beat_error) +
                  "; the NPU fault path applies, not re-armed");
        return;
      }
      if (hooks_.log)
        hooks_.log(false,
                   F("npu guard: re-arm warm-up beat ok in %.1f ms", beat_ms));
    }
    if (hooks_.process_failed && hooks_.process_failed())
      return;
  }
  const double held_s = static_cast<double>(now - tripped_at_ns_) / kNsPerS;
  tripped_ = false;
  reason_.clear();
  power_run_ = 0;
  run_means_.clear();
  rearmed_ = true;  // the next NPU chunk is the first one after a re-arm
  if (enforce_) {
    if (hooks_.set_keepalive_paused && !guard::HEARTBEAT_WHILE_HELD)
      hooks_.set_keepalive_paused(false);
    if (hooks_.publish) {
      GuardTripInfo info;
      info.held = false;
      info.trips = trips_;
      hooks_.publish(info);
    }
  }
  if (hooks_.log)
    hooks_.log(false,
               "npu guard: NPU prefill re-armed after " + F("%.0f s", held_s) +
                   ": " +
                   (power_mean >= 0 ? F("power_30s=%.1f W < %.1f", power_mean,
                                        guard::POWER_CLEAR_W)
                                    : std::string("power_30s=na")) +
                   (commit_max >= 0 ? F(", commit=%.1f %% < %.1f", commit_max,
                                        guard::MEM_COMMIT_CLEAR_PCT)
                                    : std::string(", commit=na")) +
                   "; trips=" + std::to_string(trips_));
}

bool NpuGuard::StaysHeld() const {
  if (!held())
    return false;
  if (sticky_)
    return true;
  return hooks_.now_ns &&
         hooks_.now_ns() + static_cast<std::int64_t>(
                               guard::HOLD_SKIP_BACKUP_MARGIN_S * kNsPerS) <
             hold_until_ns_;
}

bool NpuGuard::OnBeginChunk() {
  if (mode_ == GuardMode::kOff)
    return false;
  if (hooks_.process_failed && hooks_.process_failed())
    return false;  // faults win; nothing is re-armed or tripped afterwards
  const auto now = hooks_.now_ns();
  if (first_chunk_ns_ == 0)
    first_chunk_ns_ = now;
  if (tripped_) {
    if (!sticky_ && now >= hold_until_ns_)
      TryRearm(now);
    return held();
  }
  if (hooks_.memory &&
      now - first_chunk_ns_ >=
          static_cast<std::int64_t>(guard::MEM_STARTUP_GRACE_S * kNsPerS)) {
    const std::int64_t t0 =
        now - static_cast<std::int64_t>(guard::MEM_WINDOW_S * kNsPerS);
    const auto w = hooks_.memory(
        t0, now, guard::MEM_PAGES_CLAMP_FACTOR * guard::MEM_PAGES_PER_S);
    if (MemoryWindowValid(w)) {
      if (w.min_commit_pct >= guard::MEM_COMMIT_PCT) {
        Trip("memory",
             F("commit_pct_min=%.1f >= %.1f over %.0f s", w.min_commit_pct,
               guard::MEM_COMMIT_PCT, w.span_s) +
                 " (n=" + std::to_string(w.samples) + " samples)",
             std::nullopt, &w);
      } else if (PagesCondition(w)) {
        Trip("memory",
             F("pages_mean=%.0f/s >= %.0f", w.pages_mean,
               guard::MEM_PAGES_PER_S) +
                 F(" and avail_min=%.0f MiB < %.0f",
                   static_cast<double>(w.min_avail_mib), guard::MEM_AVAIL_MIB) +
                 F(" over %.0f s", w.span_s),
             std::nullopt, &w);
      }
    }
  }
  return held();
}

void NpuGuard::OnChunkEnd(const ChunkStart& start, const ChunkEnd& end) {
  if (mode_ == GuardMode::kOff || tripped_)
    return;
  if (hooks_.process_failed && hooks_.process_failed())
    return;
  if (start.idle_gap_ms >
      static_cast<std::int64_t>(guard::POWER_RESET_IDLE_S * 1000)) {
    power_run_ = 0;
    run_means_.clear();
  }
  if (!end.power.ok) {  // chip power unavailable: power rule off, run forgotten
    power_run_ = 0;
    run_means_.clear();
    return;
  }
  if (end.power.approx || end.power.intervals < guard::POWER_MIN_INTERVALS)
    return;  // too short to judge: neither counts nor resets
  if (end.power.mean_w > power_w_) {
    ++power_run_;
    run_means_.push_back(end.power.mean_w);
    if (power_run_ >= guard::POWER_CHUNKS) {
      std::string list = "chunk_means_w=[";
      for (std::size_t i = 0; i < run_means_.size(); ++i)
        list += (i ? "," : "") + F("%.1f", run_means_[i]);
      list += F("] > %.1f W", power_w_) + " x" + std::to_string(power_run_);
      const double mean = end.power.mean_w;
      Trip("power", list, mean, nullptr);
    }
  } else {
    power_run_ = 0;
    run_means_.clear();
  }
}
}  // namespace gufo::npu
