#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace expert_bench {

inline constexpr std::size_t kCacheBytes = 32U << 20;
inline constexpr std::size_t kEvictBytes = 256U << 20;
inline constexpr std::string_view kProtocol = "evict256m-per-call-v1";
inline constexpr double kMaxGbps = 250.0;

struct Timing {
  double us;
  std::string_view protocol = kProtocol;
  std::size_t evict_bytes = kEvictBytes;
  std::string reason = "ok";
  double raw_us = std::numeric_limits<double>::quiet_NaN();
  int hip_error = 0, call_rc = 0, sync_error = 0, last_error = 0;
  int sample = -1, completed = 0;
  // Record flags indicate accepted API calls, not verified device timestamps.
  bool start_recorded = false, stop_recorded = false;
};

struct Failure : std::runtime_error {
  int hip_error, call_rc, sync_error, last_error;
  Failure(const char* reason, int hip = 0, int rc = 0, int sync = 0,
          int last = 0)
      : std::runtime_error(reason),
        hip_error(hip),
        call_rc(rc),
        sync_error(sync),
        last_error(last) {}
};

inline void SetFailure(Timing& timing, const Failure& error) {
  timing.us = 0;
  timing.reason = error.what();
  timing.hip_error = error.hip_error;
  timing.call_rc = error.call_rc;
  timing.sync_error = error.sync_error;
  timing.last_error = error.last_error;
}

inline void RequireComparable(const Timing& a, const Timing& b) {
  if (a.protocol != kProtocol || b.protocol != a.protocol ||
      a.evict_bytes != kEvictBytes || b.evict_bytes != a.evict_bytes)
    throw std::runtime_error("timing cold protocols differ");
}

inline double Gbps(double bytes, double us) {
  return bytes / us / 1000.0;
}

inline double MeanDistinct(std::span<const int> sets, int samples) {
  if (sets.empty() || samples < 1)
    throw std::runtime_error("timing sets and sample count must be positive");
  double sum = 0;
  for (int i = 0; i < samples; ++i)
    sum += sets[std::size_t(i) % sets.size()];
  return sum / samples;
}

inline bool Valid(double bytes, const Timing& timing) {
  const double gbps = Gbps(bytes, timing.us);
  return timing.reason == "ok" && timing.protocol == kProtocol &&
         timing.evict_bytes == kEvictBytes && std::isfinite(bytes) &&
         bytes > 0 && std::isfinite(timing.us) && timing.us > 0 &&
         std::isfinite(gbps) && gbps <= kMaxGbps;
}

inline int ExitCode(bool strict, std::size_t invalid) {
  return strict && invalid != 0 ? 2 : 0;
}

// Eviction precedes the start event on the sample's stream. Each event interval
// contains one call; synchronization and eviction are outside that interval.
template<class Evict, class Start, class Call, class Stop, class Elapsed>
Timing Measure(int samples, Evict evict, Start start, Call call, Stop stop,
               Elapsed elapsed) {
  if (samples < 1)
    throw std::runtime_error("timing sample count must be positive");
  Timing result{0};
  try {
    double total = 0;
    for (int i = 0; i < samples; ++i) {
      result.sample = i;
      result.raw_us = std::numeric_limits<double>::quiet_NaN();
      result.start_recorded = result.stop_recorded = false;
      evict();
      start();
      result.start_recorded = true;
      const int rc = call(i);
      if (rc != 0)
        throw Failure("launch_failure", 0, rc);
      stop();
      result.stop_recorded = true;
      result.raw_us = elapsed();
      if (!std::isfinite(result.raw_us) || result.raw_us <= 0) {
        result.reason = std::isfinite(result.raw_us) ? "nonpositive_interval"
                                                     : "nonfinite_interval";
        return result;
      }
      total += result.raw_us;
      ++result.completed;
    }
    result.us = total / samples;
  } catch (const Failure& error) {
    SetFailure(result, error);
  }
  return result;
}

// A cell failure is data for the report; it must not suppress later cells.
template<class Cells, class MeasureCell, class ReportCell>
void Sweep(const Cells& cells, MeasureCell measure, ReportCell report) {
  for (const auto& cell : cells) {
    Timing timing{0};
    try {
      timing = measure(cell);
    } catch (const Failure& error) {
      SetFailure(timing, error);
    } catch (const std::exception&) {
      timing.reason = "cell_exception";
    }
    report(cell, timing);
  }
}

}  // namespace expert_bench
