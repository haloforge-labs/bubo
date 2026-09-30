#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRACE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRACE_HPP_

// GUFO_TRACE=1: the decode timeline (diagnostics only). A one-thread kernel
// writes the GPU wall clock into pinned memory at every phase boundary (in
// stream order, so it runs exactly when the GPU gets there), and the host
// notes its own clock at each stamp and after each synchronization. Every
// ~10 s of decoding it reports GPU busy time per phase and every GPU idle gap
// between phases, split into host work (sync return to next submit) and the
// rest (launch and wake-up latency).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

class Timeline {
public:
  enum Label : std::uint8_t {
    kVerifyBegin,
    kPrefixEnd,
    kBodyBegin,
    kVerifyEnd,
    kMtpBegin,
    kMtpEnd,
    kCandBegin,
    kCandEnd,
    kRollBegin,
    kRollEnd,
    kSynced,
    kLabels
  };
  static Timeline& Get() {
    static Timeline timeline;
    return timeline;
  }
  [[nodiscard]] bool Enabled() const noexcept { return enabled_; }
  void Gpu(hipStream_t stream, Label label, std::uint32_t arg = 0) {
    if (!enabled_ || next_slot_ >= kSlots) {
      return;
    }
    if (!have_start_) {
      cal_start_ = Calibrate(stream);
      have_start_ = true;
    }
    const auto slot = next_slot_++;
    GpuStamp(stamps_ + slot, stream);
    events_.push_back({label, arg, Now(), static_cast<std::int32_t>(slot)});
  }
  void Host(Label label) {
    if (enabled_) {
      events_.push_back({label, 0, Now(), -1});
    }
  }
  /// Call with the stream synchronized (after a verify pass).
  void MaybeReport(hipStream_t stream) {
    if (!enabled_ || Now() - window_start_ < 10'000'000'000LL) {
      return;
    }
    cal_end_ = Calibrate(stream);
    Report();
    cal_start_ = cal_end_;
    events_.clear();
    next_slot_ = 0;
    window_start_ = Now();
  }

private:
  static constexpr std::size_t kSlots = 1 << 18;
  struct Event {
    Label label;
    std::uint32_t arg;
    std::int64_t host_ns;
    std::int32_t slot;
  };
  struct Acc {
    double ms{0};
    double host_ms{0};
    std::uint64_t n{0};
  };
  struct Cal {
    std::int64_t host_ns{0};
    std::uint64_t tick{0};
  };
  /// Pairs a host clock reading with the GPU clock: a kernel holds on a
  /// coherent flag, the host notes its clock as it writes the flag, and the
  /// kernel stamps right after (the GPU sees the write within microseconds).
  Cal Calibrate(hipStream_t stream) {
    const std::uint32_t want = ++cal_counter_;
    WaitHost(cal_flag_, want, stream);
    GpuStamp(stamps_ + kSlots, stream);
    (void)hipStreamQuery(stream);
    const std::int64_t until = Now() + 2'000'000;
    while (Now() < until) {
    }
    Cal cal;
    cal.host_ns = Now();
    *static_cast<volatile std::uint32_t*>(cal_flag_) = want;
    (void)hipStreamSynchronize(stream);
    cal.tick = stamps_[kSlots];
    return cal;
  }
  /// Host clock of a GPU tick, interpolated between the window's two
  /// calibrations.
  [[nodiscard]] double HostOf(std::uint64_t tick) const {
    const double slope =
        cal_end_.tick > cal_start_.tick
            ? static_cast<double>(cal_end_.host_ns - cal_start_.host_ns) /
                  static_cast<double>(cal_end_.tick - cal_start_.tick)
            : ns_per_tick_;
    return static_cast<double>(cal_start_.host_ns) +
           (static_cast<double>(tick) - static_cast<double>(cal_start_.tick)) *
               slope;
  }

  Timeline() {
    const char* v = std::getenv("GUFO_TRACE");
    enabled_ = v != nullptr && v[0] != '\0' && v[0] != '0';
    if (!enabled_) {
      return;
    }
    void* stamps = nullptr;
    int device = 0;
    int khz = 0;
    void* cal = nullptr;
    if (hipHostMalloc(&stamps, (kSlots + 1) * sizeof(std::uint64_t)) !=
            hipSuccess ||
        hipHostMalloc(&cal, 64, hipHostMallocCoherent) != hipSuccess ||
        hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) !=
            hipSuccess ||
        khz <= 0) {
      enabled_ = false;
      return;
    }
    stamps_ = static_cast<std::uint64_t*>(stamps);
    cal_flag_ = static_cast<std::uint32_t*>(cal);
    *cal_flag_ = 0;
    ns_per_tick_ = 1e6 / khz;
    events_.reserve(kSlots);
    window_start_ = Now();
  }
  static std::int64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  static const char* Name(Label label) {
    static constexpr const char* kNames[] = {
        "verify",   "prefix-end",   "body",  "verify-end",
        "draft",    "draft-end",    "cand",  "cand-end",
        "rollback", "rollback-end", "synced"};
    return kNames[label];
  }
  static bool IsBegin(Label label) {
    return label == kVerifyBegin || label == kBodyBegin || label == kMtpBegin ||
           label == kCandBegin || label == kRollBegin;
  }
  static Label BeginOf(Label end) {
    switch (end) {
      case kPrefixEnd:
        return kVerifyBegin;
      case kVerifyEnd:
        return kBodyBegin;
      case kMtpEnd:
        return kMtpBegin;
      case kCandEnd:
        return kCandBegin;
      default:
        return kRollBegin;
    }
  }

  void Report() {
    std::map<std::string, Acc> busy;
    std::map<std::string, Acc> idle;
    std::map<std::string, Acc> latency;
    double pending_end = -1;  ///< host clock of the last unsynced GPU end
    Label pending_label = kVerifyEnd;
    std::array<std::uint64_t, kLabels> begin_tick{};
    std::array<std::int64_t, kLabels> begin_host{};
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    std::uint64_t cycles = 0;
    bool have_end = false;
    std::uint64_t end_tick = 0;
    std::int64_t end_host = 0;
    std::int64_t sync_host = -1;
    Label end_label = kVerifyEnd;
    double busy_total = 0;
    for (const Event& e : events_) {
      if (e.slot < 0) {
        sync_host = e.host_ns;
        if (pending_end >= 0) {
          auto& a = latency[std::string(Name(pending_label)) + " seen"];
          a.ms += (static_cast<double>(e.host_ns) - pending_end) * 1e-6;
          ++a.n;
          pending_end = -1;
        }
        continue;
      }
      const std::uint64_t tick = stamps_[e.slot];
      if (IsBegin(e.label)) {
        auto& a = latency[std::string(Name(e.label)) + " start"];
        a.ms += (HostOf(tick) - static_cast<double>(e.host_ns)) * 1e-6;
        ++a.n;
      } else {
        pending_end = HostOf(tick);
        pending_label = e.label;
      }
      if (first == 0) {
        first = tick;
      }
      last = std::max(last, tick);
      const auto ms = [&](std::uint64_t from) {
        return tick > from
                   ? static_cast<double>(tick - from) * ns_per_tick_ * 1e-6
                   : 0.0;
      };
      if (IsBegin(e.label)) {
        cycles += e.label == kVerifyBegin;
        if (have_end) {
          const std::int64_t from = sync_host > end_host ? sync_host : end_host;
          auto& a = idle[std::string(Name(end_label)) + "->" + Name(e.label)];
          a.ms += ms(end_tick);
          a.host_ms += static_cast<double>(e.host_ns - from) * 1e-6;
          ++a.n;
        }
        begin_tick[e.label] = tick;
        begin_host[e.label] = e.host_ns;
        continue;
      }
      const double phase = ms(begin_tick[BeginOf(e.label)]);
      const std::uint32_t w = std::min<std::uint32_t>(e.arg, 16);
      char key[48];
      switch (e.label) {
        case kPrefixEnd:
          std::snprintf(key, sizeof(key), "verify-prefix w%u", w);
          break;
        case kVerifyEnd:
          std::snprintf(key, sizeof(key), "verify-body w%u", w);
          break;
        case kMtpEnd:
          std::snprintf(key, sizeof(key), "draft n%u", w);
          break;
        case kCandEnd:
          std::snprintf(key, sizeof(key), "candidates k%u", w);
          break;
        default:
          std::snprintf(key, sizeof(key), "rollback");
          break;
      }
      auto& a = busy[key];
      a.ms += phase;
      a.host_ms +=
          static_cast<double>(e.host_ns - begin_host[BeginOf(e.label)]) * 1e-6;
      ++a.n;
      busy_total += phase;
      have_end = true;
      end_tick = tick;
      end_host = e.host_ns;
      end_label = e.label;
    }
    if (cycles == 0 || last <= first) {
      return;
    }
    const double span_ms =
        static_cast<double>(last - first) * ns_per_tick_ * 1e-6;
    const double per_cycle = 1.0 / static_cast<double>(cycles);
    std::string line;
    char item[160];
    for (const auto& [key, a] : busy) {
      std::snprintf(item, sizeof(item), " %s %llux%.2f (submit %.3f)",
                    key.c_str(), static_cast<unsigned long long>(a.n),
                    a.ms / a.n, a.host_ms / a.n);
      line += item;
    }
    std::fprintf(stderr,
                 "qwen38_flash_next: trace %.1f s, %llu cycles: GPU busy "
                 "%.1f%% (%.2f ms/cycle busy, %.2f idle) | GPU ms per "
                 "phase:%s\n",
                 span_ms * 1e-3, static_cast<unsigned long long>(cycles),
                 100.0 * busy_total / span_ms, busy_total * per_cycle,
                 (span_ms - busy_total) * per_cycle, line.c_str());
    line.clear();
    for (const auto& [key, a] : idle) {
      std::snprintf(item, sizeof(item), " %s %.3f (%llux%.3f, host %.3f)",
                    key.c_str(), a.ms * per_cycle,
                    static_cast<unsigned long long>(a.n), a.ms / a.n,
                    a.host_ms / a.n);
      line += item;
    }
    std::fprintf(stderr,
                 "qwen38_flash_next: trace idle ms/cycle (count x ms each, "
                 "host work each):%s\n",
                 line.c_str());
    line.clear();
    for (const auto& [key, a] : latency) {
      std::snprintf(item, sizeof(item), " %s %.3f;", key.c_str(),
                    a.n ? a.ms / a.n : 0.0);
      line += item;
    }
    std::fprintf(stderr,
                 "qwen38_flash_next: trace latency ms (start = GPU start after "
                 "the host queued it; seen = host sync after the GPU end):%s\n",
                 line.c_str());
  }

  bool enabled_{false};
  std::uint64_t* stamps_{nullptr};
  std::size_t next_slot_{0};
  double ns_per_tick_{10.0};
  std::int64_t window_start_{0};
  std::vector<Event> events_;
  std::uint32_t* cal_flag_{nullptr};
  std::uint32_t cal_counter_{0};
  bool have_start_{false};
  Cal cal_start_{};
  Cal cal_end_{};
};

inline Timeline& Tl() {
  return Timeline::Get();
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRACE_HPP_
