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
  [[nodiscard]] bool StagesEnabled() const noexcept {
    return enabled_ && stage_stamps_ != nullptr;
  }
  static constexpr std::uint32_t kStageLayers = 64;
  enum StagePoint : std::uint32_t {
    kHcAttn,
    kMixer,
    kCombineHcFfn,
    kRouter,
    kShared,
    kRouted,
    kCombineEpilogue,
    kPoints
  };
  // Fixed addresses are retained by graph nodes and overwritten on replay.
  // Neither these methods nor StageMark touch the phase ring or calibrate.
  void StageMark(hipStream_t stream, std::uint32_t layer, std::uint32_t point) {
    if (StagesEnabled() && layer < kStageLayers && point < kPoints)
      GpuStamp(stage_stamps_ + layer * kPoints + point, stream);
  }
  void StageBegin(hipStream_t stream, bool prefix) {
    if (StagesEnabled())
      GpuStamp(stage_stamps_ + kStageExtra + (prefix ? 0 : 1), stream);
  }
  void StageEmbedding(hipStream_t stream) {
    if (StagesEnabled())
      GpuStamp(stage_stamps_ + kStageExtra + 2, stream);
  }
  void StageHead(hipStream_t stream) {
    if (StagesEnabled())
      GpuStamp(stage_stamps_ + kStageExtra + 3, stream);
  }
  // Prefill-only stamps: eager prefill passes also record these three points
  // per layer in a region after the decode stamps. Graphs never run with the
  // prefill phase set, so no captured graph gains a node, and the decode
  // report, its point list and its slots are unchanged.
  enum PrefillPoint : std::uint32_t { kPfMixIn, kPfIndexer, kPfMixCore, kPfPoints };
  void PrefillMark(hipStream_t stream, std::uint32_t layer, std::uint32_t point) {
    if (StagesEnabled() && layer < kStageLayers && point < kPfPoints)
      GpuStamp(stage_stamps_ + kStageSlots + layer * kPfPoints + point, stream);
  }
  // Called after an eager prefill pass has synchronised. Sorts every stamp
  // of the pass by tick and attributes each gap to the later stamp's name, so
  // a missing mark merges two stages instead of dropping time. One stderr
  // line per pass; stamp_ms is the first-to-last stamp span and host_ms the
  // host clock for the same pass (they must agree within a few percent).
  void CollectPrefill(std::uint32_t tokens, std::uint32_t start, double host_ms,
                      int ple_layer, std::uint32_t layers) {
    if (!StagesEnabled())
      return;
    enum Name : unsigned { kEmbed, kHcAttn, kHcAttnPle, kMixIn, kIndexer, kMixCore,
                           kMixOut, kCombineFfn, kRouter, kShared, kRouted,
                           kCombineEpi, kHead, kNames };
    static constexpr const char* kLabel[kNames] = {
        "embed", "hc_attn", "hc_attn_ple", "mix_in", "indexer", "mix_core",
        "mix_out", "combine_ffn", "router", "shared", "routed", "combine_epi",
        "head"};
    // Decode point -> prefill name; the old "mixer" stamp ends the mixer.
    static constexpr Name kDecodeName[kPoints] = {kHcAttn, kMixOut, kCombineFfn,
                                                  kRouter, kShared, kRouted,
                                                  kCombineEpi};
    static constexpr Name kPfName[kPfPoints] = {kMixIn, kIndexer, kMixCore};
    const auto* slots = static_cast<volatile std::uint64_t*>(stage_stamps_);
    const std::uint64_t begin = slots[kStageExtra + 1];
    struct Stamp {
      std::uint64_t tick;
      Name name;
    };
    std::vector<Stamp> stamps;
    const auto add = [&](std::uint64_t tick, Name name) {
      if (tick >= begin && begin != 0)
        stamps.push_back({tick, name});
    };
    add(slots[kStageExtra + 2], kEmbed);
    for (std::uint32_t layer = 0; layer < layers && layer < kStageLayers; ++layer) {
      for (unsigned p = 0; p < kPoints; ++p) {
        Name name = kDecodeName[p];
        if (name == kHcAttn && static_cast<int>(layer) == ple_layer)
          name = kHcAttnPle;
        add(slots[layer * kPoints + p], name);
      }
      for (unsigned p = 0; p < kPfPoints; ++p)
        add(slots[kStageSlots + layer * kPfPoints + p], kPfName[p]);
    }
    add(slots[kStageExtra + 3], kHead);
    std::sort(stamps.begin(), stamps.end(),
              [](const Stamp& a, const Stamp& b) { return a.tick < b.tick; });
    std::array<double, kNames> ms{};
    std::array<unsigned, kNames> count{};
    std::uint64_t previous = begin;
    for (const auto& s : stamps) {
      ms[s.name] += double(s.tick - previous) * ns_per_tick_ * 1e-6;
      ++count[s.name];
      previous = s.tick;
    }
    const double stamp_ms =
        stamps.empty() ? 0.0 : double(previous - begin) * ns_per_tick_ * 1e-6;
    std::string line;
    char buf[96];
    std::snprintf(buf, sizeof buf, "[prefill-stages] tokens=%u start=%u host_ms=%.3f stamp_ms=%.3f",
                  tokens, start, host_ms, stamp_ms);
    line = buf;
    for (unsigned i = 0; i < kNames; ++i) {
      std::snprintf(buf, sizeof buf, " %s=%.3f", kLabel[i], ms[i]);
      line += buf;
    }
    line += " cnt=";
    for (unsigned i = 0; i < kNames; ++i) {
      std::snprintf(buf, sizeof buf, "%s%u", i ? "," : "", count[i]);
      line += buf;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fill_n(stage_stamps_, kStageSlots + kPfSlots, std::uint64_t{0});
  }
  // Host metadata is set for every pass, including graph replay. It does not
  // choose stamp addresses or add captured operations.
  void PrepareStages(std::uint32_t first_body_layer, std::uint32_t layers) {
    first_body_layer_ = first_body_layer;
    stage_layers_ = layers;
  }
  // Called after the pass's done-flag wait (or stream synchronization).
  // Coherent pages make the graph's preceding writes visible at this point.
  void CollectStages(std::uint32_t width) {
    if (!StagesEnabled())
      return;
    const auto* slots = static_cast<volatile std::uint64_t*>(stage_stamps_);
    const std::uint64_t prefix = slots[kStageExtra];
    const std::uint64_t body = slots[kStageExtra + 1];
    for (unsigned phase = 0; phase < 2; ++phase) {
      if (phase == 0 && first_body_layer_ == 0)
        continue;
      const auto begin = phase == 0 ? prefix : body;
      if (!begin)
        continue;
      auto& a = stages_[{phase, std::min(width, 16u)}];
      ++a.n;
      auto previous = begin;
      const auto belongs = [&](std::uint64_t tick) {
        return tick >= begin && (phase != 0 || !body || tick < body);
      };
      const auto add = [&](std::uint64_t tick, unsigned stage) {
        if (!belongs(tick)) {
          previous = 0;
          return;
        }
        if (previous && tick >= previous)
          a.ms[stage] += double(tick - previous) * ns_per_tick_ * 1e-6;
        previous = tick;
      };
      const auto embedding = slots[kStageExtra + 2];
      if (belongs(embedding))
        add(embedding, kPoints);
      const auto first_layer = phase == 0 ? 0u : first_body_layer_;
      const auto end_layer = phase == 0 ? first_body_layer_ : stage_layers_;
      for (unsigned layer = first_layer; layer < end_layer; ++layer) {
        for (unsigned point = 0; point < kPoints; ++point)
          add(slots[layer * kPoints + point], point);
      }
      const auto head = slots[kStageExtra + 3];
      if (belongs(head))
        add(head, kPoints + 1);
    }
    std::fill_n(stage_stamps_, kStageSlots, std::uint64_t{0});
  }
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
    stages_.clear();
    next_slot_ = 0;
    window_start_ = Now();
  }

private:
  static constexpr std::size_t kSlots = 1 << 18;
  static constexpr std::size_t kStageExtra = kStageLayers * kPoints;
  static constexpr std::size_t kStageSlots = kStageExtra + 4;
  static constexpr std::size_t kPfSlots = kStageLayers * kPfPoints;
  struct StageAcc {
    std::array<double, kPoints + 2> ms{};
    std::uint64_t n{0};
  };
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
    const char* stages = std::getenv("GUFO_TRACE_STAGES");
    if (v[0] == '1' && v[1] == '\0' && stages && stages[0] == '1' &&
        stages[1] == '\0') {
      void* region = nullptr;
      if (hipHostMalloc(&region, (kStageSlots + kPfSlots) * sizeof(std::uint64_t),
                        hipHostMallocCoherent) == hipSuccess) {
        stage_stamps_ = static_cast<std::uint64_t*>(region);
        std::fill_n(stage_stamps_, kStageSlots + kPfSlots, std::uint64_t{0});
      }
    }
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
    static constexpr const char* names[] = {
        "hc_attn",          "mixer",     "combine_hc_ffn",
        "router",           "shared",    "routed",
        "combine_epilogue", "embedding", "head"};
    for (const auto& [key, a] : stages_) {
      const auto [phase, width] = key;
      const std::string name =
          std::string(phase == 0 ? "verify-prefix w" : "verify-body w") +
          std::to_string(width);
      const auto found = busy.find(name);
      if (!a.n || found == busy.end() || !found->second.n)
        continue;
      line.clear();
      double sum = 0;
      for (unsigned i = 0; i < a.ms.size(); ++i) {
        const double ms = a.ms[i] / a.n;
        sum += ms;
        std::snprintf(item, sizeof(item), " %s %.3f", names[i], ms);
        line += item;
      }
      const double body_ms = found->second.ms / found->second.n;
      std::fprintf(stderr,
                   "qwen38_flash_next: stages %s %llux:%s sum %.3f body %.3f "
                   "launch+tail %.3f\n",
                   name.c_str(), static_cast<unsigned long long>(a.n),
                   line.c_str(), sum, body_ms, body_ms - sum);
    }
  }

  bool enabled_{false};
  std::uint64_t* stage_stamps_{nullptr};
  std::uint32_t first_body_layer_{0}, stage_layers_{0};
  std::map<std::pair<unsigned, std::uint32_t>, StageAcc> stages_;
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
