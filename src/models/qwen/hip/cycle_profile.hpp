#ifndef GUFO_MODELS_QWEN_HIP_CYCLE_PROFILE_HPP_
#define GUFO_MODELS_QWEN_HIP_CYCLE_PROFILE_HPP_

// Qwen3.8-27B DFlash2 diagnostics, off unless an environment switch is set.
// Single-session only: the counters are process-wide and unsynchronized.
//
//   GUFO_QWEN27_PROFILE=1  every 10 s, per-cycle host wall time of each
//                          phase, split into launch enqueue and GPU wait
//   GUFO_QWEN27_PROFILE=2  also GPU time of the verification pass by part
//                          (events between kernel groups)
//   GUFO_QWEN27_FLUSH=N    submit queued launches (hipStreamQuery) every N
//                          target layers in verification and after every
//                          draft layer; timing only, results are unchanged

#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gufo::hip::qwen27 {

enum class Phase : std::uint8_t {
  kDraft,
  kInject,
  kVerify,
  kVerifyRow,
  kState,
  kCount
};

[[nodiscard]] inline int ProfileLevel() {
  static const int level = [] {
    const char* value = std::getenv("GUFO_QWEN27_PROFILE");
    return value != nullptr ? std::atoi(value) : 0;
  }();
  return level;
}

[[nodiscard]] inline bool ProfileEnabled() { return ProfileLevel() > 0; }

enum class GpuPart : std::uint8_t {
  kAttnProj,
  kAttnCore,
  kSsmProj,
  kSsmCore,
  kFfnGateUp,
  kFfnDown,
  kOther,
  kHead,
  kCount
};

// GPU time of the verification pass, charged by part: each Mark records an
// event, and the time since the previous mark belongs to that part.
class GpuTimeline {
public:
  static GpuTimeline& Get() {
    static GpuTimeline timeline;
    return timeline;
  }
  [[nodiscard]] static bool Enabled() { return ProfileLevel() >= 2; }

  void Begin(hipStream_t stream) {
    count_ = 0;
    Record(stream, GpuPart::kOther);
  }
  void Mark(hipStream_t stream, GpuPart part) { Record(stream, part); }

  // After the stream has synchronized.
  void Collect(std::array<double, static_cast<std::size_t>(GpuPart::kCount)>&
                   totals) {
    for (std::size_t i = 1; i < count_; ++i) {
      float ms = 0.0F;
      if (hipEventElapsedTime(&ms, events_[i - 1], events_[i]) == hipSuccess)
        totals[static_cast<std::size_t>(parts_[i])] += ms;
    }
    count_ = 0;
  }

private:
  void Record(hipStream_t stream, GpuPart part) {
    if (count_ == events_.size()) {
      hipEvent_t event = nullptr;
      if (hipEventCreate(&event) != hipSuccess)
        return;
      events_.push_back(event);
      parts_.push_back(part);
    }
    parts_[count_] = part;
    (void)hipEventRecord(events_[count_], stream);
    ++count_;
  }
  std::vector<hipEvent_t> events_;
  std::vector<GpuPart> parts_;
  std::size_t count_{0};
};

[[nodiscard]] inline std::uint32_t FlushInterval() {
  static const std::uint32_t interval = [] {
    const char* value = std::getenv("GUFO_QWEN27_FLUSH");
    return value != nullptr ? static_cast<std::uint32_t>(std::atoi(value))
                            : 0U;
  }();
  return interval;
}

inline void MaybeFlush(hipStream_t stream, std::uint32_t layer) {
  const auto interval = FlushInterval();
  if (interval != 0 && (layer + 1) % interval == 0)
    (void)hipStreamQuery(stream);
}

struct ProfileState {
  using Clock = std::chrono::steady_clock;
  struct Totals {
    double total_ms{0};
    double enqueue_ms{0};
    std::uint64_t calls{0};
  };
  std::array<Totals, static_cast<std::size_t>(Phase::kCount)> phases{};
  std::array<double, static_cast<std::size_t>(GpuPart::kCount)> gpu{};
  std::uint64_t cycles{0};
  Clock::time_point window_start{};
  Clock::time_point last_cycle{};
  double cycle_ms{0};

  static ProfileState& Get() {
    static ProfileState state;
    return state;
  }

  // Called at the start of each draft block: one block per cycle.
  void CycleStart() {
    const auto now = Clock::now();
    if (last_cycle != Clock::time_point{}) {
      const double gap =
          std::chrono::duration<double, std::milli>(now - last_cycle).count();
      // A gap over a second is a new request, not a decode cycle.
      if (gap < 1000.0) {
        cycle_ms += gap;
        ++cycles;
      }
    }
    last_cycle = now;
    if (window_start == Clock::time_point{})
      window_start = now;
    if (now - window_start >= std::chrono::seconds(10) && cycles > 0)
      Print(now);
  }

  void Print(Clock::time_point now) {
    static constexpr const char* kNames[] = {"draft", "inject", "verify",
                                             "row", "state"};
    const double n = static_cast<double>(cycles);
    double phase_sum = 0;
    std::fprintf(stderr, "qwen27 profile: %llu cycles, %.1f ms/cycle |",
                 static_cast<unsigned long long>(cycles), cycle_ms / n);
    for (std::size_t i = 0; i < phases.size(); ++i) {
      const auto& p = phases[i];
      phase_sum += p.total_ms;
      std::fprintf(stderr, " %s %.1f (enq %.1f) x%.1f", kNames[i],
                   p.total_ms / n, p.enqueue_ms / n,
                   static_cast<double>(p.calls) / n);
    }
    std::fprintf(stderr, " | other %.1f ms\n", (cycle_ms - phase_sum) / n);
    if (GpuTimeline::Enabled()) {
      static constexpr const char* kParts[] = {
          "attn_proj", "attn_core", "ssm_proj", "ssm_core",
          "ffn_gate_up", "ffn_down", "other", "head"};
      double sum = 0;
      std::fprintf(stderr, "qwen27 verify gpu ms/cycle:");
      for (std::size_t i = 0; i < gpu.size(); ++i) {
        sum += gpu[i];
        std::fprintf(stderr, " %s %.1f", kParts[i], gpu[i] / n);
      }
      std::fprintf(stderr, " | total %.1f\n", sum / n);
    }
    gpu = {};
    phases = {};
    cycles = 0;
    cycle_ms = 0;
    window_start = now;
  }
};

// Times one phase call. Enqueued() marks the end of launch submission, just
// before the first call that blocks on the GPU (a pageable download or a
// synchronize); without it the whole call counts as enqueue.
class PhaseScope {
public:
  explicit PhaseScope(Phase phase)
      : phase_(phase), enabled_(ProfileEnabled()) {
    if (enabled_)
      start_ = ProfileState::Clock::now();
  }
  PhaseScope(const PhaseScope&) = delete;
  PhaseScope& operator=(const PhaseScope&) = delete;

  void Enqueued() {
    if (enabled_)
      enqueued_ = ProfileState::Clock::now();
  }

  ~PhaseScope() {
    if (!enabled_)
      return;
    const auto end = ProfileState::Clock::now();
    const auto enqueued =
        enqueued_ == ProfileState::Clock::time_point{} ? end : enqueued_;
    auto& totals =
        ProfileState::Get().phases[static_cast<std::size_t>(phase_)];
    totals.total_ms +=
        std::chrono::duration<double, std::milli>(end - start_).count();
    totals.enqueue_ms +=
        std::chrono::duration<double, std::milli>(enqueued - start_).count();
    ++totals.calls;
  }

private:
  Phase phase_;
  bool enabled_;
  ProfileState::Clock::time_point start_{};
  ProfileState::Clock::time_point enqueued_{};
};

}  // namespace gufo::hip::qwen27

#endif  // GUFO_MODELS_QWEN_HIP_CYCLE_PROFILE_HPP_
