#ifndef GUFO_CORE_PLATFORM_TUNING_HPP_
#define GUFO_CORE_PLATFORM_TUNING_HPP_

// Behaviour changes validated on Windows only.
//
// Each switch is on by default on Windows and off elsewhere, so a Linux build
// runs the code it ran before the Windows port. The decode switches change
// timing and memory placement only: outputs are bit-identical either way
// (`gufo bench --logit-eval` dumps and fixed-seed sampled texts match).
// prompt_checkpoint also splits prefill at the generation suffix, which
// changes rounding the way a different prefill chunk size does, so sampled
// texts differ from an unsplit prefill (as equally valid samples).
// GUFO_PLATFORM_TUNING overrides the defaults on any platform, which is
// how a switch is tried on Linux or the Linux path is exercised on Windows:
//
//   GUFO_PLATFORM_TUNING=+prompt_checkpoint,-hot_first_upload
//   GUFO_PLATFORM_TUNING=all        (every switch on)
//   GUFO_PLATFORM_TUNING=none       (every switch off)
//
// Items apply left to right; unknown names are reported once on stderr.

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace gufo::platform {

struct Tuning {
  /// Qwen serving: checkpoint every prompt before the generation suffix, so a
  /// client that re-sends the previous turn in another form (without its
  /// reasoning, reformatted) still hits the cache.
  bool prompt_checkpoint;
  /// Qwen3.8-Flash-Next decode (all bit-identical outputs):
  /// Small host<->device and device<->device copies as kernels instead of
  /// hipMemcpyAsync (a copy-engine hand-off per graph node on Windows).
  bool copy_kernels;
  /// Decode-sized passes wait on a completion flag in coherent pinned memory
  /// instead of hipStreamSynchronize (~0.4 ms sooner after a large graph).
  bool flag_waits;
  /// Submit queued launches (hipStreamQuery) before blocking on n-gram rows:
  /// HIP on Windows batches launches until a flush.
  bool flush_before_wait;
  /// Replay the speculative state rollback as one recorded graph per kept
  /// length instead of ~2 launches per recurrent layer.
  bool recorded_rollback;
  /// Keep rollback rows across session resets and snapshot restores, so the
  /// graphs that point at them stay valid.
  bool keep_rollback_rows;
  /// HC mixer down projection: SiluScale fused into the GEMV write and a
  /// deeper load prefetch for 2-8 tokens.
  bool fused_hc_down;
  /// Sampled decode reads GPU-selected top-64 candidate lists of each row
  /// (SamplerState::DistributionFromTop) instead of full host rows, whenever
  /// the list provably holds the whole top-k.
  bool fast_sampling;
  /// With fast_sampling: select the verify rows' candidate lists inside the
  /// verify graph instead of in a separate pass after it.
  bool verify_graph_candidates;
  /// Upload the weights read in full on every token before the routed
  /// experts, so they get the dedicated carve-out when memory runs short.
  bool hot_first_upload;
};

namespace detail {

#ifdef _WIN32
inline constexpr bool kWindowsDefault = true;
#else
inline constexpr bool kWindowsDefault = false;
#endif

struct TuningField {
  std::string_view name;
  bool Tuning::* member;
};

inline constexpr TuningField kTuningFields[] = {
    {"prompt_checkpoint", &Tuning::prompt_checkpoint},
    {"copy_kernels", &Tuning::copy_kernels},
    {"flag_waits", &Tuning::flag_waits},
    {"flush_before_wait", &Tuning::flush_before_wait},
    {"recorded_rollback", &Tuning::recorded_rollback},
    {"keep_rollback_rows", &Tuning::keep_rollback_rows},
    {"fused_hc_down", &Tuning::fused_hc_down},
    {"fast_sampling", &Tuning::fast_sampling},
    {"verify_graph_candidates", &Tuning::verify_graph_candidates},
    {"hot_first_upload", &Tuning::hot_first_upload},
};

inline Tuning ParseTuning() {
  Tuning tuning{};
  for (const auto& field : kTuningFields)
    tuning.*field.member = kWindowsDefault;
  const char* env = std::getenv("GUFO_PLATFORM_TUNING");
  std::string_view spec = env != nullptr ? env : "";
  while (!spec.empty()) {
    const auto comma = spec.find(',');
    std::string_view item = spec.substr(0, comma);
    spec = comma == std::string_view::npos ? std::string_view{}
                                           : spec.substr(comma + 1);
    while (!item.empty() && item.front() == ' ')
      item.remove_prefix(1);
    while (!item.empty() && item.back() == ' ')
      item.remove_suffix(1);
    if (item.empty())
      continue;
    if (item == "all" || item == "none") {
      for (const auto& field : kTuningFields)
        tuning.*field.member = item == "all";
      continue;
    }
    bool value = true;
    if (item.front() == '+' || item.front() == '-') {
      value = item.front() == '+';
      item.remove_prefix(1);
    }
    bool known = false;
    for (const auto& field : kTuningFields) {
      if (field.name == item) {
        tuning.*field.member = value;
        known = true;
      }
    }
    if (!known) {
      std::fprintf(stderr,
                   "gufo: GUFO_PLATFORM_TUNING: unknown switch '%.*s'\n",
                   static_cast<int>(item.size()), item.data());
    }
  }
  if (env != nullptr) {
    std::fprintf(stderr, "gufo: platform tuning:");
    for (const auto& field : kTuningFields) {
      std::fprintf(stderr, " %s%.*s", tuning.*field.member ? "+" : "-",
                   static_cast<int>(field.name.size()), field.name.data());
    }
    std::fprintf(stderr, "\n");
  }
  return tuning;
}

}  // namespace detail

/// The process-wide switches, read from the environment once.
inline const Tuning& PlatformTuning() {
  static const Tuning tuning = detail::ParseTuning();
  return tuning;
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_TUNING_HPP_
