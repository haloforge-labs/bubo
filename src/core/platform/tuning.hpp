#ifndef GUFO_CORE_PLATFORM_TUNING_HPP_
#define GUFO_CORE_PLATFORM_TUNING_HPP_

// Behaviour changes validated on Windows only.
//
// Each switch is on by default on Windows and off elsewhere, so a Linux build
// runs the code it ran before the Windows port. None of them changes model
// arithmetic: outputs are bit-identical either way (checked with
// `gufo bench --logit-eval`); they change timing, memory placement or cache
// reuse. GUFO_PLATFORM_TUNING overrides the defaults on any platform, which is
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
