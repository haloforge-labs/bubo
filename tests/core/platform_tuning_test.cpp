#include <cstdlib>
#include <iostream>
#include <string>

#include "src/core/platform/tuning.hpp"

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

gufo::platform::Tuning Parse(const char* spec) {
  if (spec == nullptr) {
    ::unsetenv("GUFO_PLATFORM_TUNING");
  } else {
    ::setenv("GUFO_PLATFORM_TUNING", spec, 1);
  }
  return gufo::platform::detail::ParseTuning();
}

}  // namespace

int main() {
  constexpr bool platform_default = gufo::platform::detail::kWindowsDefault;
  Expect(Parse(nullptr).prompt_checkpoint == platform_default,
         "without the variable every switch has its platform default");
  Expect(Parse("").prompt_checkpoint == platform_default,
         "an empty variable keeps the defaults");
  Expect(Parse("+prompt_checkpoint").prompt_checkpoint, "+name enables");
  Expect(Parse("prompt_checkpoint").prompt_checkpoint, "a bare name enables");
  Expect(!Parse("-prompt_checkpoint").prompt_checkpoint, "-name disables");
  Expect(Parse("none, +prompt_checkpoint").prompt_checkpoint,
         "items apply left to right, spaces are ignored");
  Expect(!Parse("all,-prompt_checkpoint").prompt_checkpoint,
         "a later item overrides all");
  Expect(Parse("all").prompt_checkpoint, "all enables every switch");
  Expect(!Parse("none").prompt_checkpoint, "none disables every switch");
  Expect(Parse("bogus").prompt_checkpoint == platform_default,
         "an unknown name changes nothing");
  const auto mixed = Parse("none,+fast_sampling");
  Expect(mixed.fast_sampling && !mixed.flag_waits && !mixed.prompt_checkpoint,
         "switches are independent");
  const auto all = Parse("all");
  Expect(all.copy_kernels && all.flag_waits && all.flush_before_wait &&
             all.recorded_rollback && all.keep_rollback_rows &&
             all.fused_hc_down && all.fast_sampling &&
             all.verify_graph_candidates && all.hot_first_upload,
         "all reaches every switch");
  if (failures == 0) {
    std::cout << "platform_tuning_test: ok\n";
  }
  return failures == 0 ? 0 : 1;
}
