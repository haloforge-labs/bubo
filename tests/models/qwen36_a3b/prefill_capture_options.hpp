#ifndef GUFO_TESTS_MODELS_QWEN36_A3B_PREFILL_CAPTURE_OPTIONS_HPP_
#define GUFO_TESTS_MODELS_QWEN36_A3B_PREFILL_CAPTURE_OPTIONS_HPP_

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

inline constexpr const char* kCaptureUsage =
    "usage: prefill_capture MODEL PROMPT_UTF8 OUTPUT_PREFIX "
    "[--chunk N] [--audit lean|full] [--prompt-tokens N]";
struct CaptureOptions {
  std::uint32_t chunk{2048};
  std::uint32_t tokens{32768};
  std::string audit;
};
inline CaptureOptions ParseCaptureOptions(int argc, const char* const* argv) {
  if (argc < 4)
    throw std::invalid_argument(kCaptureUsage);
  CaptureOptions result;
  for (int i = 4; i < argc; i += 2) {
    if (i + 1 == argc)
      throw std::invalid_argument(kCaptureUsage);
    const std::string_view key(argv[i]), value(argv[i + 1]);
    if (key == "--audit") {
      if (value != "lean" && value != "full")
        throw std::invalid_argument("audit must be lean or full");
      result.audit = value;
    } else if (key == "--chunk" || key == "--prompt-tokens") {
      std::uint32_t number{};
      const auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), number);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() ||
          (key == "--prompt-tokens" && number == 0))
        throw std::invalid_argument("invalid capture row count");
      if (key == "--chunk")
        result.chunk = number;
      else
        result.tokens = number;
    } else {
      throw std::invalid_argument(kCaptureUsage);
    }
  }
  return result;
}
#endif
