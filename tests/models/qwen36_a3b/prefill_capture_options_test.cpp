#include "prefill_capture_options.hpp"

#include <stdexcept>
#include <string>
#include <vector>

CaptureOptions Parse(std::initializer_list<const char*> args) {
  std::vector<const char*> argv{"capture", "model", "prompt", "out"};
  argv.insert(argv.end(), args.begin(), args.end());
  return ParseCaptureOptions(static_cast<int>(argv.size()), argv.data());
}
int main() {
  const auto defaults = Parse({});
  if (defaults.chunk != 2048 || !defaults.audit.empty() ||
      defaults.tokens != 32768)
    throw std::runtime_error("capture defaults changed");
  for (auto audit : {"lean", "full"}) {
    const auto options =
        Parse({"--chunk", "0", "--audit", audit, "--prompt-tokens", "9"});
    if (options.chunk != 0 || options.audit != audit || options.tokens != 9)
      throw std::runtime_error("explicit options ignored");
  }
  for (const auto& args : {std::vector<const char*>{"--chunk", "-1"},
                           {"--chunk", "1x"},
                           {"--chunk", "4294967296"},
                           {"--audit", "bad"},
                           {"--audit"},
                           {"--prompt-tokens", "0"},
                           {"--unknown", "1"}}) {
    std::vector<const char*> argv{"capture", "model", "prompt", "out"};
    argv.insert(argv.end(), args.begin(), args.end());
    bool failed = false;
    try {
      ParseCaptureOptions(static_cast<int>(argv.size()), argv.data());
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    if (!failed)
      throw std::runtime_error("bad option accepted");
  }
}
