#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

#include "src/core/behaviour.hpp"

// Local-only executable: fixture contents and paths are supplied externally.
// No CTest registration; exit 77 denotes a skip without GUFO_LOOPFIX_IDS.
int main() {
  const char* path = std::getenv("GUFO_LOOPFIX_IDS");
  if (!path || !*path) {
    std::cout << "SKIPPED: set GUFO_LOOPFIX_IDS to a token-id file\n";
    return 77;
  }
  std::ifstream input(path);
  if (!input) {
    std::cerr << "Cannot open GUFO_LOOPFIX_IDS file\n";
    return 1;
  }
  using Breaker = gufo::sampling::LoopBreaker;
  Breaker breaker;
  std::cout << "channel=THINK max_period=" << breaker.cfg.think_max_period
            << '\n';
  uint64_t index = 0;
  std::string word;
  while (input >> word) {
    int32_t token = 0;
    const auto result =
        std::from_chars(word.data(), word.data() + word.size(), token);
    if (result.ec != std::errc{} || result.ptr != word.data() + word.size() ||
        token < 0) {
      std::cerr << "Invalid token id at index " << index << '\n';
      return 1;
    }
    const auto event = breaker.accept(token, Breaker::THINK);
    if (event.what != Breaker::NONE) {
      std::cout << (event.what == Breaker::NUDGE ? "NUDGE" : "FORCE_THINK")
                << " index=" << index << " period=" << event.period
                << " repeats=" << event.repeats << '\n';
    }
    ++index;
  }
  if (input.bad() || !input.eof() || index == 0) {
    std::cerr << "Unreadable or empty token-id file\n";
    return 1;
  }
  if (breaker.detections == 0)
    std::cout << "NO DETECTION\n";
  std::cout << "tokens=" << index << " detections=" << breaker.detections
            << '\n';
  return 0;
}
