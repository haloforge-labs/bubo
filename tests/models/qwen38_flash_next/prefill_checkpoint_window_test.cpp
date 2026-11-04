// The in-pass checkpoint window must hold a chat prompt's stable boundary after
// it is rounded down to the attention query group. A boundary the assistant
// framing (about seven rows) before the prompt end, rounded down by up to three
// rows, lay nine or ten rows before the end and outside the old window of eight:
// the pass was split there and cost an extra weight sweep (cold 2K prompts
// -5.6 %).
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "src/models/qwen38_flash_next/config.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
// Rows between the aligned stable boundary and the prompt end.
std::uint32_t Suffix(std::uint32_t prompt, std::uint32_t framing) {
  const std::uint32_t stable = prompt - framing;
  const std::uint32_t aligned = stable - stable % 4;
  return prompt - aligned;
}
}  // namespace

int main() {
  try {
    constexpr std::uint32_t kOldWindow = 8;
    Require(q::kPrefillCheckpointSuffix >= kOldWindow, "window shrank");
    // Every residue of the prompt length, with the framings chat templates use.
    bool old_window_missed = false;
    for (std::uint32_t framing = 4; framing <= 9; ++framing) {
      for (std::uint32_t prompt = 1996; prompt < 2004; ++prompt) {
        const auto rows = Suffix(prompt, framing);
        Require(rows <= q::kPrefillCheckpointSuffix,
                "an aligned stable boundary lies outside the in-pass window");
        old_window_missed |= rows > kOldWindow;
      }
    }
    Require(old_window_missed,
            "the old window of eight would have held every case: the test no "
            "longer shows what the wider window is for");
    // The measured cases: prompts of 1998, 1997 and 2182 tokens, framing 7.
    Require(Suffix(1998, 7) == 10 && Suffix(1997, 7) == 9 &&
                Suffix(2182, 7) == 10,
            "the measured suffixes changed");
    std::cout << "prefill checkpoint window tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
