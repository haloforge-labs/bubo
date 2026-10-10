#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/core/behaviour.hpp"

namespace {
using Breaker = gufo::sampling::LoopBreaker;

#define CHECK(condition)                                             \
  do {                                                               \
    if (!(condition))                                                \
      throw std::runtime_error(std::string(#condition) +             \
                               " line=" + std::to_string(__LINE__)); \
  } while (false)

struct Expected {
  int period;
  int think_nudge;
  int think_force;
  int answer_nudge;
  int answer_stop;
};

// Zero-based indices from the external reference, THINK cap 128 / ANSWER
// cap 16.
constexpr std::array<Expected, 7> kExpected{{
    {1, 127, 143, 511, 527},
    {8, 127, 143, 511, 527},
    {16, 255, 271, 511, 527},
    {19, 303, 319, -1, -1},
    {44, 703, 719, -1, -1},
    {64, 1023, 1039, -1, -1},
    {128, 2047, 2063, -1, -1},
}};

void CheckPattern(Breaker& breaker, int period, Breaker::channel channel,
                  int nudge, int terminal) {
  const auto terminal_action =
      channel == Breaker::THINK ? Breaker::FORCE_THINK : Breaker::STOP_ANSWER;
  for (int i = 0; i < 8192; ++i) {
    const auto event = breaker.accept(1000 + i % period, channel);
    const auto expected = i == nudge      ? Breaker::NUDGE
                          : i == terminal ? terminal_action
                                          : Breaker::NONE;
    CHECK(event.what == expected);
    if (expected != Breaker::NONE) {
      CHECK(event.where == channel);
      CHECK(event.period == period);
      CHECK(event.repeats == (i + 1) / period);
    }
    if (i == nudge) {
      CHECK(breaker.nudge_left == 16);
      CHECK(breaker.watch_left == 16 + 256);
      CHECK(!breaker.terminal);
    }
    if (nudge >= 0 && i > nudge && i < terminal) {
      CHECK(breaker.nudging());
      CHECK(!breaker.terminal);
    }
    if (terminal >= 0 && i >= terminal) {
      CHECK(breaker.terminal);
      CHECK(breaker.nudge_left == 0);
      CHECK(breaker.watch_left == 0);
    }
  }
  if (nudge < 0) {
    CHECK(!breaker.terminal);
    CHECK(breaker.take_event().what == Breaker::NONE);
    return;
  }
  CHECK(breaker.take_event().what == Breaker::NUDGE);
  CHECK(breaker.take_event().what == terminal_action);
  CHECK(breaker.take_event().what == Breaker::NONE);
}

void ExactPeriods() {
  for (const auto& row : kExpected) {
    for (const auto channel : {Breaker::THINK, Breaker::ANSWER}) {
      Breaker breaker;
      CHECK(breaker.cfg.think_repeats == 16);
      CHECK(breaker.cfg.think_tokens == 128);
      CHECK(breaker.cfg.answer_repeats == 32);
      CHECK(breaker.cfg.answer_tokens == 512);
      CHECK(breaker.tail.size() == 128);
      CHECK(breaker.matching.size() == 128);
      CHECK(breaker.pending.size() == 16);
      CHECK(breaker.cfg.think_max_period == 128);
      CHECK(breaker.cfg.answer_max_period == 16);
      CheckPattern(
          breaker, row.period, channel,
          channel == Breaker::THINK ? row.think_nudge : row.answer_nudge,
          channel == Breaker::THINK ? row.think_force : row.answer_stop);
      const bool detected = channel == Breaker::THINK || row.period <= 16;
      CHECK(breaker.detections == (detected ? 2u : 0u));
      CHECK(breaker.nudges == (detected ? 1u : 0u));
      CHECK(breaker.forced_thinks == (channel == Breaker::THINK ? 1u : 0u));
      CHECK(breaker.stopped_answers ==
            (channel == Breaker::ANSWER && detected ? 1u : 0u));
    }
  }
}

void BeyondMaxPeriod() {
  for (const auto channel : {Breaker::THINK, Breaker::ANSWER}) {
    Breaker breaker;
    for (int i = 0; i < 8192; ++i)
      CHECK(breaker.accept(1000 + i % 129, channel).what == Breaker::NONE);
    CHECK(breaker.detections == 0);
    CHECK(!breaker.terminal);
    CHECK(breaker.take_event().what == Breaker::NONE);
  }
}

void RandomStreams() {
  // Explicit xorshift32 avoids implementation-dependent distribution output.
  for (const uint32_t seed : {1u, 42u, 0x12345678u, 0xDEADBEEFu}) {
    for (const auto channel : {Breaker::THINK, Breaker::ANSWER}) {
      Breaker breaker;
      uint32_t state = seed;
      for (int i = 0; i < 20000; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        CHECK(breaker.accept(static_cast<int32_t>(state & 0x7FFFFFFFu), channel)
                  .what == Breaker::NONE);
      }
      CHECK(breaker.detections == 0);
      CHECK(breaker.take_event().what == Breaker::NONE);
    }
  }
}

void ChannelSwitches() {
  for (const auto from : {Breaker::THINK, Breaker::ANSWER}) {
    const auto to = from == Breaker::THINK ? Breaker::ANSWER : Breaker::THINK;
    const int nudge = from == Breaker::THINK ? 255 : 511;
    // Switch just before detection, during its watch, and after termination.
    for (const int prefix : {nudge, nudge + 1, nudge + 17}) {
      Breaker breaker;
      for (int i = 0; i < prefix; ++i)
        breaker.accept(1000 + i % 16, from);
      while (breaker.take_event().what != Breaker::NONE) {
      }
      const auto detections = breaker.detections;
      CheckPattern(breaker, 16, to, to == Breaker::THINK ? 255 : 511,
                   to == Breaker::THINK ? 271 : 527);
      CHECK(breaker.current == to);
      CHECK(breaker.detections == detections + 2);
    }
  }
}

void BoundaryAndReset() {
  Breaker breaker;
  CheckPattern(breaker, 128, Breaker::THINK, 2047, 2063);
  breaker.boundary(Breaker::ANSWER);
  CHECK(breaker.count == 0);
  CHECK(breaker.current == Breaker::ANSWER);
  CHECK(breaker.nudge_left == 0);
  CHECK(breaker.watch_left == 0);
  CHECK(breaker.nudges_in_span == 0);
  CHECK(!breaker.terminal);
  CHECK(breaker.detections == 2);
  CHECK(std::all_of(breaker.tail.begin(), breaker.tail.end(),
                    [](int32_t token) { return token == 0; }));
  CHECK(std::all_of(breaker.matching.begin(), breaker.matching.end(),
                    [](uint32_t run) { return run == 0; }));
  CheckPattern(breaker, 16, Breaker::ANSWER, 511, 527);
  breaker.cfg = {3, 7, 5, 11};
  breaker.reset();
  CHECK(breaker.cfg.think_repeats == 3);
  CHECK(breaker.cfg.think_tokens == 7);
  CHECK(breaker.cfg.answer_repeats == 5);
  CHECK(breaker.cfg.answer_tokens == 11);
  CHECK(breaker.count == 0);
  CHECK(breaker.current == Breaker::ANSWER);
  CHECK(breaker.nudge_left == 0);
  CHECK(breaker.watch_left == 0);
  CHECK(breaker.nudges_in_span == 0);
  CHECK(!breaker.terminal);
  CHECK(breaker.detections == 0);
  CHECK(breaker.nudges == 0);
  CHECK(breaker.forced_thinks == 0);
  CHECK(breaker.stopped_answers == 0);
  CHECK(breaker.pending_head == 0);
  CHECK(breaker.pending_size == 0);
  CHECK(std::all_of(breaker.tail.begin(), breaker.tail.end(),
                    [](int32_t token) { return token == 0; }));
  CHECK(std::all_of(breaker.matching.begin(), breaker.matching.end(),
                    [](uint32_t run) { return run == 0; }));
  breaker.cfg = {};
  CheckPattern(breaker, 128, Breaker::THINK, 2047, 2063);
}
void DisturbedNudge() {
  for (const auto channel : {Breaker::THINK, Breaker::ANSWER}) {
    const int period = channel == Breaker::THINK ? 44 : 8;
    const int nudge = channel == Breaker::THINK ? 703 : 511;
    const int terminal = channel == Breaker::THINK ? 1412 : 1028;
    Breaker breaker;
    for (int i = 0; i < 8192; ++i) {
      const auto event =
          breaker.accept(i == nudge + 5 ? 900000 : 1000 + i % period, channel);
      CHECK(event.what == (i == nudge      ? Breaker::NUDGE
                           : i == terminal ? (channel == Breaker::THINK
                                                  ? Breaker::FORCE_THINK
                                                  : Breaker::STOP_ANSWER)
                                           : Breaker::NONE));
      if (i == terminal)
        CHECK(breaker.terminal && breaker.nudges_in_span == 1);
    }
    CHECK(breaker.detections == 2 && breaker.nudges == 1);
  }
}

void NearExactCounter() {
  // One incrementing ID per otherwise fixed block: explicitly outside the exact
  // detector's scope.
  for (const auto channel : {Breaker::THINK, Breaker::ANSWER}) {
    Breaker breaker;
    for (int i = 0; i < 8192; ++i) {
      const int token = i % 44 == 43 ? 100000 + i / 44 : 1000 + i % 44;
      CHECK(breaker.accept(token, channel).what == Breaker::NONE);
    }
    CHECK(breaker.detections == 0);
  }
}
}  // namespace

int main() {
  try {
    ExactPeriods();
    BeyondMaxPeriod();
    RandomStreams();
    ChannelSwitches();
    BoundaryAndReset();
    DisturbedNudge();
    NearExactCounter();
    std::cout << "loop_breaker_test: 7 groups PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "loop_breaker_test: " << error.what() << '\n';
    return 1;
  }
}
