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

// Zero-based indices from the external loop_sim_reference.py simulate(),
// pmax=64, stream token[i]=1000+i%period, with the default channel thresholds.
constexpr std::array<Expected, 5> kExpected{{
    {1, 127, 143, 511, 527},
    {16, 255, 271, 511, 527},
    {17, 271, 287, 543, 559},
    {44, 703, 719, 1407, 1423},
    {64, 1023, 1039, 2047, 2063},
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
    if (i > nudge && i < terminal) {
      CHECK(breaker.nudging());
      CHECK(!breaker.terminal);
    }
    if (i >= terminal) {
      CHECK(breaker.terminal);
      CHECK(breaker.nudge_left == 0);
      CHECK(breaker.watch_left == 0);
    }
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
      CHECK(breaker.tail.size() == 64);
      CHECK(breaker.matching.size() == 64);
      CHECK(breaker.pending.size() == 16);
      CheckPattern(
          breaker, row.period, channel,
          channel == Breaker::THINK ? row.think_nudge : row.answer_nudge,
          channel == Breaker::THINK ? row.think_force : row.answer_stop);
      CHECK(breaker.detections == 2);
      CHECK(breaker.nudges == 1);
      CHECK(breaker.forced_thinks == (channel == Breaker::THINK ? 1u : 0u));
      CHECK(breaker.stopped_answers == (channel == Breaker::ANSWER ? 1u : 0u));
    }
  }
}

void BeyondMaxPeriod() {
  for (const auto channel : {Breaker::THINK, Breaker::ANSWER}) {
    Breaker breaker;
    for (int i = 0; i < 8192; ++i)
      CHECK(breaker.accept(1000 + i % 65, channel).what == Breaker::NONE);
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
    const int nudge = from == Breaker::THINK ? 1023 : 2047;
    // Switch just before detection, during its watch, and after termination.
    for (const int prefix : {nudge, nudge + 1, nudge + 17}) {
      Breaker breaker;
      for (int i = 0; i < prefix; ++i)
        breaker.accept(1000 + i % 64, from);
      while (breaker.take_event().what != Breaker::NONE) {
      }
      const auto detections = breaker.detections;
      CheckPattern(breaker, 64, to, to == Breaker::THINK ? 1023 : 2047,
                   to == Breaker::THINK ? 1039 : 2063);
      CHECK(breaker.current == to);
      CHECK(breaker.detections == detections + 2);
    }
  }
}

void BoundaryAndReset() {
  Breaker breaker;
  CheckPattern(breaker, 64, Breaker::THINK, 1023, 1039);
  breaker.boundary(Breaker::ANSWER);
  CHECK(breaker.count == 0);
  CHECK(breaker.current == Breaker::ANSWER);
  CHECK(breaker.nudge_left == 0);
  CHECK(breaker.watch_left == 0);
  CHECK(!breaker.terminal);
  CHECK(breaker.detections == 2);
  CHECK(std::all_of(breaker.tail.begin(), breaker.tail.end(),
                    [](int32_t token) { return token == 0; }));
  CHECK(std::all_of(breaker.matching.begin(), breaker.matching.end(),
                    [](uint32_t run) { return run == 0; }));
  CheckPattern(breaker, 64, Breaker::ANSWER, 2047, 2063);
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
  CheckPattern(breaker, 64, Breaker::THINK, 1023, 1039);
}
}  // namespace

int main() {
  try {
    ExactPeriods();
    BeyondMaxPeriod();
    RandomStreams();
    ChannelSwitches();
    BoundaryAndReset();
    std::cout << "loop_breaker_test: 5 groups PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "loop_breaker_test: " << error.what() << '\n';
    return 1;
  }
}
