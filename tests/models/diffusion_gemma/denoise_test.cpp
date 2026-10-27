// DiffusionGemma host sampler decisions: temperature schedule, entropy-bound
// acceptance, block convergence and reply length (CPU only).
#include "src/models/diffusion_gemma/denoise.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace dg = gufo::models::diffusion_gemma;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void Temperature() {
  const dg::DiffusionDefaults d;  // 48 steps, t 0.8 -> 0.4.
  Expect(std::abs(dg::StepTemperature(d, 48) - 0.8F) < 1e-6F,
         "first step at t_max");
  Expect(std::abs(dg::StepTemperature(d, 24) - 0.6F) < 1e-6F,
         "halfway temperature");
  Expect(dg::StepTemperature(d, 1) > d.t_min, "last step above t_min");
}

void Acceptance() {
  // Sorted: 0.01 (0 before), 0.02 (0.01 before), 0.2 (0.03 before) accepted;
  // 0.5 has 0.23 before it.
  const std::vector<float> entropy = {0.5F, 0.01F, 0.02F, 0.2F};
  Expect(dg::AcceptRows(entropy, 0.1F) ==
             std::vector<bool>({false, true, true, true}),
         "entropy-bound acceptance");
  const std::vector<float> uncertain = {3.0F, 2.0F, 5.0F};
  Expect(dg::AcceptRows(uncertain, 0.1F) ==
             std::vector<bool>({false, true, false}),
         "the most certain row is always accepted");
  const std::vector<float> ties = {1.0F, 1.0F, 1.0F};
  Expect(dg::AcceptRows(ties, 0.5F) == std::vector<bool>({true, false, false}),
         "ties keep row order");
}

void Convergence() {
  dg::DiffusionDefaults d;  // stability on, confidence 0.005.
  const std::vector<std::int32_t> a = {1, 2, 3}, b = {1, 2, 4};
  const std::vector<float> sure = {0.001F, 0.002F, 0.003F};
  const std::vector<float> unsure = {0.001F, 0.002F, 0.3F};
  Expect(!dg::BlockConverged(d, {}, a, sure), "first step never stable");
  Expect(dg::BlockConverged(d, a, a, sure), "stable and confident");
  Expect(!dg::BlockConverged(d, a, b, sure), "changed argmax");
  Expect(!dg::BlockConverged(d, a, a, unsure), "mean entropy too high");
  d.stability_threshold = 0;
  Expect(dg::BlockConverged(d, {}, a, sure), "stability check disabled");
}

void Reply() {
  const std::vector<int> eos = {1, 106};
  const std::vector<std::int32_t> none = {5, 6, 7};
  const std::vector<std::int32_t> stop = {5, 6, 106, 7, 1};
  auto reply = dg::TakeReply(none, eos, 10);
  Expect(reply.tokens == 3 && !reply.end_of_sequence, "no end token");
  reply = dg::TakeReply(stop, eos, 10);
  Expect(reply.tokens == 2 && reply.end_of_sequence,
         "stops at the first end token");
  reply = dg::TakeReply(stop, eos, 2);
  Expect(reply.tokens == 2 && reply.end_of_sequence,
         "end token right after the limit still ends the reply");
  reply = dg::TakeReply(stop, eos, 1);
  Expect(reply.tokens == 1 && !reply.end_of_sequence,
         "a limit before the end token cuts the reply");
}

void Seeds() {
  // SplitMix64's first output for state 0.
  Expect(dg::MixSeed(0) == 0xe220a8397b1dcdafULL, "SplitMix64 finaliser");
  Expect(dg::MixSeed(1) != dg::MixSeed(2), "distinct seeds");
}

}  // namespace

int main() {
  Temperature();
  Acceptance();
  Convergence();
  Reply();
  Seeds();
  if (failures == 0)
    std::printf("diffusion_gemma_denoise_test: OK\n");
  return failures == 0 ? 0 : 1;
}
