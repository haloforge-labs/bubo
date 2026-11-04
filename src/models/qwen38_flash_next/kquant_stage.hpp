// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace gufo::models::qwen38_flash_next::dense_gemv {

// Bench-only ablation stages of the K-quant matrix-unit kernel. They are not
// dense_gemv::Variant values and never reach the production selector. Every
// stage keeps the kernel's grid, block, static LDS, launch bounds, K loop and
// output epilogue; each rung adds work and ends in a sink that depends on it.
enum KQuantStage : int {
  kStageLoads = 1,       // A: Fragment loads + decode ALU, register sink
  kStageLoadsNoBarrier,  // A0: A without the two barriers per K256
  kStageLds,             // B: A + decode writes to the LDS tile, read-back
  kStageLdsNoBarrier,    // B0: B without the two barriers
  kStageActivation,      // C1: B + activation block loads and sums
  kStageWmma,            // C2: C1 + matrix-unit products
  kStageShuffle,         // C3: C2 + the scale/min fan-out
  kStageFull             // D: the production kernel
};
inline constexpr int kStageCount = kStageFull;

constexpr const char* KQuantStageName(int stage) {
  switch (stage) {
    case kStageLoads:
      return "A";
    case kStageLoadsNoBarrier:
      return "A0";
    case kStageLds:
      return "B";
    case kStageLdsNoBarrier:
      return "B0";
    case kStageActivation:
      return "C1";
    case kStageWmma:
      return "C2";
    case kStageShuffle:
      return "C3";
    case kStageFull:
      return "D";
    default:
      return nullptr;
  }
}

// -1 for an unknown name.
constexpr int ParseKQuantStage(const char* text) {
  if (!text)
    return -1;
  for (int s = 1; s <= kStageCount; ++s) {
    const char* name = KQuantStageName(s);
    int i = 0;
    while (text[i] && text[i] == name[i])
      ++i;
    if (text[i] == name[i])
      return s;
  }
  return -1;
}

}  // namespace gufo::models::qwen38_flash_next::dense_gemv
