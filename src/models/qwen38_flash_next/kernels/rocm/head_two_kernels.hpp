#pragma once

#include <hip/hip_runtime.h>

#include "src/models/qwen38_flash_next/head_two.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

struct HeadTwoBuffers {
  float* scores;
  HeadTwoPair* sort[2];
  std::int32_t* ids;
  std::int32_t* extras;
  float* exact;
  HeadTwoResult* result;
};

// q8 is the engine's padded Q8_1 activation, including its stored half scale.
bool HeadTwoPre(const void* cheap, const void* q8, int width, int rows,
                int cols, int shortlist, HeadTwoBuffers b, hipStream_t stream,
                int row_rotation = 0, int rescore = 0);
bool HeadTwoExtract(float* full, int width, int rows, int shortlist,
                    HeadTwoBuffers b, hipStream_t stream, bool extract = true);
bool HeadTwoCheck(const float* full, int width, int rows, int shortlist,
                  HeadTwoBuffers b, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm
