// SPDX-License-Identifier: MIT
// Few-row dense dot4 projection, width 5 (one width per translation unit so
// the unrolled kernels build in parallel). Kernel: dense_dot4.hpp.
#include "dense_dot4.hpp"

template bool gufo::models::qwen38_flash_next::dense_dot4::Launch<5>(
    int, const void*, const void*, float*, int, int, int, hipStream_t);
