// SPDX-License-Identifier: MIT
// Host declarations of the few-row dense dot4 projection (dense_dot4.hpp).
// One width per translation unit (dense_dot4_w1..8.hip.cpp); this header has
// no device code, so the C++17 adapter can include it.
#pragma once

#include <hip/hip_runtime.h>

namespace gufo::models::qwen38_flash_next::dense_dot4 {

// y[t * m + row] for t < Width. x points to Width rows of block_q8_1, the row
// stride is x_block_stride blocks. Returns false (nothing launched) when the
// kernel does not cover (type, k, m): the caller keeps its existing route.
template<int Width>
bool Launch(int type, const void* w, const void* x, float* y, int m, int k,
            int x_block_stride, hipStream_t stream);

// Width 1..8 (else false). Defined in dense_dot4_dispatch.hip.cpp, which sees
// only these declarations: a width unit that saw a call to every Launch<N>
// would instantiate all eight widths itself.
bool Dispatch(int width, int type, const void* w, const void* x, float* y,
              int m, int k, int x_block_stride, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::dense_dot4
