// SPDX-License-Identifier: MIT
// Width dispatch of the few-row dense dot4 projection. Only the declarations
// are visible here; each Launch<N> is instantiated in dense_dot4_w<N>.hip.cpp.
#include "dense_dot4.h"

namespace gufo::models::qwen38_flash_next::dense_dot4 {

bool Dispatch(int width, int type, const void* w, const void* x, float* y,
              int m, int k, int x_block_stride, hipStream_t stream) {
  switch (width) {
    case 1:
      return Launch<1>(type, w, x, y, m, k, x_block_stride, stream);
    case 2:
      return Launch<2>(type, w, x, y, m, k, x_block_stride, stream);
    case 3:
      return Launch<3>(type, w, x, y, m, k, x_block_stride, stream);
    case 4:
      return Launch<4>(type, w, x, y, m, k, x_block_stride, stream);
    case 5:
      return Launch<5>(type, w, x, y, m, k, x_block_stride, stream);
    case 6:
      return Launch<6>(type, w, x, y, m, k, x_block_stride, stream);
    case 7:
      return Launch<7>(type, w, x, y, m, k, x_block_stride, stream);
    case 8:
      return Launch<8>(type, w, x, y, m, k, x_block_stride, stream);
    default:
      return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next::dense_dot4
