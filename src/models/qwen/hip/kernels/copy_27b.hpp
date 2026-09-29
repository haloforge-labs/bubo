#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_COPY_27B_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_COPY_27B_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::hip {

/// Device-to-device copies as kernels for the Qwen3.8-27B decode cycle.
/// On Windows a hipMemcpyAsync between device buffers is handed to the copy
/// engine, which costs a queue switch per call (~150 us) and runs well below
/// the shader bandwidth for large state copies. Bytes are copied unchanged.
void LaunchDeviceCopy27(void* dst, const void* src, std::size_t bytes,
                        hipStream_t stream);

/// Row-strided copy: `rows` rows of `row_bytes`, like hipMemcpy2DAsync.
void LaunchDeviceCopy2D27(void* dst, std::size_t dst_pitch, const void* src,
                          std::size_t src_pitch, std::size_t row_bytes,
                          std::size_t rows, hipStream_t stream);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_COPY_27B_HPP_
