#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_REFUSAL_MATH_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_REFUSAL_MATH_HPP_

// The same lane dot and subtract compile for the device and the CPU arithmetic
// test. No HIP context is needed for the host test. Products are materialized
// separately so production fast-math cannot contract away the F32 boundaries.
#ifdef __HIPCC__
#define QFN_REFUSAL_HD __host__ __device__
#else
#define QFN_REFUSAL_HD
#endif
namespace gufo::models::qwen38_flash_next {
QFN_REFUSAL_HD inline float RefusalLaneDot(const float* y, const float* r,
                                           unsigned dim, unsigned lane,
                                           unsigned threads) {
  float sum = 0;
  for (unsigned i = lane; i < dim; i += threads) {
#ifdef __HIP_DEVICE_COMPILE__
    sum = __fadd_rn(sum, __fmul_rn(y[i], r[i]));
#else
    volatile float product = y[i] * r[i];
    sum = sum + product;
#endif
  }
  return sum;
}
QFN_REFUSAL_HD inline float RefusalSubtract(float y, float r, float dot,
                                            float strength = 1.0F) {
#ifdef __HIP_DEVICE_COMPILE__
  return __fsub_rn(y, __fmul_rn(strength, __fmul_rn(r, dot)));
#else
  volatile float rd = r * dot;
  volatile float product = strength * rd;
  return y - product;
#endif
}
}  // namespace gufo::models::qwen38_flash_next
#undef QFN_REFUSAL_HD
#endif
