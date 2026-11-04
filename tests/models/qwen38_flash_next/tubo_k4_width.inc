// SPDX-License-Identifier: MIT
// Bench-only. One width of the tubo control kernels per translation unit, so
// the large unrolled kernels compile in parallel. The including stub defines
// TUBO_WIDTH (see tubo_k4_w*.hip.cpp). Kernels: tubo_k4.hpp (copied, MIT).
#include "tubo_k4.hpp"

#define TUBO_CAT2(a, b) a##b
#define TUBO_CAT(a, b) TUBO_CAT2(a, b)

namespace {

template<std::size_t B>
int TuboRun(int cfg, const void* w, std::int8_t* xq, float2* xds,
            std::int32_t* xsum, float* y, std::uint32_t m, std::uint32_t k) {
  using namespace gufo::hip;
#define TUBO_K4(...)                                                         \
  LaunchSnQ4KK4<B, __VA_ARGS__>(w, nullptr, y, xq, xds, m, k, nullptr, 1U,   \
                                nullptr, nullptr, false, xsum);              \
  return 0
  switch (cfg) {
    // RowWaves, KPhases, SplitK, Interleaved, Q5, Hoist, Packed, Probe,
    // Dynamic, Q6, IQ4
    case 0: TUBO_K4(4, 4, false, false, false, false, false);
    case 1: TUBO_K4(4, 4, false, false, false, true, false);
    case 2: TUBO_K4(4, 4, false, true, false, true, false);
    case 3: TUBO_K4(2, 8, false, true, false, true, true);
    case 4: TUBO_K4(4, 4, false, false, true, false, false);
    case 5: TUBO_K4(4, 4, false, false, true, true, false);
    case 6: TUBO_K4(4, 4, false, true, true, true, false);
    case 7: TUBO_K4(2, 8, false, true, true, true, true);
    case 8: TUBO_K4(2, 8, false, true, false, true, true, 0, false, true, false);
    case 9: TUBO_K4(2, 4, false, true, false, true, false, 0, false, false, true);
    default:
      return -1;
  }
#undef TUBO_K4
}

}  // namespace

extern "C" int TUBO_CAT(bench_tubo_run_w, TUBO_WIDTH)(
    int cfg, const void* w, const void* xq, const void* xds, const void* xsum,
    float* y, int m, int k) {
  return TuboRun<TUBO_WIDTH>(
      cfg, w, const_cast<std::int8_t*>(static_cast<const std::int8_t*>(xq)),
      const_cast<float2*>(static_cast<const float2*>(xds)),
      const_cast<std::int32_t*>(static_cast<const std::int32_t*>(xsum)), y,
      std::uint32_t(m), std::uint32_t(k));
}

extern "C" void TUBO_CAT(bench_tubo_pack_w, TUBO_WIDTH)(int q6, const void* xds,
                                                        void* xsum, int k) {
  if (q6)
    gufo::hip::LaunchSnQ4KPackSums<TUBO_WIDTH, true>(
        static_cast<const float2*>(xds), static_cast<std::int32_t*>(xsum),
        std::uint32_t(k), nullptr);
  else
    gufo::hip::LaunchSnQ4KPackSums<TUBO_WIDTH, false>(
        static_cast<const float2*>(xds), static_cast<std::int32_t*>(xsum),
        std::uint32_t(k), nullptr);
}
