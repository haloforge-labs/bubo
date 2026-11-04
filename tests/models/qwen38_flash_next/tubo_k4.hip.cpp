// SPDX-License-Identifier: MIT
// Bench-only entry points of the tubo control kernels (new code, not tubo's).
// The copied kernels are in tubo_k4.hpp (MIT, BonkusCheemus/tubo); each width
// is instantiated in its own translation unit (tubo_k4_w*.hip.cpp). Declared
// for the bench in ablate.hpp. Nothing here is reachable from the engine.
#include "tubo_k4.hpp"

#include <atomic>

extern "C" {
int bench_tubo_run_w1(int, const void*, const void*, const void*, const void*,
                      float*, int, int);
int bench_tubo_run_w2(int, const void*, const void*, const void*, const void*,
                      float*, int, int);
int bench_tubo_run_w4(int, const void*, const void*, const void*, const void*,
                      float*, int, int);
int bench_tubo_run_w8(int, const void*, const void*, const void*, const void*,
                      float*, int, int);
void bench_tubo_pack_w1(int, const void*, void*, int);
void bench_tubo_pack_w2(int, const void*, void*, int);
void bench_tubo_pack_w4(int, const void*, void*, int);
void bench_tubo_pack_w8(int, const void*, void*, int);
}

namespace {

struct TuboInfo {
  const char* name;
  int type;     // ggml type id
  bool rows32;  // needs tubo's 32-row interleave (a re-layout of the file)
  int act;      // 0: q8_K activations (one scale per 256)
};
// The former dot4lds_* configs are the production dense variant `dot4` now
// (dense_dot4.hpp, Q8_1 activations); run them with --kernel dot4.
constexpr TuboInfo kTubo[] = {
    {"q4_native_k4", 12, false, 0},    {"q4_native_k4h", 12, false, 0},
    {"q4_rows32_k4h", 12, true, 0},    {"q4_rows32_r2k8hp", 12, true, 0},
    {"q5_native_k4", 13, false, 0},    {"q5_native_k4h", 13, false, 0},
    {"q5_rows32_k4h", 13, true, 0},    {"q5_rows32_r2k8hp", 13, true, 0},
    {"q6_rows32_r2k8hp", 14, true, 0}, {"iq4_rows32_r2k4hp", 23, true, 0}};
constexpr int kTuboConfigs = sizeof(kTubo) / sizeof(kTubo[0]);
std::atomic<unsigned long long> g_tubo_launches[kTuboConfigs][8]{};

bool WidthCompiled(int n) {
  return n == 1 || n == 2 || n == 4 || n == 8;
}

}  // namespace

extern "C" int bench_tubo_config_count() {
  return kTuboConfigs;
}

extern "C" int bench_tubo_config(int cfg, const char** name, int* type,
                                 int* rows32, int* act) {
  if (cfg < 0 || cfg >= kTuboConfigs)
    return -1;
  *name = kTubo[cfg].name;
  *type = kTubo[cfg].type;
  *rows32 = kTubo[cfg].rows32 ? 1 : 0;
  *act = kTubo[cfg].act;
  return 0;
}

extern "C" unsigned long long bench_tubo_launch_count(int cfg, int width) {
  if (cfg < 0 || cfg >= kTuboConfigs || width < 1 || width > 8)
    return 0;
  return g_tubo_launches[cfg][width - 1].load(std::memory_order_relaxed);
}

// Host-only: tubo's 32-row interleave of one tensor (rows % 32 == 0,
// k % 256 == 0). src and dst are distinct host buffers of the same size.
extern "C" int bench_tubo_repack(const void* src, void* dst, int rows, int k,
                                 int block_bytes) {
  try {
    gufo::models::InterleaveKQuantRows32(
        src, dst, std::size_t(rows), std::size_t(k), std::size_t(block_bytes));
  } catch (const std::exception&) {
    return -1;
  }
  return 0;
}

// Quantizes n activation rows (device float [n][k]) for config `cfg` and, for
// the packed-sum configs, packs the min-term sums. Not timed.
// xq: n*k bytes, xds: n*k/16 float2, xsum: k/256*9*n int32.
extern "C" int bench_tubo_prepare(int cfg, const float* x, void* xq, void* xds,
                                  void* xsum, int n, int k) {
  if (cfg < 0 || cfg >= kTuboConfigs || !WidthCompiled(n) || k <= 0 || k % 256)
    return -1;
  const std::uint32_t subs = std::uint32_t(n) * std::uint32_t(k / 16);
  gufo::hip::QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256>>>(
      x, static_cast<std::int8_t*>(xq), static_cast<float2*>(xds), subs);
  const int q6 = kTubo[cfg].type == 14 ? 1 : 0;
  switch (n) {
    case 1:
      bench_tubo_pack_w1(q6, xds, xsum, k);
      break;
    case 2:
      bench_tubo_pack_w2(q6, xds, xsum, k);
      break;
    case 4:
      bench_tubo_pack_w4(q6, xds, xsum, k);
      break;
    default:
      bench_tubo_pack_w8(q6, xds, xsum, k);
      break;
  }
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

// Returns the config id, or -1 unsupported (width not compiled, m % 32 for an
// interleaved config, k % 256), -3 launch error. xq/xds/xsum come from
// bench_tubo_prepare for this config; w is the layout the config names.
extern "C" int bench_tubo_launch(int cfg, const void* w, const void* xq,
                                 const void* xds, const void* xsum, float* y,
                                 int n, int m, int k) {
  if (cfg < 0 || cfg >= kTuboConfigs || !w || !xq || !xds || !y || m <= 0 ||
      k <= 0 || k % 256 || !WidthCompiled(n) || (kTubo[cfg].rows32 && m % 32))
    return -1;
  int result = -1;
  switch (n) {
    case 1:
      result = bench_tubo_run_w1(cfg, w, xq, xds, xsum, y, m, k);
      break;
    case 2:
      result = bench_tubo_run_w2(cfg, w, xq, xds, xsum, y, m, k);
      break;
    case 4:
      result = bench_tubo_run_w4(cfg, w, xq, xds, xsum, y, m, k);
      break;
    default:
      result = bench_tubo_run_w8(cfg, w, xq, xds, xsum, y, m, k);
      break;
  }
  if (result != 0)
    return -1;
  if (hipGetLastError() != hipSuccess)
    return -3;
  g_tubo_launches[cfg][n - 1].fetch_add(1, std::memory_order_relaxed);
  return cfg;
}
