



























#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::size_t kExperts = 512, kUsed = 10, kHidden = 2560, kFF = 640;
// The layer mix of the hybrid model (one definition for every figure below).
constexpr double kLayers = 48, kLayersIq2S = 20, kLayersIq3Xxs = 17,
                 kLayersIq3S = 10 + 1 /* IQ4_XS mapped to IQ3_S */,
                 kLayersDownIq4Nl = 39 + 9 /* Q2_0 mapped to IQ4_NL */;

constexpr double kAbbaGapR2p = 0.504, kAbbaGapR2p0 = 0.326;

int g_failed = 0;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(hipMemcpy(host.data(), device, count * sizeof(T),
                     hipMemcpyDeviceToHost),
           "download");
  return host;
}

struct Format {
  const char* name;
  q::WeightType type;
  std::size_t block_values;
  std::size_t block_bytes;
  int scale_halves;  // fp16 scales at the block start (1: d, 2: d and dmin / m)
};

constexpr Format kIQ2_S{"IQ2_S", q::WeightType::kIQ2_S, 256, 82, 1};
constexpr Format kIQ3_XXS{"IQ3_XXS", q::WeightType::kIQ3_XXS, 256, 98, 1};
constexpr Format kIQ3_S{"IQ3_S", q::WeightType::kIQ3_S, 256, 110, 1};
constexpr Format kIQ4_NL{"IQ4_NL", q::WeightType::kIQ4_NL, 32, 18, 1};
constexpr Format kQ4_K{"Q4_K", q::WeightType::kQ4_K, 256, 144, 2};
constexpr Format kQ5_1{"Q5_1", q::WeightType::kQ5_1, 32, 24, 2};

/// Random blocks of [experts * rows] rows of `cols` values; every byte pattern
/// of the i-quants decodes, only the fp16 scale(s) at the block start are set.
std::vector<std::uint8_t> MakeWeights(const Format& f, std::size_t rows,
                                      std::size_t cols, std::uint32_t seed) {
  const std::size_t row_bytes = cols / f.block_values * f.block_bytes;
  std::vector<std::uint8_t> bytes(kExperts * rows * row_bytes);
  for (auto& b : bytes)
    b = static_cast<std::uint8_t>(NextRandom(&seed) >> 7);
  const std::size_t blocks = bytes.size() / f.block_bytes;
  for (std::size_t i = 0; i < blocks; ++i) {
    const float scale =
        0.002F + 0.004F * static_cast<float>(NextRandom(&seed) % 1000) / 1000.F;
    const __half d = __float2half(scale);
    std::memcpy(bytes.data() + i * f.block_bytes, &d, sizeof(d));
    if (f.scale_halves == 2) {
      const __half m = __float2half(f.type == q::WeightType::kQ4_K
                                        ? scale * 0.5F
                                        : 0.F);
      std::memcpy(bytes.data() + i * f.block_bytes + sizeof(d), &m, sizeof(m));
    }
  }
  return bytes;
}

struct Probe {
  int variant, ablate;
  const char* name;
};
// The (variant, ablate) pairs that have instances in LaunchRoutedIq.
// variant bits: 1 = 16-byte activation staging, 2 = register prefetch,
// 4 = LDS codebooks, 8 = F16 sign XOR, 16 = full output-row tiles,
// 32 = prefetch the grid/sign gather.
// ablate bits: 1 = decode skipped (and with it the raw weight fetch of variant 0),
// 2 = activation staging skipped, 4 = matrix math skipped, 8 = raw weight loads skipped.
const Probe kProbes[] = {
    {0, 0, "reviewed kernel"},
    {-1, -1, "DEFAULT dispatch"},
    {1, 0, "16-byte staging"},
    {2, 0, "register prefetch"},
    {3, 0, "staging + prefetch"},
    {4, 0, "LDS codebooks"},
    {7, 0, "staging + prefetch + LDS codebooks"},
    {11, 0, "staging + prefetch + sign bits"},
    {19, 0, "staging + prefetch + full rows"},
    {27, 0, "staging + prefetch + sign bits + full rows"},
    {35, 0, "staging + prefetch + grid gather"},
    {43, 0, "staging + prefetch + grid gather + sign bits"},
    {59, 0, "staging + prefetch + grid gather + full rows + sign bits"},
    {0, 1, "reviewed, decode skipped"},
    {0, 2, "reviewed, staging skipped"},
    {0, 4, "reviewed, matrix math skipped"},
    {7, 1, "all three, decode skipped"},
    {7, 2, "all three, staging skipped"},
    {7, 4, "all three, matrix math skipped"},
    {7, 9, "all three, decode + raw loads skipped"}};

std::vector<Probe> ActiveProbes(bool wide, bool pair) {
  std::vector<Probe> out;
  for (const Probe& p : kProbes) {
    if (p.variant >= 0 && (p.variant & 32) && (!wide || !pair))
      continue;
    if (wide || p.variant == -1 ||
        (p.ablate == 0 &&
         (p.variant == 0 || p.variant == 3 || p.variant == 7 ||
          p.variant == 11 || p.variant == 19 || p.variant == 27)) ||
        (p.ablate != 0 && (p.variant == 0 || p.variant == 7)))
      out.push_back(p);
  }
  return out;
}

struct Routing {
  std::size_t tokens, slots;
  std::int32_t *d_bounds, *d_rows_token, *d_rows_slot;
  std::uint32_t* d_counts;
  // IQ route tiles (the executor's choice, see the header).
  bool wide;
  std::uint32_t pair_rows, down_rows, rt_rows;
  std::int32_t *d_tiles_pair, *d_tiles_down;
  std::uint32_t n_pair, n_down;
  // P (lane-row) tiles.
  std::uint32_t p_pair_rows;
  std::int32_t *d_p_pair, *d_p_down;
  std::uint32_t n_p_pair, n_p_down;
  std::vector<std::uint32_t> counts;
};

Routing MakeRouting(std::size_t tokens, double skew) {
  Routing r{};
  r.tokens = tokens;
  r.slots = tokens * kUsed;
  std::uint32_t state = 0x9E3779B9U;
  std::vector<std::int32_t> ids(r.slots);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t s = 0; s < kUsed; ++s) {
      for (;;) {
        // skew > 0 favours low expert ids (real routing is skewed): e = E * u^(1 + skew)
        const double u = static_cast<double>(NextRandom(&state) >> 8) / 16777216.0;
        const auto e = static_cast<std::int32_t>(
            std::min<double>(kExperts - 1, kExperts * std::pow(u, 1.0 + skew)));
        bool dup = false;
        for (std::size_t p = 0; p < s; ++p)
          dup = dup || ids[t * kUsed + p] == e;
        if (!dup) {
          ids[t * kUsed + s] = e;
          break;
        }
      }
    }
  }
  r.counts.assign(kExperts, 0);
  for (auto e : ids)
    ++r.counts[e];
  auto* d_ids = Upload(ids);
  r.d_counts = Upload(r.counts);
  const std::size_t compact = q::RoutedCompactRows(r.slots, kExperts);
  const std::vector<std::int32_t> zeros(compact + kExperts + 1, 0);
  r.d_bounds = Upload(std::vector<std::int32_t>(kExperts + 1, 0));
  auto* d_cursors = Upload(std::vector<std::int32_t>(kExperts, 0));
  r.d_rows_token = Upload(zeros);
  r.d_rows_slot = Upload(zeros);
  q::RoutedCompact(d_ids, r.d_counts, r.d_bounds, d_cursors, r.d_rows_token,
                   r.d_rows_slot, static_cast<std::uint32_t>(tokens),
                   static_cast<std::uint32_t>(kUsed),
                   static_cast<std::uint32_t>(kExperts), nullptr);
  CheckHip(hipDeviceSynchronize(), "compact");
  const auto tile_map = [&](std::uint32_t rows) {
    std::vector<std::int32_t> tiles;
    for (std::size_t e = 0; e < kExperts; ++e)
      for (std::uint32_t j = 0; j < (r.counts[e] + rows - 1) / rows; ++j)
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    return tiles;
  };
  // IQ route
  r.rt_rows = r.slots >= 16 * kExperts ? 48 : 16;
  r.wide = tokens >= 1024 && r.rt_rows == 48;
  r.pair_rows = r.wide ? 128 : 64;
  r.down_rows = r.wide ? 96 : r.rt_rows;
  const auto pair = tile_map(r.pair_rows), down = tile_map(r.down_rows);
  r.n_pair = static_cast<std::uint32_t>(pair.size());
  r.n_down = static_cast<std::uint32_t>(down.size());
  r.d_tiles_pair = Upload(pair);
  r.d_tiles_down = Upload(down);
  // P lane-row route: executor.cpp RouteHints, lanerow branch.
  const auto t64 = tile_map(64), t128 = tile_map(128), t96 = tile_map(96);
  const bool pair128 = t128.size() * 4 <= t64.size() * 3;
  r.p_pair_rows = pair128 ? 128 : 64;
  r.n_p_pair = static_cast<std::uint32_t>(pair128 ? t128.size() : t64.size());
  r.d_p_pair = Upload(pair128 ? t128 : t64);
  r.n_p_down = static_cast<std::uint32_t>(t96.size());
  r.d_p_down = Upload(t96);
  std::uint32_t max_rows = 0;
  std::uint64_t hash = 1469598103934665603ULL;
  for (auto c : r.counts) {
    max_rows = std::max(max_rows, c);
    hash = (hash ^ c) * 1099511628211ULL;
  }
  std::printf("routing: %zu tokens x %zu used = %zu slots over %zu experts; rows per expert mean %.1f max %u; skew %.2f; counts hash %016llx\n"
              "  %s batch. IQ route: pair tiles(%u rows) %u, down tiles(%u rows) %u.  P lane-row: pair tiles(%u rows) %u, down tiles(96 rows) %u\n",
              tokens, kUsed, r.slots, kExperts,
              static_cast<double>(r.slots) / kExperts, max_rows, skew,
              static_cast<unsigned long long>(hash), r.wide ? "WIDE" : "NARROW",
              r.pair_rows, r.n_pair, r.down_rows, r.n_down, r.p_pair_rows,
              r.n_p_pair, r.n_p_down);
  return r;
}

struct Timing {
  double min = 0, med = 0, max = 0;
};

Timing Summary(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return {samples.front(), samples[samples.size() / 2], samples.back()};
}

/// Warm the GPU clocks for ~300 ms, then time `reps` calls. `call` must launch and
/// return; a launch error aborts the measurement.
template<typename Call>
Timing TimeIt(hipEvent_t a, hipEvent_t b, int reps, const Call& call) {
  const auto start = std::chrono::steady_clock::now();
  do {
    call();
    CheckHip(hipGetLastError(), "launch (warm-up)");
    CheckHip(hipDeviceSynchronize(), "warm-up");
  } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(300));
  std::vector<float> times;
  for (int i = 0; i < reps; ++i) {
    CheckHip(hipEventRecord(a, nullptr), "record");
    call();
    CheckHip(hipEventRecord(b, nullptr), "record");
    CheckHip(hipEventSynchronize(b), "sync");
    CheckHip(hipGetLastError(), "launch");
    float ms = 0;
    CheckHip(hipEventElapsedTime(&ms, a, b), "elapsed");
    times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  return {times.front(), times[times.size() / 2], times.back()};
}

template<typename Call>
double TimeOnce(hipEvent_t a, hipEvent_t b, const Call& call) {
  CheckHip(hipEventRecord(a, nullptr), "record");
  call();
  CheckHip(hipEventRecord(b, nullptr), "record");
  CheckHip(hipEventSynchronize(b), "sync");
  CheckHip(hipGetLastError(), "launch");
  float ms = 0;
  CheckHip(hipEventElapsedTime(&ms, a, b), "elapsed");
  return ms;
}

struct RoundResults {
  Timing p;
  std::vector<Timing> probe;
  std::vector<double> ratio_to_p;
};




template<typename PCall, typename IQCall>
RoundResults TimeRoundRobin(hipEvent_t a, hipEvent_t b, int reps,
                            const std::vector<Probe>& probes,
                            const PCall& p_call, const IQCall& iq_call) {
  std::vector<double> p_samples;
  std::vector<std::vector<double>> times(probes.size()), ratios(probes.size());
  for (const Probe& p : probes) {
    q::RoutedIqSetProbe(p.variant, p.ablate);
    iq_call();
    CheckHip(hipDeviceSynchronize(), "probe warm-up");
  }
  for (int round = 0; round < reps; ++round) {
    const double p_before = TimeOnce(a, b, p_call);
    std::vector<double> round_ms(probes.size());
    for (std::size_t i = 0; i < probes.size(); ++i) {
      const std::size_t index = (i + static_cast<std::size_t>(round)) % probes.size();
      const Probe& p = probes[index];
      q::RoutedIqSetProbe(p.variant, p.ablate);
      round_ms[index] = TimeOnce(a, b, iq_call);
    }
    const double p_same = (p_before + TimeOnce(a, b, p_call)) * 0.5;
    p_samples.push_back(p_same);
    for (std::size_t i = 0; i < probes.size(); ++i) {
      times[i].push_back(round_ms[i]);
      ratios[i].push_back(round_ms[i] / p_same);
    }
  }
  q::RoutedIqSetProbe(-1, -1);
  RoundResults out;
  out.p = Summary(p_samples);
  for (std::size_t i = 0; i < probes.size(); ++i) {
    out.probe.push_back(Summary(times[i]));
    out.ratio_to_p.push_back(Summary(ratios[i]).med);
  }
  return out;
}

/// Number of elements whose raw half bits differ (NaN-safe), and whether the
/// output looks like a real result (>= 99 % finite and nonzero).
std::size_t BitDiffs(const std::vector<__half>& a, const std::vector<__half>& b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::uint16_t x, y;
    std::memcpy(&x, &a[i], 2);
    std::memcpy(&y, &b[i], 2);
    n += x != y;
  }
  return n;
}

bool LooksReal(const std::vector<__half>& a) {
  std::size_t good = 0;
  for (const auto& h : a) {
    const float f = __half2float(h);
    good += std::isfinite(f) && f != 0.F;
  }
  return good * 100 >= a.size() * 99;
}

void PrintEnvironment() {
  int rt = 0, drv = 0;
  hipRuntimeGetVersion(&rt);
  hipDriverGetVersion(&drv);
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, 0);
  std::printf("environment: device '%s' arch %s; hip runtime %d driver %d\n",
              prop.name, prop.gcnArchName, rt, drv);
#ifdef _WIN32
  char path[1024] = "(not loaded)";
  if (HMODULE m = GetModuleHandleA("amdhip64_7.dll"))
    GetModuleFileNameA(m, path, sizeof(path));
  std::printf("environment: amdhip64_7.dll = %s\n", path);
#endif
}

void Fail(const char* what) {
  ++g_failed;
  std::printf("FAIL: %s\n", what);
}

struct SizeResult {
  double p_gu = 0, p_dn = 0, p_gu_round[3] = {0, 0, 0},
         p_dn_round = 0, iq_gu[3] = {0, 0, 0}, iq_dn = 0,
         default_gu[3] = {0, 0, 0}, default_dn = 0;
};

SizeResult RunOne(std::size_t tokens, int reps, double skew) {
  std::printf("\n==== %zu tokens, skew %.2f ====\n", tokens, skew);
  const Routing r = MakeRouting(tokens, skew);
  std::vector<__half> x(tokens * kHidden);
  std::uint32_t seed = 0x1234567U;
  for (auto& v : x)
    v = __float2half(static_cast<float>(static_cast<int>(NextRandom(&seed) % 2001) - 1000) / 2500.F);
  auto* d_x = Upload(x);
  __half *d_up = nullptr, *d_down = nullptr;
  CheckHip(hipMalloc(&d_up, r.slots * kFF * sizeof(__half) + 4096), "up");
  CheckHip(hipMalloc(&d_down, r.slots * kHidden * sizeof(__half) + 4096), "down");
  hipEvent_t ea, eb;
  CheckHip(hipEventCreate(&ea), "event");
  CheckHip(hipEventCreate(&eb), "event");

  const double gu_flops = 2.0 * 2.0 * kFF * kHidden * static_cast<double>(r.slots);
  const double dn_flops = 2.0 * kHidden * kFF * static_cast<double>(r.slots);
  SizeResult res;

  // ---- the lane-row Q4_K / Q5_1 kernels P runs (the speed to reach)
  Timing p_gu, p_dn;
  const auto p_gu_call = [&](const void* g, const void* u) {
    if (!q::RoutedLanerowGatedF16Gemm(g, u, d_x, r.d_p_pair, r.n_p_pair, r.p_pair_rows, r.d_bounds, r.d_counts, r.d_rows_token, r.d_rows_slot, d_up, kFF, kHidden, nullptr))
      throw std::runtime_error("P lane-row gate/up rejected");
  };
  const auto p_dn_call = [&](const void* w) {
    if (!q::RoutedLanerowF16Gemm(w, q::WeightType::kQ5_1, d_up, r.d_p_down, r.n_p_down, 96, r.d_bounds, r.d_counts, r.d_rows_slot, r.d_rows_slot, d_down, kHidden, kFF, nullptr))
      throw std::runtime_error("P lane-row down rejected");
  };
  {
    const auto wg = MakeWeights(kQ4_K, kFF, kHidden, 11U);
    const auto wu = MakeWeights(kQ4_K, kFF, kHidden, 13U);
    const auto dn = MakeWeights(kQ5_1, kHidden, kFF, 12U);
    auto *d_g = Upload(wg), *d_u = Upload(wu), *d_dn = Upload(dn);
    p_gu = TimeIt(ea, eb, reps, [&] { p_gu_call(d_g, d_u); });
    p_dn = TimeIt(ea, eb, reps, [&] { p_dn_call(d_dn); });
    res.p_gu = p_gu.med;
    res.p_dn = p_dn.med;
    std::printf("REF P lane-row  Q4_K gate/up %.3f ms (min %.3f max %.3f, %.1f TFLOP/s)  Q5_1 down %.3f ms (min %.3f max %.3f, %.1f TFLOP/s)  -> %.4f ms/token over %.0f layers\n",
                p_gu.med, p_gu.min, p_gu.max, gu_flops / p_gu.med / 1e9, p_dn.med, p_dn.min, p_dn.max, dn_flops / p_dn.med / 1e9,
                (p_gu.med + p_dn.med) * kLayers / static_cast<double>(tokens), kLayers);
    if (p_gu.max > 1.10 * p_gu.min || p_dn.max > 1.10 * p_dn.min)
      std::printf("WARNING: P timing spread above 10 %% (max/min): inspect same-round ratios\n");
    // ---- gate/up pairs, IQ types (the P buffers stay alive for the drift re-time at the end)
    const Format* const gu_formats[3] = {&kIQ2_S, &kIQ3_XXS, &kIQ3_S};
    for (int fi = 0; fi < 3; ++fi) {
      const Format* f = gu_formats[fi];
      const auto wgi = MakeWeights(*f, kFF, kHidden, 21U);
      const auto wui = MakeWeights(*f, kFF, kHidden, 23U);
      auto *d_wg = Upload(wgi), *d_wu = Upload(wui);
      const auto probes = ActiveProbes(r.wide, true);
      const auto iq_call = [&] {
        if (!q::RoutedGatedF16Gemm(d_wg, d_wu, f->type, d_x,
                                   r.d_tiles_pair, r.n_pair, r.pair_rows,
                                   r.d_bounds, r.d_rows_token, r.d_rows_slot,
                                   d_up, kFF, kHidden, nullptr))
          throw std::runtime_error("IQ gate/up launch refused (missing probe instance?)");
      };
      const auto rounds = TimeRoundRobin(ea, eb, reps, probes,
                                         [&] { p_gu_call(d_g, d_u); }, iq_call);
      res.p_gu_round[fi] = rounds.p.med;
      std::printf("ROUND REF %s gate/up P %.3f ms (min %.3f max %.3f); probe ratios use same-round P\n",
                  f->name, rounds.p.med, rounds.p.min, rounds.p.max);
      std::vector<__half> reference;
      double t00 = 0, t01 = 0, t02 = 0, t04 = 0, t70 = 0, t71 = 0, t79 = 0;
      std::size_t applied = 0, wanted = 0, identical_ok = 0, identical_want = 0;
      std::vector<std::pair<int, double>> ranking;
      for (std::size_t pi = 0; pi < probes.size(); ++pi) {
        const Probe& p = probes[pi];
        q::RoutedIqSetProbe(p.variant, p.ablate);
        try {
          CheckHip(hipMemset(d_up, 0, r.slots * kFF * sizeof(__half)), "clear");
          iq_call();
          CheckHip(hipGetLastError(), "IQ gate/up launch");
          const Timing t = rounds.probe[pi];
          const auto out = Download(d_up, r.slots * kFF);
          char check[96] = "";
          if (p.ablate <= 0) {
            if (p.variant == 0) {
              reference = out;
              if (!LooksReal(reference)) Fail("reference output is not a real result (zero/non-finite)");
              std::snprintf(check, sizeof(check), "reference");
            } else {
              const std::size_t d = BitDiffs(out, reference);
              ++identical_want;
              identical_ok += d == 0;
              std::snprintf(check, sizeof(check), d == 0 ? "bit-identical to reference" : "DIFFERS from reference in %zu of %zu values", d, out.size());
            }
          } else {
            const std::size_t d = BitDiffs(out, reference);
            ++wanted;
            applied += d != 0;
            std::snprintf(check, sizeof(check), d != 0 ? "ablated result differs (ablation ran)" : "ABLATION NOT APPLIED (output equals reference)");
          }
          const bool ablated = p.ablate > 0;
          std::printf("GATE/UP %-8s %-38s %8.3f ms (min %.3f max %.3f) %5.1f TFLOP/s ", f->name, p.name, t.med, t.min, t.max, gu_flops / t.med / 1e9);
          if (ablated)
            std::printf("(wrong result by design) ");
          else
            std::printf("x%.2f of same-round P ", rounds.ratio_to_p[pi]);
          std::printf("%s\n", check);
          if (t.max > 1.10 * t.min) std::printf("  WARNING: spread above 10 %% for this line\n");
          if (!ablated)
            ranking.emplace_back(static_cast<int>(pi), rounds.ratio_to_p[pi]);
          if (p.variant == 0 && p.ablate == 0) t00 = t.med;
          if (p.variant == 0 && p.ablate == 1) t01 = t.med;
          if (p.variant == 0 && p.ablate == 2) t02 = t.med;
          if (p.variant == 0 && p.ablate == 4) t04 = t.med;
          if (p.variant == 7 && p.ablate == 0) t70 = t.med;
          if (p.variant == 7 && p.ablate == 1) t71 = t.med;
          if (p.variant == 7 && p.ablate == 9) t79 = t.med;
          if (p.variant == 0 && p.ablate == 0) res.iq_gu[fi] = t.med;
          if (p.variant == -1) res.default_gu[fi] = t.med;
        } catch (const std::exception& e) {
          std::printf("GATE/UP %-8s %-38s FAILED: %s\n", f->name, p.name, e.what());
          Fail("a probe launch failed");
        }
      }
      q::RoutedIqSetProbe(-1, -1);
      if (t00 > 0) {
        std::printf("MARGINAL COSTS %s gate/up (stages overlap; they do not sum to 100 %%): reviewed kernel: decode+weight fetch %.3f ms (%.0f %%), activation staging %.3f ms (%.0f %%), matrix math %.3f ms (%.0f %%)\n",
                    f->name, t00 - t01, 100 * (t00 - t01) / t00, t00 - t02, 100 * (t00 - t02) / t00, t00 - t04, 100 * (t00 - t04) / t00);
        if (t70 > 0)
          std::printf("MARGINAL COSTS %s gate/up: all three: decode %.3f ms (%.0f %%), raw weight loads %.3f ms (%.0f %%)\n", f->name, t70 - t71, 100 * (t70 - t71) / t70, t71 - t79, 100 * (t71 - t79) / t70);
      }
      std::printf("PROBE CHECK %s gate/up: ablations applied %zu of %zu; non-ablated variants bit-identical %zu of %zu\n",
                  f->name, applied, wanted, identical_ok, identical_want);
      if (applied != wanted) Fail("an ablation was not applied");
      if (identical_ok != identical_want) Fail("a variant is not bit-identical");
      std::sort(ranking.begin(), ranking.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      std::printf("RANKING %s gate/up %zu tokens:", f->name, tokens);
      for (const auto& [index, ratio] : ranking) {
        if (probes[index].variant == -1)
          std::printf(" DEFAULT=%.3fxP", 1.0 / ratio);
        else
          std::printf(" v%d=%.3fxP", probes[index].variant, 1.0 / ratio);
      }
      std::printf("\n");
      if (!ranking.empty()) {
        const auto& best = ranking.front();
        if (probes[best.first].variant == -1)
          std::printf("BEST %s gate/up %zu tokens: DEFAULT speed_vs_P=%.3f\n",
                      f->name, tokens, 1.0 / best.second);
        else
          std::printf("BEST %s gate/up %zu tokens: v%d speed_vs_P=%.3f\n",
                      f->name, tokens, probes[best.first].variant,
                      1.0 / best.second);
      }
      CheckHip(hipFree(d_wg), "free");
      CheckHip(hipFree(d_wu), "free");
    }

    // ---- down, IQ4_NL
    {
      const auto dni = MakeWeights(kIQ4_NL, kHidden, kFF, 31U);
      auto* d_dni = Upload(dni);
      // the input rows: a fixed pseudo-random F16 plane
      std::vector<__half> h(r.slots * kFF);
      for (auto& v : h)
        v = __float2half(static_cast<float>(static_cast<int>(NextRandom(&seed) % 2001) - 1000) / 2500.F);
      CheckHip(hipMemcpy(d_up, h.data(), h.size() * sizeof(__half), hipMemcpyHostToDevice), "up fill");
      const auto probes = ActiveProbes(r.wide, false);
      const auto iq_call = [&] {
        if (!q::RoutedF16Gemm(d_dni, q::WeightType::kIQ4_NL, d_up,
                              r.d_tiles_down, r.n_down, r.down_rows, r.d_bounds,
                              r.d_rows_slot, r.d_rows_slot, nullptr, nullptr,
                              d_down, kHidden, kFF, nullptr))
          throw std::runtime_error("IQ4_NL down launch refused (missing probe instance?)");
      };
      const auto rounds = TimeRoundRobin(ea, eb, reps, probes,
                                         [&] { p_dn_call(d_dn); }, iq_call);
      res.p_dn_round = rounds.p.med;
      std::printf("ROUND REF IQ4_NL down P %.3f ms (min %.3f max %.3f); probe ratios use same-round P\n",
                  rounds.p.med, rounds.p.min, rounds.p.max);
      std::vector<__half> reference;
      double t00 = 0, t01 = 0, t02 = 0, t04 = 0;
      std::size_t applied = 0, wanted = 0, identical_ok = 0, identical_want = 0;
      std::vector<std::pair<int, double>> ranking;
      for (std::size_t pi = 0; pi < probes.size(); ++pi) {
        const Probe& p = probes[pi];
        q::RoutedIqSetProbe(p.variant, p.ablate);
        try {
          CheckHip(hipMemset(d_down, 0, r.slots * kHidden * sizeof(__half)), "clear");
          iq_call();
          CheckHip(hipGetLastError(), "IQ4_NL down launch");
          const Timing t = rounds.probe[pi];
          const auto out = Download(d_down, r.slots * kHidden);
          char check[96] = "";
          if (p.ablate <= 0) {
            if (p.variant == 0) {
              reference = out;
              if (!LooksReal(reference)) Fail("down reference output is not a real result (zero/non-finite)");
              std::snprintf(check, sizeof(check), "reference");
            } else {
              const std::size_t d = BitDiffs(out, reference);
              ++identical_want;
              identical_ok += d == 0;
              std::snprintf(check, sizeof(check), d == 0 ? "bit-identical to reference" : "DIFFERS from reference in %zu of %zu values", d, out.size());
            }
          } else {
            const std::size_t d = BitDiffs(out, reference);
            ++wanted;
            applied += d != 0;
            std::snprintf(check, sizeof(check), d != 0 ? "ablated result differs (ablation ran)" : "ABLATION NOT APPLIED (output equals reference)");
          }
          std::printf("DOWN    %-8s %-38s %8.3f ms (min %.3f max %.3f) %5.1f TFLOP/s ", "IQ4_NL", p.name, t.med, t.min, t.max, dn_flops / t.med / 1e9);
          if (p.ablate > 0)
            std::printf("(wrong result by design) ");
          else
            std::printf("x%.2f of same-round P ", rounds.ratio_to_p[pi]);
          std::printf("%s\n", check);
          if (t.max > 1.10 * t.min) std::printf("  WARNING: spread above 10 %% for this line\n");
          if (p.ablate <= 0)
            ranking.emplace_back(static_cast<int>(pi), rounds.ratio_to_p[pi]);
          if (p.variant == 0 && p.ablate == 0) { t00 = t.med; res.iq_dn = t.med; }
          if (p.variant == -1) res.default_dn = t.med;
          if (p.variant == 0 && p.ablate == 1) t01 = t.med;
          if (p.variant == 0 && p.ablate == 2) t02 = t.med;
          if (p.variant == 0 && p.ablate == 4) t04 = t.med;
        } catch (const std::exception& e) {
          std::printf("DOWN    IQ4_NL %-38s FAILED: %s\n", p.name, e.what());
          Fail("a probe launch failed");
        }
      }
      q::RoutedIqSetProbe(-1, -1);
      if (t00 > 0) {
        std::printf("MARGINAL COSTS IQ4_NL down (stages overlap): decode+weight fetch %.3f ms (%.0f %%), activation staging %.3f ms (%.0f %%), matrix math %.3f ms (%.0f %%)\n",
                    t00 - t01, 100 * (t00 - t01) / t00, t00 - t02, 100 * (t00 - t02) / t00, t00 - t04, 100 * (t00 - t04) / t00);
      }
      std::printf("PROBE CHECK IQ4_NL down: ablations applied %zu of %zu; non-ablated variants bit-identical %zu of %zu\n",
                  applied, wanted, identical_ok, identical_want);
      if (applied != wanted) Fail("an ablation was not applied");
      if (identical_ok != identical_want) Fail("a variant is not bit-identical");
      std::sort(ranking.begin(), ranking.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      std::printf("RANKING IQ4_NL down %zu tokens:", tokens);
      for (const auto& [index, ratio] : ranking) {
        if (probes[index].variant == -1)
          std::printf(" DEFAULT=%.3fxP", 1.0 / ratio);
        else
          std::printf(" v%d=%.3fxP", probes[index].variant, 1.0 / ratio);
      }
      std::printf("\n");
      if (!ranking.empty()) {
        const auto& best = ranking.front();
        if (probes[best.first].variant == -1)
          std::printf("BEST IQ4_NL down %zu tokens: DEFAULT speed_vs_P=%.3f\n",
                      tokens, 1.0 / best.second);
        else
          std::printf("BEST IQ4_NL down %zu tokens: v%d speed_vs_P=%.3f\n",
                      tokens, probes[best.first].variant,
                      1.0 / best.second);
      }
      CheckHip(hipFree(d_dni), "free");
    }

    // ---- drift check: P again after the IQ lines
    CheckHip(hipMemcpy(d_up, std::vector<__half>(r.slots * kFF, __float2half(0.01F)).data(), r.slots * kFF * sizeof(__half), hipMemcpyHostToDevice), "up fill");
    const Timing p_gu2 = TimeIt(ea, eb, reps, [&] { p_gu_call(d_g, d_u); });
    const Timing p_dn2 = TimeIt(ea, eb, reps, [&] { p_dn_call(d_dn); });
    const double drift_gu = 100 * (p_gu2.med / p_gu.med - 1), drift_dn = 100 * (p_dn2.med / p_dn.med - 1);
    std::printf("DRIFT P re-timed at the end: gate/up %.3f ms (%+.1f %%), down %.3f ms (%+.1f %%)%s\n", p_gu2.med, drift_gu, p_dn2.med, drift_dn,
                 (std::abs(drift_gu) > 3 || std::abs(drift_dn) > 3) ? "  WARNING: drift above 3 %, inspect same-round ratios" : "");
    CheckHip(hipFree(d_g), "free");
    CheckHip(hipFree(d_u), "free");
    CheckHip(hipFree(d_dn), "free");
  }
  // ---- does the bench reproduce the engine? Added cost per prompt token over P with the 48-layer mix.
  if (res.iq_gu[0] > 0 && res.iq_dn > 0) {
    const double iq_ms = kLayersIq2S * res.iq_gu[0] + kLayersIq3Xxs * res.iq_gu[1] + kLayersIq3S * res.iq_gu[2] + kLayersDownIq4Nl * res.iq_dn;
    const double p_ms = kLayersIq2S * res.p_gu_round[0] +
                        kLayersIq3Xxs * res.p_gu_round[1] +
                        kLayersIq3S * res.p_gu_round[2] +
                        kLayersDownIq4Nl * res.p_dn_round;
    std::printf("ENGINE GAP (reviewed kernel over P, %.0f layers, per prompt token): %.3f ms.  The 10-21 ABBA measured %.3f (R2p) and %.3f (R2p0); within +-20 %% of %.3f means the bench reproduces the engine.\n",
                kLayers, (iq_ms - p_ms) / static_cast<double>(tokens), kAbbaGapR2p, kAbbaGapR2p0, kAbbaGapR2p);
    const double p_tok = p_ms / static_cast<double>(tokens);
    std::printf("P itself (experts only): %.3f ms per prompt token over %.0f layers.  Gap allowed for 0.95x of P (same experts-only basis): %.3f ms; for 1.0x: 0.\n", p_tok, kLayers, p_tok * (1.0 / 0.95 - 1.0));
  }
  if (res.default_gu[0] > 0 && res.default_dn > 0) {
    const double default_ms = kLayersIq2S * res.default_gu[0] +
                              kLayersIq3Xxs * res.default_gu[1] +
                              kLayersIq3S * res.default_gu[2] +
                              kLayersDownIq4Nl * res.default_dn;
    const double p_ms = kLayersIq2S * res.p_gu_round[0] +
                        kLayersIq3Xxs * res.p_gu_round[1] +
                        kLayersIq3S * res.p_gu_round[2] +
                        kLayersDownIq4Nl * res.p_dn_round;
    std::printf("ENGINE GAP (DEFAULT dispatch over P, %.0f layers, per prompt token): %.3f ms.\n",
                kLayers, (default_ms - p_ms) / static_cast<double>(tokens));
  }
  CheckHip(hipFree(d_x), "free");
  CheckHip(hipFree(d_up), "free");
  CheckHip(hipFree(d_down), "free");
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    std::size_t tokens = 0;  // 0 = the sweep below
    int reps = 21;
    double skew = 0.0;
    for (int i = 1; i < argc; i += 2) {
      const std::string key = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
      if (key == "--tokens")
        tokens = std::strtoull(argv[i + 1], nullptr, 10);
      else if (key == "--reps")
        reps = std::atoi(argv[i + 1]);
      else if (key == "--skew")
        skew = std::atof(argv[i + 1]);
      else
        throw std::runtime_error("unknown flag " + key);
    }
    if (reps < 1)
      throw std::runtime_error("--reps must be positive");
    std::printf("routed_wmma_iq_bench v4; reps %d\n", reps);
    PrintEnvironment();
    if (tokens != 0) {
      RunOne(tokens, reps, skew);
    } else {
      RunOne(4096, reps, 0.0);
      RunOne(4096, reps, 1.0);
      for (const std::size_t t : {1024, 512, 128})
        RunOne(t, reps, 0.0);
    }
    std::printf("\nDONE; failures: %d\n", g_failed);
    std::printf(g_failed == 0 ? "BENCH VERDICT: OK (every probe ran, ablations applied, variants bit-identical)\n" : "BENCH VERDICT: FAILED (see FAIL lines)\n");
    return g_failed == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
