// Decode-shape routed-expert kernel bench (multislot, 2026-10-20).
//
// Real modes survey native headers and replay captured routing. The default
// production route uses grouped gated vectors; V2K requires GUFO_EXPERT_V2K=1.
// Synthetic sweeps below also time V2K explicitly as a control. No synthetic
// model sum attributes the native file's whole-pass verify time.
//
// For every type this times the routes the executor really takes at decode
// (executor.cpp GatedExperts / Experts / the V2K branch of MoeExperts) at the
// production geometry (512 experts, top-10, hidden 2560, expert width 640)
// for 1..8 tokens, with 256 MiB of separate cache traffic before each timed
// call, and reports microseconds per call and effective GB/s = bytes of the
// DISTINCT experts read / time. Weights are random bytes with uniform IQ2_S
// grid indices; LDS versus global grid gathers may rank differently on real
// weights. Real-weight checks compare bytes only. It also prints a modelled
// expert time per verify round for P and for R2p (layer counts of the two
// files) so that the model can be set against the measured decode ratio (slot
// B: 42.4 vs 42.6 tok/s).
//
// usage: qwen38_flash_next_expert_decode_bench [--iters N] [--sets K]
//        [--survey first-shard] | [--bench-real first-shard]
// Nothing else may use the GPU during timed runs. Survey reads only headers.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "expert_bench_policy.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/iq_decode_bits.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

hipError_t ExpertBenchEvict(void* buffer, std::size_t bytes);

namespace {

using gufo::core::GgmlType;

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kHidden = 2560;
constexpr int kFF = 640;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct Fmt {
  const char* name;
  GgmlType type;
  std::size_t block;
  std::size_t bytes;
};
constexpr Fmt kQ4_K{"Q4_K", GgmlType::kQ4_K, 256, 144};
constexpr Fmt kQ5_K{"Q5_K", GgmlType::kQ5_K, 256, 176};
constexpr Fmt kQ5_1{"Q5_1", GgmlType::kQ5_1, 32, 24};
constexpr Fmt kQ8_0{"Q8_0", GgmlType::kQ8_0, 32, 34};
constexpr Fmt kIQ3_S{"IQ3_S", GgmlType::kIQ3_S, 256, 110};
constexpr Fmt kIQ4_XS{"IQ4_XS", GgmlType::kIQ4_XS, 256, 136};
constexpr Fmt kIQ4_NL{"IQ4_NL", GgmlType::kIQ4_NL, 32, 18};
constexpr Fmt kIQ3_XXS{"IQ3_XXS", GgmlType::kIQ3_XXS, 256, 98};
constexpr Fmt kIQ2_S{"IQ2_S", GgmlType::kIQ2_S, 256, 82};
constexpr Fmt kIQ2_XXS{"IQ2_XXS", GgmlType::kIQ2_XXS, 256, 66};
constexpr Fmt kIQ2_XS{"IQ2_XS", GgmlType::kIQ2_XS, 256, 74};
constexpr Fmt kIQ1_M{"IQ1_M", GgmlType::kIQ1_M, 256, 56};
constexpr Fmt kQ2_0{"Q2_0", GgmlType::kQ2_0, 64, 18};

std::size_t RowBytes(const Fmt& f, int cols) { return std::size_t(cols) / f.block * f.bytes; }

std::uint32_t NextRandom(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

/// Device buffer of random bytes, a 16 MiB chunk repeated; IQ2_S grid indices are uniform.
void* RandomDevice(std::size_t bytes, std::uint32_t seed) {
  static std::vector<std::uint8_t> chunk;
  if (chunk.empty()) {
    chunk.resize(16u << 20);
    std::uint32_t s = seed | 1U;
    for (auto& b : chunk) b = static_cast<std::uint8_t>(NextRandom(&s));
  }
  void* d = nullptr;
  Check(hipMalloc(&d, bytes + 4096), "hipMalloc weights");
  for (std::size_t off = 0; off < bytes; off += chunk.size()) {
    const std::size_t n = std::min(chunk.size(), bytes - off);
    Check(hipMemcpy(static_cast<char*>(d) + off, chunk.data(), n, hipMemcpyHostToDevice), "upload weights");
  }
  return d;
}

float* DeviceFloats(std::size_t n, float scale, std::uint32_t seed) {
  std::vector<float> h(n);
  std::uint32_t s = seed | 1U;
  for (auto& v : h) v = scale * (static_cast<float>(int(NextRandom(&s) % 2001) - 1000) / 1000.F);
  float* d = nullptr;
  Check(hipMalloc(&d, n * sizeof(float) + 4096), "hipMalloc floats");
  Check(hipMemcpy(d, h.data(), n * sizeof(float), hipMemcpyHostToDevice), "upload floats");
  return d;
}

/// K expert sets: per token ten DISTINCT experts of 512 (the router's top-10), different in every set.
struct IdSets {
  int tokens = 0, sets = 0;
  std::vector<std::vector<std::int32_t>> host;   // [set][token * kUsed + slot]
  std::vector<int> distinct;                     // distinct experts of the set (all tokens)
  std::int32_t* device = nullptr;                // [set][tokens * kUsed]
};

IdSets MakeIds(int tokens, int sets, std::uint32_t seed) {
  IdSets r;
  r.tokens = tokens;
  r.sets = sets;
  std::uint32_t s = seed | 1U;
  std::vector<std::int32_t> all;
  for (int k = 0; k < sets; ++k) {
    std::vector<std::int32_t> ids;
    std::set<int> uniq;
    for (int t = 0; t < tokens; ++t) {
      std::set<int> row;
      while (static_cast<int>(row.size()) < kUsed) row.insert(static_cast<int>(NextRandom(&s) % kExperts));
      for (int e : row) {
        ids.push_back(e);
        uniq.insert(e);
      }
    }
    r.distinct.push_back(static_cast<int>(uniq.size()));
    all.insert(all.end(), ids.begin(), ids.end());
    r.host.push_back(std::move(ids));
  }
  Check(hipMalloc(&r.device, all.size() * sizeof(std::int32_t) + 4096), "hipMalloc ids");
  Check(hipMemcpy(r.device, all.data(), all.size() * sizeof(std::int32_t), hipMemcpyHostToDevice), "upload ids");
  return r;
}

double MeanDistinct(const IdSets& s, int iters) {
  return expert_bench::MeanDistinct(s.distinct, iters);
}

/// Mean of short event intervals; the cache sweep is ordered before each start.
expert_bench::Timing TimeUs(const std::function<int(int)>& call, int iters,
                            const std::function<void()>& prepare = {}) {
  const auto checked = [](hipError_t error, const char* phase) {
    if (error != hipSuccess) throw expert_bench::Failure(phase, int(error));
  };
  const auto sync_checked = [](const char* phase, hipError_t launch = hipSuccess, int rc = 0) {
    const auto sync = hipStreamSynchronize(nullptr);
    const auto last = hipGetLastError();
    const auto error = launch != hipSuccess ? launch : sync != hipSuccess ? sync : last;
    if (error != hipSuccess || rc != 0)
      throw expert_bench::Failure(phase, int(error), rc, int(sync), int(last));
  };
  const auto checked_call = [&](int i, bool warmup) {
    const int rc = call(i);
    const auto launch = hipGetLastError();
    if (warmup || rc != 0 || launch != hipSuccess)
      sync_checked(warmup ? "warmup_launch" : "expert_launch", launch, rc);
    return 0;
  };
  struct CacheBuffer {
    void* data = nullptr;
    CacheBuffer() {
      Check(hipMalloc(&data, expert_bench::kEvictBytes), "eviction allocation");
      Check(hipMemset(data, 0, expert_bench::kEvictBytes), "eviction initialization");
    }
    ~CacheBuffer() { (void)hipFree(data); }
  };
  hipEvent_t a = nullptr, b = nullptr;
  expert_bench::Timing result{0};
  try {
    sync_checked("cell_preflight");
    static CacheBuffer cache;
    for (int i = 0; i < 6; ++i) {
      result.sample = -i - 2;
      if (prepare)
        prepare();
      checked_call(i, true);
    }
    result.sample = -1;
    checked(hipEventCreate(&a), "event_create_start");
    checked(hipEventCreate(&b), "event_create_stop");
    result = expert_bench::Measure(
        iters,
        [&] {
          if (prepare)
            prepare();
          sync_checked("eviction_launch",
                       ExpertBenchEvict(cache.data, expert_bench::kEvictBytes));
        },
        [&] { checked(hipEventRecord(a, nullptr), "event_record_start"); },
        [&](int i) { return checked_call(i, false); },
        [&] { checked(hipEventRecord(b, nullptr), "event_record_stop"); },
        [&] {
          sync_checked("expert_completion");
          checked(hipEventSynchronize(b), "event_sync_stop");
          float ms = std::numeric_limits<float>::quiet_NaN();
          checked(hipEventElapsedTime(&ms, a, b), "event_elapsed_api");
          return 1000.0 * ms;
        });
  } catch (const expert_bench::Failure& error) {
    expert_bench::SetFailure(result, error);
  } catch (const std::exception&) {
    result.reason = "cell_setup_exception";
  }
  // Drain the cell even when event recording or a launch failed. A poisoned
  // context can invalidate later cells, but every cell still receives a row.
  const auto sync = hipStreamSynchronize(nullptr);
  const auto last = hipGetLastError();
  if (result.reason == "ok" && (sync != hipSuccess || last != hipSuccess))
    expert_bench::SetFailure(result, expert_bench::Failure("cell_completion", int(sync != hipSuccess ? sync : last), 0, int(sync), int(last)));
  if (a) (void)hipEventDestroy(a);
  if (b) (void)hipEventDestroy(b);
  return result;
}

struct Result {
  double us = -1, gbps = 0;
};

std::map<std::string, Result> g_results;   // key "gu|<fmt>|<route>|T" and "down|<fmt>|vec|T" and "v2k|<gu>+<down>|T"
std::vector<std::string> g_invalid_cells;
std::ofstream g_row_csv;

void OpenTimingCsv(const std::string& path) {
  g_row_csv.open(path, std::ios::out | std::ios::trunc);
  if (!g_row_csv) throw std::runtime_error("Cannot open timing CSV");
  g_row_csv << "kind,fmt,route,variant,T,distinct,us,GBps,valid,protocol,evict_bytes,reason,raw_us,hip_error,call_rc,sample,completed,sync_error,last_error,start_recorded,stop_recorded\n";
  g_row_csv.flush();
}

void Report(const std::string& key, const char* kind, const char* fmt, const char* route, int tokens, double distinct, const expert_bench::Timing& timing, double bytes, const char* variant = "default") {
  expert_bench::RequireComparable(timing, {1.0});
  const double us = timing.us;
  Result r;
  r.us = us;
  r.gbps = us > 0 ? expert_bench::Gbps(bytes, us) : 0;
  const bool valid = expert_bench::Valid(bytes, timing);
  const std::string reason = valid ? "ok" : timing.reason == "ok" ? "bandwidth_limit" : timing.reason;
  if (!valid) g_invalid_cells.push_back(key + " reason=" + reason + " sample=" + std::to_string(timing.sample) + " hip_error=" + std::to_string(timing.hip_error));
  g_results[key] = r;
  std::printf("BENCH kind=%s fmt=%s route=%s variant=%s T=%d distinct=%.1f us=%.3f GBps=%.3f valid=%d protocol=%.*s evict_bytes=%zu reason=%s raw_us=%.9g hip_error=%d call_rc=%d sample=%d completed=%d sync_error=%d last_error=%d start_recorded=%d stop_recorded=%d%s\n", kind, fmt, route, variant, tokens, distinct, us, r.gbps, int(valid), int(timing.protocol.size()), timing.protocol.data(), timing.evict_bytes, reason.c_str(), timing.raw_us, timing.hip_error, timing.call_rc, timing.sample, timing.completed, timing.sync_error, timing.last_error, int(timing.start_recorded), int(timing.stop_recorded), valid ? "" : " INVALID");
  std::fflush(stdout);
  if (g_row_csv.is_open()) {
    g_row_csv << std::setprecision(17) << kind << ',' << fmt << ',' << route << ',' << variant << ',' << tokens << ',' << distinct << ',' << us << ',' << r.gbps << ',' << int(valid) << ',' << timing.protocol << ',' << timing.evict_bytes << ',' << reason << ',' << timing.raw_us << ',' << timing.hip_error << ',' << timing.call_rc << ',' << timing.sample << ',' << timing.completed << ',' << timing.sync_error << ',' << timing.last_error << ',' << int(timing.start_recorded) << ',' << int(timing.stop_recorded) << '\n';
    g_row_csv.flush();
    if (!g_row_csv) throw std::runtime_error("Cannot flush timing CSV");
  }
}

int TimingStatus(bool strict) {
  std::printf("TIMING_SUMMARY invalid=%zu protocol=%.*s evict_bytes=%zu\n", g_invalid_cells.size(), int(expert_bench::kProtocol.size()), expert_bench::kProtocol.data(), expert_bench::kEvictBytes);
  for (const auto& key : g_invalid_cells)
    std::fprintf(stderr, "INVALID_CELL %s: timing invalid\n", key.c_str());
  return expert_bench::ExitCode(strict, g_invalid_cells.size());
}

const int kTokens[] = {1, 2, 3, 4, 5, 6, 7, 8};

/// Gate/up of one format: paired generic vector kernel (what the executor calls for every type but Q4_K/Q5_K/Q8_0 at decode) and, for Q4_K/Q5_K, the fused gated kernel.
void BenchGateUp(const Fmt& f, int iters, int nsets, const bool gated) {
  const std::size_t bytes = std::size_t(kExperts) * kFF * RowBytes(f, kHidden);
  void* wa = RandomDevice(bytes, 0x1234U);
  void* wb = RandomDevice(bytes, 0x4321U);
  for (int T : kTokens) {
    IdSets ids = MakeIds(T, nsets, 0x9E3779B9U + T);
    float* x = DeviceFloats(std::size_t(T) * kHidden, 1.F, 77U);
    float* oa = DeviceFloats(std::size_t(T) * kUsed * kFF, 0.F, 1U);
    float* ob = DeviceFloats(std::size_t(T) * kUsed * kFF, 0.F, 2U);
    const double moved = MeanDistinct(ids, iters) * 2.0 * kFF * RowBytes(f, kHidden);
    auto vec = [&](int i) {
      const std::int32_t* id = ids.device + std::size_t(i % nsets) * T * kUsed;
      return qfn_mmq_moe_vec(static_cast<int>(f.type), wa, x, id, oa, kFF, kHidden, T, kExperts, kUsed, nullptr, wb, ob);
    };
    Report(std::string("gu|") + f.name + "|vec-pair|" + std::to_string(T), "gu", f.name, "vec-pair", T, MeanDistinct(ids, iters), TimeUs(vec, iters), moved);
    if (gated) {
      auto fused = [&](int i) {
        const std::int32_t* id = ids.device + std::size_t(i % nsets) * T * kUsed;
        return qfn_mmq_moe_gated_vec(static_cast<int>(f.type), wa, wb, x, id, oa, kFF, kHidden, T, kExperts, kUsed, nullptr);
      };
      Report(std::string("gu|") + f.name + "|gated|" + std::to_string(T), "gu", f.name, "gated", T, MeanDistinct(ids, iters), TimeUs(fused, iters), moved);
    }
    (void)hipFree(x);
    (void)hipFree(oa);
    (void)hipFree(ob);
    (void)hipFree(ids.device);
  }
  (void)hipFree(wa);
  (void)hipFree(wb);
}

/// Down projection: one input row per (token, slot), one expert each (n_used == 1), as the executor runs it.
void BenchDown(const Fmt& f, int iters, int nsets) {
  const std::size_t bytes = std::size_t(kExperts) * kHidden * RowBytes(f, kFF);
  void* w = RandomDevice(bytes, 0x5555U);
  for (int T : kTokens) {
    IdSets ids = MakeIds(T, nsets, 0xC0FFEEU + T);
    const int slots = T * kUsed;
    float* xs = DeviceFloats(std::size_t(slots) * kFF, 1.F, 91U);
    float* out = DeviceFloats(std::size_t(slots) * kHidden, 0.F, 3U);
    const double moved = MeanDistinct(ids, iters) * kHidden * RowBytes(f, kFF);
    auto vec = [&](int i) {
      const std::int32_t* id = ids.device + std::size_t(i % nsets) * slots;
      return qfn_mmq_moe_vec(static_cast<int>(f.type), w, xs, id, out, kHidden, kFF, slots, kExperts, 1, nullptr);
    };
    Report(std::string("down|") + f.name + "|vec|" + std::to_string(T), "down", f.name, "vec", T, MeanDistinct(ids, iters), TimeUs(vec, iters), moved);
    (void)hipFree(xs);
    (void)hipFree(out);
    (void)hipFree(ids.device);
  }
  (void)hipFree(w);
}

/// Production's fused chain for N = 2..8: Q4_K gate/up with Q5_1 or Q8_0 down.
void BenchV2k(const Fmt& down, int iters, int nsets) {
  const std::size_t gu_bytes = std::size_t(kExperts) * kFF * RowBytes(kQ4_K, kHidden);
  const std::size_t dn_bytes = std::size_t(kExperts) * kHidden * RowBytes(down, kFF);
  void* gate = RandomDevice(gu_bytes, 0x1111U);
  void* up = RandomDevice(gu_bytes, 0x2222U);
  void* dw = RandomDevice(dn_bytes, 0x3333U);
  const std::size_t scratch_bytes = qfn_mmq_expert_v2k_scratch_bytes(8);
  void* scratch = nullptr;
  Check(hipMalloc(&scratch, scratch_bytes + 4096), "scratch");
  for (int T : kTokens) {
    if (T < 2) continue;
    IdSets ids = MakeIds(T, nsets, 0xABCDU + T);
    float* x = DeviceFloats(std::size_t(T) * kHidden, 1.F, 55U);
    float* ge = DeviceFloats(std::size_t(T) * kUsed * kFF, 0.F, 4U);
    float* ue = DeviceFloats(std::size_t(T) * kUsed * kFF, 0.F, 5U);
    float* out = DeviceFloats(std::size_t(T) * kUsed * kHidden, 0.F, 6U);
    const double moved = MeanDistinct(ids, iters) * (2.0 * kFF * RowBytes(kQ4_K, kHidden) + double(kHidden) * RowBytes(down, kFF));
    auto call = [&](int i) {
      const std::int32_t* id = ids.device + std::size_t(i % nsets) * T * kUsed;
      return qfn_mmq_expert_v2k(static_cast<int>(kQ4_K.type), static_cast<int>(down.type), gate, up, dw, x, id, ge, ue, out, T, scratch, scratch_bytes, nullptr);
    };
    const std::string name = std::string("Q4_K+") + down.name;
    Report("v2k|" + name + "|" + std::to_string(T), "chain", name.c_str(), "v2k", T, MeanDistinct(ids, iters), TimeUs(call, iters), moved);
    (void)hipFree(x);
    (void)hipFree(ge);
    (void)hipFree(ue);
    (void)hipFree(out);
    (void)hipFree(ids.device);
  }
  (void)hipFree(gate);
  (void)hipFree(up);
  (void)hipFree(dw);
  (void)hipFree(scratch);
}

double Us(const std::string& key) {
  const auto it = g_results.find(key);
  return it == g_results.end() ? -1 : it->second.us;
}

/// Modelled expert time per verify round (all 48 layers) from the measured per-layer microseconds and the layer counts of the two files.
void Model(int T) {
  const std::string t = std::to_string(T);
  // Production Huihui UD-Q4_K_XL: 47 layers Q4_K gate/up (V2K chain; down Q5_1 on 42, Q8_0 on 5), 1 layer Q5_K gate/up with Q5_1 down (gated + vec).
  const double p = 42 * Us("v2k|Q4_K+Q5_1|" + t) + 5 * Us("v2k|Q4_K+Q8_0|" + t) + (Us("gu|Q5_K|gated|" + t) + Us("down|Q5_1|vec|" + t));
  // ISTA IQ3_S file: gate/up IQ3_XXS 17, IQ3_S 10, IQ2_S 20, IQ4_XS 1 layers; down IQ4_NL 39, Q2_0 9 layers (pairing of the two per layer ignored: independent sums).
  const double r = 17 * Us("gu|IQ3_XXS|vec-pair|" + t) + 10 * Us("gu|IQ3_S|vec-pair|" + t) + 20 * Us("gu|IQ2_S|vec-pair|" + t) + 1 * Us("gu|IQ4_XS|vec-pair|" + t) +
                   39 * Us("down|IQ4_NL|vec|" + t) + 9 * Us("down|Q2_0|vec|" + t);
  std::printf("MODEL T=%d expert_ms_per_round P=%.2f R2p=%.2f ratio_R2p_over_P=%.2f (P: V2K chain on 47 layers; R2p: generic vec-pair + vec on all 48; swiglu and launch gaps not included)\n", T, p / 1000.0,
              r / 1000.0, p > 0 ? r / p : 0.0);
}

#include "expert_iq_decode.hpp"
#include "expert_real_decode.hpp"

}  // namespace

int main(int argc, char** argv) try {
  if (const auto result = TryRealBench(argc, argv))
    return *result;
  if (const auto result = TryIqBench(argc, argv))
    return *result;
  int iters = 40, nsets = 16;
  bool strict_timing = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--strict-timing") == 0) strict_timing = true;
    else if (i + 1 < argc && std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[++i]);
    else if (i + 1 < argc && std::strcmp(argv[i], "--sets") == 0) nsets = std::atoi(argv[++i]);
    else throw std::runtime_error("Unknown or incomplete bench option");
  }
  if (iters < 1 || nsets < 1) throw std::runtime_error("Bench counts must be positive");
  if (qfn_mmq_init(0) != 0) std::fprintf(stderr, "warning: qfn_mmq_init(0) returned non-zero; continuing (the kernels initialise lazily in the other GPU tests)\n");
  std::printf("expert_decode_bench iters=%d sets=%d geometry experts=%d used=%d hidden=%d ff=%d\n", iters, nsets, kExperts, kUsed, kHidden, kFF);
  BenchV2k(kQ5_1, iters, nsets);
  BenchV2k(kQ8_0, iters, nsets);
  BenchGateUp(kQ4_K, iters, nsets, true);
  BenchGateUp(kQ5_K, iters, nsets, true);
  for (const Fmt* f : {&kIQ3_XXS, &kIQ3_S, &kIQ2_S, &kIQ2_XXS, &kIQ2_XS, &kIQ1_M, &kIQ4_XS}) BenchGateUp(*f, iters, nsets, false);
  for (const Fmt* f : {&kQ5_1, &kQ8_0, &kIQ4_NL, &kQ2_0}) BenchDown(*f, iters, nsets);
  for (int T : kTokens)
    if (T >= 2) Model(T);   // P's chain exists for 2..8 tokens only
  return TimingStatus(strict_timing);
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
