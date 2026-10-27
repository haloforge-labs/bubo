












#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

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
constexpr Fmt kQ2_0{"Q2_0", GgmlType::kQ2_0, 64, 18};

std::size_t RowBytes(const Fmt& f, int cols) { return std::size_t(cols) / f.block * f.bytes; }

std::uint32_t NextRandom(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

/// Device buffer of random bytes (a 16 MiB random chunk repeated: the kernels' time does not depend on the values).
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

double MeanDistinct(const IdSets& s) {
  double sum = 0;
  for (int d : s.distinct) sum += d;
  return sum / s.distinct.size();
}

/// Mean microseconds per call over `iters` calls (set i % sets), after warm-up. Returns -1 when a call fails.
double TimeUs(const std::function<int(int)>& call, int iters) {
  for (int i = 0; i < 6; ++i)
    if (call(i) != 0) return -1;
  Check(hipDeviceSynchronize(), "warm-up");
  hipEvent_t a, b;
  Check(hipEventCreate(&a), "event");
  Check(hipEventCreate(&b), "event");
  Check(hipEventRecord(a, nullptr), "record");
  for (int i = 0; i < iters; ++i)
    if (call(i) != 0) return -1;
  Check(hipEventRecord(b, nullptr), "record");
  Check(hipEventSynchronize(b), "sync");
  float ms = 0;
  Check(hipEventElapsedTime(&ms, a, b), "elapsed");
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return 1000.0 * ms / iters;
}

struct Result {
  double us = -1, gbps = 0;
};

std::map<std::string, Result> g_results;   // key "gu|<fmt>|<route>|T" and "down|<fmt>|vec|T" and "v2k|<gu>+<down>|T"

void Report(const std::string& key, const char* kind, const char* fmt, const char* route, int tokens, double distinct, double us, double bytes) {
  Result r;
  r.us = us;
  r.gbps = us > 0 ? bytes / (us * 1e-6) / 1e9 : 0;
  g_results[key] = r;
  std::printf("BENCH kind=%s fmt=%s route=%s T=%d distinct=%.1f us=%.1f GBps=%.1f%s\n", kind, fmt, route, tokens, distinct, us, r.gbps, us < 0 ? " FAILED" : "");
  std::fflush(stdout);
}

const int kTokens[] = {1, 2, 3, 4, 8};

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
    const double moved = MeanDistinct(ids) * 2.0 * kFF * RowBytes(f, kHidden);
    auto vec = [&](int i) {
      const std::int32_t* id = ids.device + std::size_t(i % nsets) * T * kUsed;
      return qfn_mmq_moe_vec(static_cast<int>(f.type), wa, x, id, oa, kFF, kHidden, T, kExperts, kUsed, nullptr, wb, ob);
    };
    Report(std::string("gu|") + f.name + "|vec-pair|" + std::to_string(T), "gu", f.name, "vec-pair", T, MeanDistinct(ids), TimeUs(vec, iters), moved);
    if (gated) {
      auto fused = [&](int i) {
        const std::int32_t* id = ids.device + std::size_t(i % nsets) * T * kUsed;
        return qfn_mmq_moe_gated_vec(static_cast<int>(f.type), wa, wb, x, id, oa, kFF, kHidden, T, kExperts, kUsed, nullptr);
      };
      Report(std::string("gu|") + f.name + "|gated|" + std::to_string(T), "gu", f.name, "gated", T, MeanDistinct(ids), TimeUs(fused, iters), moved);
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
    const double moved = MeanDistinct(ids) * kHidden * RowBytes(f, kFF);
    auto vec = [&](int i) {
      const std::int32_t* id = ids.device + std::size_t(i % nsets) * slots;
      return qfn_mmq_moe_vec(static_cast<int>(f.type), w, xs, id, out, kHidden, kFF, slots, kExperts, 1, nullptr);
    };
    Report(std::string("down|") + f.name + "|vec|" + std::to_string(T), "down", f.name, "vec", T, MeanDistinct(ids), TimeUs(vec, iters), moved);
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
    const double moved = MeanDistinct(ids) * (2.0 * kFF * RowBytes(kQ4_K, kHidden) + double(kHidden) * RowBytes(down, kFF));
    auto call = [&](int i) {
      const std::int32_t* id = ids.device + std::size_t(i % nsets) * T * kUsed;
      return qfn_mmq_expert_v2k(static_cast<int>(kQ4_K.type), static_cast<int>(down.type), gate, up, dw, x, id, ge, ue, out, T, scratch, scratch_bytes, nullptr);
    };
    const std::string name = std::string("Q4_K+") + down.name;
    Report("v2k|" + name + "|" + std::to_string(T), "chain", name.c_str(), "v2k", T, MeanDistinct(ids), TimeUs(call, iters), moved);
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

}  // namespace

int main(int argc, char** argv) try {
  int iters = 40, nsets = 16;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--sets") == 0) nsets = std::atoi(argv[i + 1]);
  }
  if (qfn_mmq_init(0) != 0) std::fprintf(stderr, "warning: qfn_mmq_init(0) returned non-zero; continuing (the kernels initialise lazily in the other GPU tests)\n");
  std::printf("expert_decode_bench iters=%d sets=%d geometry experts=%d used=%d hidden=%d ff=%d\n", iters, nsets, kExperts, kUsed, kHidden, kFF);
  BenchV2k(kQ5_1, iters, nsets);
  BenchV2k(kQ8_0, iters, nsets);
  BenchGateUp(kQ4_K, iters, nsets, true);
  BenchGateUp(kQ5_K, iters, nsets, true);
  for (const Fmt* f : {&kIQ3_XXS, &kIQ3_S, &kIQ2_S, &kIQ4_XS}) BenchGateUp(*f, iters, nsets, false);
  for (const Fmt* f : {&kQ5_1, &kQ8_0, &kIQ4_NL, &kQ2_0}) BenchDown(*f, iters, nsets);
  for (int T : kTokens)
    if (T >= 2) Model(T);   // P's chain exists for 2..8 tokens only
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
