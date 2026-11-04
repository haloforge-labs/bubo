#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kv_q8_prefill_view.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace qfn = gufo::models::qwen38_flash_next;
namespace {
using Type = qfn::KvCacheType;
constexpr Type kQ8 = Type::kQ8_0, kF16 = Type::kF16;
constexpr std::uint32_t kHeads = 24, kKvHeads = 2, kDim = 256, kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim, kKvWidth = kKvHeads * kDim;
constexpr std::size_t kRow = qfn::KvRowBytes(kKvWidth, kQ8);

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid GPU KV payload/input accepted by portable codec");
}

void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

template<typename T>
class Buffer {
public:
  explicit Buffer(std::size_t count) : count_(count) {
    void* p = nullptr;
    Hip(hipMalloc(&p, bytes()));
    p_ = static_cast<T*>(p);
    Hip(hipMemset(p_, 0, bytes()));
  }
  ~Buffer() { (void)hipFree(p_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() { return p_; }
  std::size_t bytes() const { return count_ * sizeof(T); }
  void Put(const std::vector<T>& v) {
    Require(v.size() == count_, "upload size");
    Hip(hipMemcpy(p_, v.data(), bytes(), hipMemcpyHostToDevice));
  }
  std::vector<T> Host() {
    Hip(hipGetLastError());
    std::vector<T> v(count_);
    Hip(hipMemcpy(v.data(), p_, bytes(), hipMemcpyDeviceToHost));
    return v;
  }

private:
  T* p_{};
  std::size_t count_;
};

template<typename T>
void Exact(const std::vector<T>& a, const std::vector<T>& b, const char* label,
           std::size_t count = 0) {
  Require(a.size() == b.size(), "comparison size");
  if (count == 0)
    count = a.size();
  for (std::size_t i = 0; i < count; ++i) {
    if constexpr (std::is_same_v<T, float>)
      Require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite output");
    if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0)
      throw std::runtime_error(std::string(label) + " at " + std::to_string(i));
  }
}

std::uint32_t RandomWord(std::uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

std::vector<float> Values(std::size_t count, std::uint32_t seed,
                          float scale = 1.0F) {
  std::vector<float> v(count);
  for (float& x : v)
    x = scale * (static_cast<int>(RandomWord(seed) & 0xFFFFU) - 32768) /
        32768.0F;
  return v;
}

// Independent of the HIP insertion/load implementations: the portable codec
// defines canonical bytes and rounds reconstructed operands to F16 once.
std::vector<std::byte> Encode(const std::vector<float>& v) {
  Require(v.size() % kKvWidth == 0, "encode rows");
  std::vector<std::byte> bytes(v.size() / kKvWidth * kRow);
  for (std::size_t t = 0; t < v.size() / kKvWidth; ++t)
    qfn::EncodeKvRow(std::span(v).subspan(t * kKvWidth, kKvWidth), kQ8,
                     std::span(bytes).subspan(t * kRow, kRow));
  return bytes;
}
std::vector<__half> Decode(const std::vector<std::byte>& bytes) {
  std::vector<__half> half(bytes.size() / kRow * kKvWidth);
  std::vector<float> row(kKvWidth);
  for (std::size_t t = 0; t < bytes.size() / kRow; ++t) {
    qfn::DecodeKvRow(std::span(bytes).subspan(t * kRow, kRow), kQ8, row);
    std::transform(row.begin(), row.end(), half.begin() + t * kKvWidth,
                   [](float x) { return __float2half(x); });
  }
  return half;
}

// Independent FP64 formula for the first query, retaining attention_ops_test's
// 1e-6 gated scalar envelope. This is not an exact scalar-versus-WMMA claim.
void CheckFormula(const std::vector<float>& query,
                  const std::vector<float>& gate, const std::vector<__half>& k,
                  const std::vector<__half>& v, const std::uint32_t* mask,
                  std::uint32_t start, const std::vector<float>& actual) {
  std::vector<std::uint32_t> keys;
  for (std::uint32_t j = 0; j <= start; ++j)
    if (mask == nullptr || j >= (start + 1) / kRatio * kRatio ||
        (mask[j / kRatio / 32] & (1U << (j / kRatio % 32))))
      keys.push_back(j);
  std::vector<double> scores(keys.size());
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const auto head = h / (kHeads / kKvHeads) * kDim;
    double maximum = -INFINITY;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      double dot = 0;
      for (std::uint32_t d = 0; d < kDim; ++d)
        dot += double(query[h * kDim + d]) *
               __half2float(k[std::size_t(keys[i]) * kKvWidth + head + d]);
      scores[i] = dot / std::sqrt(double(kDim));
      maximum = std::max(maximum, scores[i]);
    }
    double denominator = 0;
    for (double& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (std::uint32_t d = 0; d < kDim; ++d) {
      double sum = 0;
      for (std::size_t i = 0; i < keys.size(); ++i)
        sum += scores[i] *
               __half2float(v[std::size_t(keys[i]) * kKvWidth + head + d]);
      const auto index = h * kDim + d;
      const double expected =
          sum / denominator / (1 + std::exp(-double(gate[index])));
      Require(std::isfinite(actual[index]) &&
                  std::abs(actual[index] - expected) <= 1e-6,
              "scalar exceeds unchanged FP64 envelope");
    }
  }
}

struct Graph {
  hipStream_t stream{};
  hipGraph_t graph{};
  hipGraphExec_t exec{};
  Graph() { Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
  ~Graph() {
    if (exec)
      (void)hipGraphExecDestroy(exec);
    if (graph)
      (void)hipGraphDestroy(graph);
    (void)hipStreamDestroy(stream);
  }
  void Begin() {
    Hip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  }
  void End() {
    Hip(hipStreamEndCapture(stream, &graph));
    Hip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  }
  void Replay() {
    Hip(hipGraphLaunch(exec, stream));
    Hip(hipStreamSynchronize(stream));
  }
};

void CheckStore() {
  constexpr std::uint32_t start = 3, n = 17;
  auto input = Values(std::size_t(n) * kKvWidth, 891);
  // Zero blocks, exact half-integer ties, and stored-scale underflow. Codes
  // must use the unrounded F32 scale, not the serialized half scale.
  std::fill_n(input.begin(), 96, 0.0F);
  input[32] = 127.0F;
  input[33] = 0.5F;
  input[34] = -0.5F;
  input[35] = 1.5F;
  input[36] = -1.5F;
  input[64] = 1e-7F;
  input[65] = -1e-7F;
  Buffer<float> src(input.size());
  Buffer<std::byte> cache((start + n + 2) * kRow);
  Buffer<std::uint32_t> pos(1);
  src.Put(input);
  const auto encoded = Encode(input);
  std::vector<std::byte> expected(cache.bytes(), std::byte{0xA5});
  std::copy(encoded.begin(), encoded.end(), expected.begin() + start * kRow);
  for (const auto chunk : {17U, 17U, 1U, 2U, 3U, 7U, 9U}) {
    Hip(hipMemset(cache.get(), 0xA5, cache.bytes()));
    for (std::uint32_t off = 0; off < n; off += chunk) {
      pos.Put({start + off});
      q::StoreKv(src.get() + std::size_t(off) * kKvWidth, cache.get(),
                 std::min(chunk, n - off), kKvWidth, pos.get(), nullptr, kQ8);
    }
    Exact(expected, cache.Host(), "StoreKv bytes/repeat/chunks/guards");
  }
  // Keep the first row fixed while changing every launch companion.
  for (std::size_t i = kKvWidth; i < input.size(); ++i)
    input[i] *= 31.0F;
  src.Put(input);
  pos.Put({start});
  q::StoreKv(src.get(), cache.get(), n, kKvWidth, pos.get(), nullptr, kQ8);
  const auto companions = Encode(input);
  std::copy(companions.begin(), companions.end(),
            expected.begin() + start * kRow);
  Exact(expected, cache.Host(), "StoreKv companion-local scaling");
}

void CheckStoreScalesAndPopulation() {
  constexpr std::uint32_t n = 2048, start = 3;
  auto input = Values(std::size_t(n) * kKvWidth, 1907, 19.0F);
  std::size_t fixture_row = 0;
  const auto fixture = [&](float amax, float center, std::uint16_t scale,
                           std::array<int, 3> codes) {
    auto block = std::span(input).subspan(fixture_row++ * kKvWidth, 32);
    std::fill(block.begin(), block.end(), 0.0F);
    std::array<int, 32> expected{};
    block[0] = amax;
    block[1] = -amax;
    expected[0] = 127;
    expected[1] = -127;
    const std::array<float, 3> neighbours = {
        std::nextafter(center, 0.0F), center,
        std::nextafter(center, std::numeric_limits<float>::infinity())};
    for (std::size_t j = 0; j < neighbours.size(); ++j) {
      block[2 + j] = neighbours[j];
      block[5 + j] = -neighbours[j];
      expected[2 + j] = codes[j];
      expected[5 + j] = -codes[j];
    }
    // Assert the CPU oracle's known bytes before asking HIP to match it.
    // No reconstruction or roundtrip is used to generate these expectations.
    std::array<std::byte, 34> bytes{};
    qfn::EncodeKvRow(block, kQ8, bytes);
    Require(
        bytes[0] == std::byte(scale & 255) && bytes[1] == std::byte(scale >> 8),
        "CPU non-power-of-two scale fixture");
    for (std::size_t j = 0; j < expected.size(); ++j)
      Require(bytes[2 + j] == std::byte(static_cast<std::uint8_t>(expected[j])),
              "CPU non-power-of-two signed code fixture");
  };
  // Every d=3 half-integer tie and +/-1 input ULP, with both sign mirrors.
  for (int k = 0; k < 127; ++k)
    fixture(381.0F, (k + 0.5F) * 3.0F, 0x4200, {k, k + 1, k + 1});
  // Exact-rational/binary32-RNE fixtures, mirrored in kv_cache_test.cpp.
  // Nonexact d and reciprocal-sensitive cases exercise BOTH divisions.
  struct Fixture {
    std::uint32_t amax, center;
    std::uint16_t scale;
    std::array<int, 3> codes;
  };
  constexpr Fixture fixtures[] = {
      {0x3f80003c, 0x3b810241, 0x2008, {0, 1, 1}},
      {0x3f80003c, 0x3f00003c, 0x2008, {63, 63, 64}},
      {0x41200000, 0x3f53a74f, 0x2d0a, {11, 11, 11}},
      {0x41200000, 0x411f5ebd, 0x2d0a, {126, 126, 127}},
      {0x447a0000, 0x407bf7f0, 0x47e0, {0, 1, 1}},
      {0x447a0000, 0x42a55ab6, 0x47e0, {10, 11, 11}},
      {0x447a0000, 0x43fa0000, 0x47e0, {63, 64, 64}},
      {0x3dcccccd, 0x3c077551, 0x1273, {10, 11, 11}},
      // Pin roundf(RNE(x/d)), not rounding the unrounded exact ratio.
      {0x43be8003, 0x40900002, 0x4200, {1, 2, 2}},
  };
  for (const auto& f : fixtures)
    fixture(std::bit_cast<float>(f.amax), std::bit_cast<float>(f.center),
            f.scale, f.codes);
  // F32 subnormal scales must not silently become zero before codes are
  // computed; their stored F16 scales are zero but canonical codes survive.
  fixture(std::numeric_limits<float>::min(), 0.0F, 0, {0, 0, 0});
  fixture(std::numeric_limits<float>::min() / 8, 0.0F, 0, {0, 0, 0});
  // 1,048,576 inserted elements, including 138 independently checked blocks;
  // all remaining elements are seeded random. Small chunk/companion cases
  // stay in CheckStore rather than multiplying this larger population.
  Buffer<float> src(input.size());
  Buffer<std::byte> cache((start + n + 2) * kRow);
  Buffer<std::uint32_t> pos(1);
  src.Put(input);
  pos.Put({start});
  Hip(hipMemset(cache.get(), 0xA5, cache.bytes()));
  const auto encoded = Encode(input);
  std::vector<std::byte> expected(cache.bytes(), std::byte{0xA5});
  std::copy(encoded.begin(), encoded.end(), expected.begin() + start * kRow);
  q::StoreKv(src.get(), cache.get(), n, kKvWidth, pos.get(), nullptr, kQ8);
  Exact(expected, cache.Host(), "StoreKv million-element population/scales");
}

void CheckStoreInvalid() {
  // Invalid geometry/type must throw before dereferencing or launching.
  Reject([] { q::StoreKv(nullptr, nullptr, 1, 31, nullptr, nullptr, kQ8); });
  Reject([] {
    q::StoreKv(nullptr, nullptr, 1, kKvWidth, nullptr, nullptr,
               static_cast<Type>(42));
  });
  constexpr std::uint32_t start = 2, n = 2;
  const std::array<float, 8> bad = {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::max(),
                                    -std::numeric_limits<float>::max(),
                                    70000.0F,
                                    -70000.0F,
                                    65504.0F};
  auto input = Values(std::size_t(n) * kKvWidth, 991);
  for (std::size_t b = 0; b < bad.size(); ++b)
    input[b * 32 + (b % 2 ? 31 : 0)] = bad[b];
  Buffer<float> src(input.size());
  Buffer<std::byte> cache((start + n + 2) * kRow);
  Buffer<std::uint32_t> pos(1);
  src.Put(input);
  pos.Put({start});
  Hip(hipMemset(cache.get(), 0xA5, cache.bytes()));
  q::StoreKv(src.get(), cache.get(), n, kKvWidth, pos.get(), nullptr, kQ8);
  // Download raw bytes, NEVER call attention or Decode on these poisoned rows.
  const auto raw = cache.Host();
  const auto row = std::span(raw).subspan(start * kRow, kRow);
  for (std::size_t b = 0; b < bad.size(); ++b) {
    const auto block = row.subspan(b * 34, 34);
    const auto scale = std::to_integer<unsigned>(block[0]) |
                       (std::to_integer<unsigned>(block[1]) << 8);
    Require((scale & 0x7c00) == 0x7c00 && (scale & 0x03ff) != 0,
            "invalid GPU input did not surface as a NaN scale");
    Require(std::all_of(block.begin() + 2, block.end(),
                        [](std::byte x) { return x == std::byte{0}; }),
            "invalid GPU block did not zero its codes");
    Reject([&] { qfn::ValidateKvRow(block, 32, kQ8); });
    Reject([&] {
      std::array<std::byte, 34> bytes{};
      qfn::EncodeKvRow(std::span(input).subspan(b * 32, 32), kQ8, bytes);
    });
  }
  Reject([&] { qfn::ValidateKvRow(row, kKvWidth, kQ8); });
  const auto companion =
      Encode(std::vector<float>(input.begin() + kKvWidth, input.end()));
  std::vector<std::byte> expected(raw.size(), std::byte{0xA5});
  // Preserve the intentionally invalid row; all other rows/guards stay exact.
  std::copy(row.begin(), row.end(), expected.begin() + start * kRow);
  std::copy(companion.begin(), companion.end(),
            expected.begin() + (start + 1) * kRow);
  Exact(expected, raw, "invalid StoreKv companion/guards");
  qfn::ValidateKvRow(std::span(raw).subspan((start + 1) * kRow, kRow), kKvWidth,
                     kQ8);
}

void CheckAttention(std::uint32_t start, std::uint32_t n, bool sparse) {
  const auto context = start + n;
  const auto words = ((context + kRatio - 1) / kRatio + 31) / 32;
  const std::size_t count = std::size_t(n) * kQWidth;
  const auto kb = Encode(Values(std::size_t(context) * kKvWidth, 891));
  const auto vb = Encode(Values(std::size_t(context) * kKvWidth, 347));
  Buffer<std::byte> keys(kb.size()), values(vb.size());
  Buffer<__half> kh(std::size_t(context) * kKvWidth), vh(kh.bytes() / 2);
  Buffer<float> queries(count), gates(count), out(count);
  Buffer<float> partials(std::size_t(n) * kHeads * 8 * (kDim + 2));
  Buffer<std::uint32_t> masks(std::size_t(n) * words), pos(1);
  keys.Put(kb);
  values.Put(vb);
  kh.Put(Decode(kb));
  vh.Put(Decode(vb));
  const auto reference_k = kh.Host(), reference_v = vh.Host();
  Hip(hipMemset(kh.get(), 0xa5, kh.bytes()));
  Hip(hipMemset(vh.get(), 0xa5, vh.bytes()));
  Require(q::DequantizeQ8Kv(keys.get(), values.get(), kh.get(), vh.get(),
                            context, kKvWidth, nullptr),
          "transient q8 view rejected valid rows");
  Exact(reference_k, kh.Host(), "transient K view");
  Exact(reference_v, vh.Host(), "transient V view");
  queries.Put(Values(count, 412, 4.0F));
  gates.Put(Values(count, 721, 3.0F));
  pos.Put({start});
  std::vector<std::uint32_t> mask(std::size_t(n) * words);
  for (std::uint32_t t = 0; t < n; ++t)
    for (std::uint32_t b = 0; b < (start + t + 1) / kRatio; ++b)
      if ((b + t) % 5 != 0)
        mask[std::size_t(t) * words + b / 32] |= 1U << (b % 32);
  masks.Put(mask);
  auto* selection = sparse ? masks.get() : nullptr;
  const auto scalar = [&](Type type, std::uint32_t splits) {
    Hip(hipMemset(out.get(), 0xFF, out.bytes()));
    q::Attention(
        queries.get(),
        type == kQ8 ? keys.get() : static_cast<const void*>(kh.get()),
        type == kQ8 ? values.get() : static_cast<const void*>(vh.get()),
        selection, words, out.get(), splits ? partials.get() : nullptr, splits,
        n, pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr, type);
    return out.Host();
  };
  for (const auto splits : {0U, 1U, 3U, 8U}) {
    const auto reference = scalar(kF16, splits);
    Exact(reference, scalar(kQ8, splits), "scalar decoded operands");
    if (splits == 0) {
      q::SigmoidMul(out.get(), gates.get(), count, nullptr);
      CheckFormula(queries.Host(), gates.Host(), kh.Host(), vh.Host(),
                   sparse ? mask.data() : nullptr, start, out.Host());
    }
  }
  // A kernel-level snapshot: raw packed bytes, NOT decode/re-encode. This
  // checks attention contexts, not full-model logits or session persistence.
  const auto uninterrupted = scalar(kQ8, 8);
  const auto saved_k = keys.Host(), saved_v = values.Host();
  Hip(hipMemset(keys.get(), 0, keys.bytes()));
  Hip(hipMemset(values.get(), 0, values.bytes()));
  keys.Put(saved_k);
  values.Put(saved_v);
  Exact(saved_k, keys.Host(), "restored K bytes");
  Exact(saved_v, values.Host(), "restored V bytes");
  Exact(uninterrupted, scalar(kQ8, 8), "restored attention context");
  const auto wmma = [&](Type type, std::uint32_t first, std::uint32_t rows,
                        bool last = false) {
    const auto offset = std::size_t(first) * kQWidth;
    if (!last)
      Hip(hipMemset(out.get() + offset, 0xFF,
                    std::size_t(rows) * kQWidth * sizeof(float)));
    Require(q::WmmaCausalAttention(
                queries.get() + offset, gates.get() + offset,
                type == kQ8 ? keys.get() : static_cast<const void*>(kh.get()),
                type == kQ8 ? values.get() : static_cast<const void*>(vh.get()),
                sparse ? masks.get() + std::size_t(first) * words : nullptr,
                words, out.get() + offset, rows, start + first, kHeads,
                kKvHeads, kDim, kRatio, nullptr, last, type),
            "WMMA rejected geometry");
    return out.Host();
  };
  const auto expected = wmma(kQ8, 0, n);
  Exact(expected, wmma(kF16, 0, n), "WMMA decoded operands");
  Exact(expected, wmma(kQ8, 0, n), "WMMA repeat");
  auto tail = expected;
  const auto begin =
      std::size_t(sparse ? (n - 1) / 4 * 4 : (n - 1) / 16 * 16) * kQWidth;
  std::fill(tail.begin(), tail.begin() + begin, -12345.0F);
  out.Put(std::vector<float>(count, -12345.0F));
  Exact(tail, wmma(kQ8, 0, n, true), "WMMA last-only/guards");
  if (sparse) {
    auto companions = mask;
    for (std::uint32_t t = 1; t < n; ++t)
      std::copy_n(mask.begin(), words,
                  companions.begin() + std::size_t(t) * words);
    masks.Put(companions);
    const auto changed_masks = wmma(kQ8, 0, n);
    Exact(changed_masks, wmma(kF16, 0, n), "WMMA companion-mask operands");
    masks.Put(mask);
    auto future = vb;
    std::fill(future.begin() + std::size_t(start + 1) * kRow, future.end(),
              std::byte{0});
    values.Put(future);
    vh.Put(Decode(future));
    const auto changed_values = wmma(kQ8, 0, n);
    Exact(changed_values, wmma(kF16, 0, n), "WMMA future-V operands");
    values.Put(vb);
    vh.Put(Decode(vb));
  }
  for (const auto cut : {1U, 2U, 3U, 4U, 7U, 8U, 9U, 16U}) {
    if (cut >= n)
      continue;
    auto prefix = vb;
    std::fill(prefix.begin() + std::size_t(start + cut) * kRow, prefix.end(),
              std::byte{0});
    // Upstream sparse WMMA groups four queries. Changing a call boundary
    // may change that group's key union; compare packed/decoded operands at
    // identical boundaries, not against another attention algorithm.
    values.Put(prefix);
    (void)wmma(kQ8, 0, cut);
    values.Put(vb);
    const auto split = wmma(kQ8, cut, n - cut);
    vh.Put(Decode(prefix));
    (void)wmma(kF16, 0, cut);
    vh.Put(Decode(vb));
    Exact(split, wmma(kF16, cut, n - cut), "WMMA split/prefix operands");
  }
}

void CheckPrefillViewConfiguredMax() {
  // Operator/policy regression only: no model, session or executor dispatch.
  // One full-prefix reconstruction, then packed-q8 and staged-F16 WMMA over
  // the last 256 queries. A compact view must end at 200000, not 262144.
  constexpr std::uint32_t configured = 200000, n = 256;
  constexpr std::uint32_t start = configured - n;
  constexpr std::uint32_t tail_start = start - 4 * kRatio;
  constexpr std::uint32_t words = (configured / kRatio + 31) / 32;
  constexpr std::size_t guard = 16;  // Half elements; keep uint4 alignment.
  constexpr std::size_t elements = std::size_t(configured) * kKvWidth;
  constexpr std::size_t count = std::size_t(n) * kQWidth;
  constexpr std::size_t peak_bytes =
      2 * configured * kRow + 2 * (elements + 2 * guard) * sizeof(__half) +
      3 * count * sizeof(float) +
      std::size_t(n) * words * sizeof(std::uint32_t);
  static_assert(peak_bytes == 647675008);
  static_assert(peak_bytes < std::size_t{2} * 1024 * 1024 * 1024);
  const auto capacity = q::detail::Q8PrefillViewCapacity(
      configured, 262144,
      q::WmmaCausalAttentionContextLimit(kHeads, kKvHeads, kDim, kRatio));
  Require(capacity == configured &&
              q::detail::Q8PrefillViewFits(capacity, start, n),
          "configured-max prefill must fit the compact view");

  // Buffer zero-initializes EVERY canonical block32 scale/code in the full
  // prefix. Only the last 272 rows need nonzero, position-varying fixtures;
  // no full-sized CPU F16 reference or float prefix is materialized.
  Buffer<std::byte> keys(std::size_t(capacity) * kRow), values(keys.bytes());
  Buffer<__half> kh(std::size_t(capacity) * kKvWidth + 2 * guard),
      vh(std::size_t(capacity) * kKvWidth + 2 * guard);
  auto* k_view = kh.get() + guard;
  auto* v_view = vh.get() + guard;
  const auto kb = Encode(
      Values(std::size_t(configured - tail_start) * kKvWidth, 891, 2.0F));
  const auto vb = Encode(
      Values(std::size_t(configured - tail_start) * kKvWidth, 347, 3.0F));
  Hip(hipMemcpy(keys.get() + std::size_t(tail_start) * kRow, kb.data(),
                kb.size(), hipMemcpyHostToDevice));
  Hip(hipMemcpy(values.get() + std::size_t(tail_start) * kRow, vb.data(),
                vb.size(), hipMemcpyHostToDevice));
  Hip(hipMemset(kh.get(), 0xA5, kh.bytes()));
  Hip(hipMemset(vh.get(), 0xA5, vh.bytes()));
  Require(q::DequantizeQ8Kv(keys.get(), values.get(), k_view, v_view, start + n,
                            kKvWidth, nullptr),
          "configured-max q8 reconstruction rejected");
  const auto download = [](const __half* p, std::size_t size) {
    std::vector<__half> result(size);
    Hip(hipGetLastError());
    Hip(hipMemcpy(result.data(), p, size * sizeof(__half),
                  hipMemcpyDeviceToHost));
    return result;
  };
  const auto check_guards = [&] {
    std::vector<__half> canary(guard);
    std::memset(canary.data(), 0xA5, guard * sizeof(__half));
    Exact(canary, download(kh.get(), guard), "configured-max K leading guard");
    Exact(canary, download(vh.get(), guard), "configured-max V leading guard");
    Exact(canary, download(k_view + elements, guard),
          "configured-max K trailing guard");
    Exact(canary, download(v_view + elements, guard),
          "configured-max V trailing guard");
  };
  check_guards();
  const auto tail_offset = std::size_t(tail_start) * kKvWidth;
  Exact(Decode(kb), download(k_view + tail_offset, elements - tail_offset),
        "configured-max K tail reconstruction");
  Exact(Decode(vb), download(v_view + tail_offset, elements - tail_offset),
        "configured-max V tail reconstruction");
  const std::vector<__half> zero_block(kRatio * kKvWidth, __float2half(0.0F));
  Exact(zero_block, download(k_view, zero_block.size()),
        "configured-max K prefix reconstruction");
  Exact(zero_block, download(v_view, zero_block.size()),
        "configured-max V prefix reconstruction");

  Buffer<float> queries(count), gates(count), out(count);
  Buffer<std::uint32_t> masks(std::size_t(n) * words);
  queries.Put(Values(count, 412, 4.0F));
  gates.Put(Values(count, 721, 3.0F));
  std::vector<std::uint32_t> mask(std::size_t(n) * words);
  for (std::uint32_t t = 0; t < n; ++t) {
    const auto row = std::size_t(t) * words;
    mask[row] = 1;  // Also visit the initialized beginning of the prefix.
    const auto complete = (start + t + 1) / kRatio;
    for (std::uint32_t b = complete - 4; b < complete; ++b)
      mask[row + b / 32] |= 1U << (b % 32);
  }
  // Five selected complete blocks plus the implicit 0..3-token causal tail:
  // at most 23 keys/query, two WMMA key tiles, despite the 200K address span.
  masks.Put(mask);
  const auto attend = [&](Type type) {
    Hip(hipMemset(out.get(), 0xFF, out.bytes()));
    Require(q::WmmaCausalAttention(
                queries.get(), gates.get(),
                type == kQ8 ? keys.get() : static_cast<const void*>(k_view),
                type == kQ8 ? values.get() : static_cast<const void*>(v_view),
                masks.get(), words, out.get(), n, start, kHeads, kKvHeads, kDim,
                kRatio, nullptr, false, type),
            "configured-max WMMA rejected");
    return out.Host();
  };
  // The control loads canonical q8 directly, independently of the staging
  // kernel. Keep the exact finite-output contract, not a relaxed tolerance.
  const auto expected = attend(kQ8);
  Exact(expected, attend(kF16), "configured-max WMMA decoded operands");
  check_guards();
}

void CheckPreparation(bool fused, const qfn::YarnRope& yarn) {
  // Two distinct cohorts: 1,048,576 K and V elements EACH per path/YaRN
  // mode. Keep the documented 1024-row fused minimum, not one huge batch.
  constexpr std::uint32_t n = 1024, cohorts = 2, start = 5;
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth), hidden = 2560;
  constexpr std::size_t count = std::size_t(n) * kQWidth;
  constexpr std::size_t weight_bytes =
      std::size_t(stride) * (hidden / 32) * 34 + 4096;
  constexpr std::size_t cache_bytes = (start + n + 2) * kRow;
  // Reference and actual Q/G/K/V buffers do not coexist. Projection buffers
  // are scoped separately too: peak explicit device allocation <150 MB.
  constexpr std::size_t peak_bytes =
      std::size_t(n) * stride * 4 + count * 2 * 4 + cache_bytes * 2 +
      weight_bytes + std::size_t(n) * hidden * 2 + 2 * kDim * 4 + 4;
  static_assert(peak_bytes < 150000000);
  Buffer<float> gamma(2 * kDim);
  Buffer<std::uint32_t> pos(1);
  pos.Put({start});
  auto g = Values(2 * kDim, 37, 0.5F);
  for (float& x : g)
    x += 1.0F;
  gamma.Put(g);
  for (std::uint32_t cohort = 0; cohort < cohorts; ++cohort) {
    Buffer<float> packed(std::size_t(n) * stride);
    std::vector<std::byte> w(fused ? weight_bytes : 0);
    std::vector<__half> half(fused ? std::size_t(n) * hidden : 0);
    if (fused) {
      // Synthetic row-major Q8_0 [13312,2560], no model weights loaded.
      // Random independent codes avoid the original periodic projection.
      std::uint32_t seed = 7919 + cohort;
      for (std::size_t b = 0; b < std::size_t(stride) * (hidden / 32); ++b) {
        w[b * 34] = std::byte{0};  // F16 0x1c00 = 1/256.
        w[b * 34 + 1] = std::byte{0x1c};
        for (std::size_t j = 0; j < 32; ++j)
          w[b * 34 + 2 + j] = std::byte(static_cast<std::uint8_t>(
              static_cast<int>(RandomWord(seed) % 63) - 31));
      }
      const auto x = Values(half.size(), 123 + cohort);
      std::transform(x.begin(), x.end(), half.begin(),
                     [](float v) { return __float2half(v); });
      Buffer<std::byte> weights(w.size());
      Buffer<__half> input(half.size());
      weights.Put(w);
      input.Put(half);
      Require(q::DenseF16Gemm(weights.get(), input.get(), packed.get(), n,
                              stride, hidden, nullptr),
              "projection rejected");
      Hip(hipDeviceSynchronize());
    } else {
      packed.Put(Values(std::size_t(n) * stride, 17 + cohort, 4.0F));
    }
    std::vector<float> reference_q, reference_g;
    std::vector<std::byte> reference_k, reference_v;
    {
      Buffer<float> qr(count), gr(count);
      Buffer<float> kr(std::size_t(n) * kKvWidth),
          vr(std::size_t(n) * kKvWidth);
      q::UnpackQGate(packed.get(), stride, qr.get(), gr.get(), kr.get(),
                     vr.get(), n, kHeads, kDim, kKvWidth, nullptr);
      q::RmsNormRows(qr.get(), gamma.get(), qr.get(), n * kHeads, kDim, 1,
                     1e-6F, nullptr);
      q::RmsNormRows(kr.get(), gamma.get() + kDim, kr.get(), n * kKvHeads, kDim,
                     1, 1e-6F, nullptr);
      q::Rope(qr.get(), n, kHeads, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
              yarn);
      q::Rope(kr.get(), n, kKvHeads, kDim, 64, pos.get(), 1e7F, nullptr,
              nullptr, yarn);
      reference_q = qr.Host();
      reference_g = gr.Host();
      reference_k = Encode(kr.Host());
      reference_v = Encode(vr.Host());
    }
    Buffer<float> qo(count), go(count);
    Buffer<std::byte> ko(cache_bytes), vo(cache_bytes);
    std::vector<std::byte> ek(cache_bytes, std::byte{0xA5}), ev = ek;
    std::copy(reference_k.begin(), reference_k.end(),
              ek.begin() + start * kRow);
    std::copy(reference_v.begin(), reference_v.end(),
              ev.begin() + start * kRow);
    const auto reset = [&] {
      Hip(hipMemset(qo.get(), 0xFF, qo.bytes()));
      Hip(hipMemset(go.get(), 0xFF, go.bytes()));
      Hip(hipMemset(ko.get(), 0xA5, ko.bytes()));
      Hip(hipMemset(vo.get(), 0xA5, vo.bytes()));
    };
    const auto check = [&] {
      Exact(reference_q, qo.Host(), "prepared Q");
      Exact(reference_g, go.Host(), "prepared gates");
      Exact(ek, ko.Host(), "prepared packed K/guards");
      Exact(ev, vo.Host(), "prepared packed V/guards");
    };
    if (fused) {
      Buffer<std::byte> weights(w.size());
      Buffer<__half> input(half.size());
      weights.Put(w);
      input.Put(half);
      reset();
      Require(q::AttentionF16Gemm(weights.get(), input.get(), gamma.get(),
                                  gamma.get() + kDim, qo.get(), go.get(),
                                  ko.get(), vo.get(), n, pos.get(), 1e7F, 1e-6F,
                                  nullptr, nullptr, yarn, kQ8),
              "fused attention projection rejected");
      check();
    } else {
      for (const bool prefill : {false, true}) {
        reset();
        Require(q::PrepareAttention(packed.get(), stride, gamma.get(),
                                    gamma.get() + kDim, qo.get(), go.get(),
                                    ko.get(), vo.get(), n, kHeads, kKvHeads,
                                    kDim, 64, pos.get(), 1e7F, 1e-6F, nullptr,
                                    nullptr, prefill, yarn, kQ8),
                "preparation rejected");
        check();
      }
      // Exercise the one-head-per-block decode preparation specialization
      // too; the wide cohort above uses four heads even with prefill=false.
      for (const auto chunk : {1U, 7U, 17U}) {
        for (std::uint32_t off = 0; off < 33; off += chunk) {
          pos.Put({start + off});
          Require(
              q::PrepareAttention(
                  packed.get() + std::size_t(off) * stride, stride, gamma.get(),
                  gamma.get() + kDim, qo.get() + std::size_t(off) * kQWidth,
                  go.get() + std::size_t(off) * kQWidth, ko.get(), vo.get(),
                  std::min(chunk, 33U - off), kHeads, kKvHeads, kDim, 64,
                  pos.get(), 1e7F, 1e-6F, nullptr, nullptr, false, yarn, kQ8),
              "narrow preparation rejected");
        }
        check();
      }
      pos.Put({start});
    }
  }
}

void CheckGraph() {
  constexpr std::uint32_t rows = 24;
  auto kb = Encode(Values(rows * kKvWidth, 100));
  auto vb = Encode(Values(rows * kKvWidth, 200));
  Buffer<std::byte> keys(kb.size()), values(vb.size());
  Buffer<__half> kh(rows * kKvWidth), vh(rows * kKvWidth);
  Buffer<float> k(kKvWidth), v(kKvWidth), query(kQWidth), out(kQWidth),
      ref(kQWidth);
  Buffer<std::uint32_t> pos(1);
  keys.Put(kb);
  values.Put(vb);
  query.Put(Values(kQWidth, 412, 4.0F));
  pos.Put({3});
  Graph graph;
  graph.Begin();
  q::StoreKv(k.get(), keys.get(), 1, kKvWidth, pos.get(), graph.stream, kQ8);
  q::StoreKv(v.get(), values.get(), 1, kKvWidth, pos.get(), graph.stream, kQ8);
  q::Attention(query.get(), keys.get(), values.get(), nullptr, 0, out.get(),
               nullptr, 0, 1, pos.get(), kHeads, kKvHeads, kDim, kRatio,
               graph.stream, kQ8);
  graph.End();
  for (const auto position : {3U, 7U, 12U, 7U}) {
    const auto kf = Values(kKvWidth, 891 + position);
    const auto vf = Values(kKvWidth, 347 + position);
    k.Put(kf);
    v.Put(vf);
    pos.Put({position});
    const auto a = Encode(kf), b = Encode(vf);
    std::copy(a.begin(), a.end(), kb.begin() + position * kRow);
    std::copy(b.begin(), b.end(), vb.begin() + position * kRow);
    Hip(hipMemsetAsync(out.get(), 0xFF, out.bytes(), graph.stream));
    Hip(hipMemset(ref.get(), 0xFF, ref.bytes()));
    graph.Replay();
    Exact(kb, keys.Host(), "graph StoreKv K/position");
    Exact(vb, values.Host(), "graph StoreKv V/position");
    kh.Put(Decode(kb));
    vh.Put(Decode(vb));
    q::Attention(query.get(), kh.get(), vh.get(), nullptr, 0, ref.get(),
                 nullptr, 0, 1, pos.get(), kHeads, kKvHeads, kDim, kRatio,
                 nullptr);
    Exact(ref.Host(), out.Host(), "graph narrow attention/position");
  }
}
}  // namespace

int main() {
  try {
    CheckPrefillViewConfiguredMax();
    CheckStore();
    CheckStoreScalesAndPopulation();
    CheckStoreInvalid();
    CheckAttention(37, 17, false);
    CheckAttention(2412, 17, true);  // Partial four-query sparse tile.
    CheckGraph();
    CheckPreparation(false, {});
    const auto yarn = qfn::MakeYarnRope(
        {.factor = 1.5625F, .original_context = 262144}, 64, 1e7F);
    CheckPreparation(false, yarn);
    CheckPreparation(true, {});
    CheckPreparation(true, yarn);
    std::cout << "q8 packed bytes, value-equal operand controls and graphs: "
                 "exact\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
