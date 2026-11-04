// Rotated operator acceptance, NOT model-quality or legacy-epoch evidence.
// Design: separate project docs/kv-q8-hadamard-v3-design.md, sections 4/9/10.
// Native execution is coordinator-owned. Keep kv_q8_ops_test.cpp unchanged.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <xmmintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"
#include "src/models/qwen38_flash_next/kv_rotation.hpp"

#ifdef __FAST_MATH__
#error "The operator test and its CPU oracle must not use fast-math"
#endif

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace qfn = gufo::models::qwen38_flash_next;
namespace {
using Type = qfn::KvCacheType;
using Kind = qfn::KvRotationKind;
constexpr Type kRotated = Type::kQ8_0Rotated, kQ8 = Type::kQ8_0;
constexpr Type kF16 = Type::kF16;
constexpr Kind kQK = Kind::kKeyQuery, kVO = Kind::kValueOutput;
constexpr std::uint32_t kHeads = 24, kKvHeads = 2, kDim = 256, kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim, kKvWidth = kKvHeads * kDim;
constexpr std::size_t kRow = qfn::KvRowBytes(kKvWidth, kRotated);
constexpr std::size_t kGuard = 256;
constexpr float kSentinel = -12345.0F;
// Bounds ALL simultaneous explicit device scratch, not just one allocation.
constexpr std::size_t kDeviceBudget = 128 * 1024 * 1024;
std::size_t live_bytes = 0, peak_bytes = 0;

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
void Cleanup(hipError_t error) noexcept {
  if (error != hipSuccess) {
    std::cerr << "fatal HIP test cleanup: " << hipGetErrorString(error) << '\n';
    std::abort();
  }
}
template<class F>
void Reject(F&& action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "expected invalid_argument rejection");
}

template<typename T>
class Buffer {
public:
  explicit Buffer(std::size_t count) : count_(count) {
    Require(count != 0 && count <= kDeviceBudget / sizeof(T),
            "invalid/oversize test allocation");
    Require(bytes() <= kDeviceBudget - live_bytes, "device scratch budget");
    void* p = nullptr;
    Hip(hipMalloc(&p, bytes()));
    p_ = static_cast<T*>(p);
    live_bytes += bytes();
    peak_bytes = std::max(peak_bytes, live_bytes);
  }
  ~Buffer() {
    Cleanup(hipFree(p_));
    live_bytes -= bytes();
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() { return p_; }
  std::size_t bytes() const { return count_ * sizeof(T); }
  void Put(const std::vector<T>& values) {
    Require(values.size() == count_, "upload size");
    Hip(hipMemcpy(p_, values.data(), bytes(), hipMemcpyHostToDevice));
  }
  std::vector<T> Host() {
    Hip(hipGetLastError());
    std::vector<T> values(count_);
    Hip(hipMemcpy(values.data(), p_, bytes(), hipMemcpyDeviceToHost));
    return values;
  }

private:
  T* p_{};
  std::size_t count_;
};

// Rejection checks are captured but NEVER replayed. An empty graph proves no
// kernel (including a compaction pre-launch) was enqueued. Invalid geometry
// therefore cannot dereference our deliberately undersized fixture buffers.
class NoLaunch {
public:
  NoLaunch() {
    Hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    const auto status =
        hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
    if (status != hipSuccess) {
      Cleanup(hipStreamDestroy(stream_));
      Hip(status);
    }
    capturing_ = true;
  }
  ~NoLaunch() {
    if (capturing_)
      Cleanup(hipStreamEndCapture(stream_, &graph_));
    if (graph_)
      Cleanup(hipGraphDestroy(graph_));
    Cleanup(hipStreamDestroy(stream_));
  }
  hipStream_t stream() { return stream_; }
  void Check() {
    Hip(hipStreamEndCapture(stream_, &graph_));
    capturing_ = false;
    std::size_t nodes = 0;
    Hip(hipGraphGetNodes(graph_, nullptr, &nodes));
    Require(nodes == 0, "rejected wrapper enqueued GPU work");
  }

private:
  hipStream_t stream_{};
  hipGraph_t graph_{};
  bool capturing_ = false;
};

template<typename T>
void Exact(const std::vector<T>& expected, const std::vector<T>& actual,
           const char* label) {
  Require(expected.size() == actual.size(), "comparison size");
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (std::memcmp(&expected[i], &actual[i], sizeof(T)) != 0)
      throw std::runtime_error(std::string(label) + " at " + std::to_string(i));
}
// D7: not an epsilon. All finite nonzero F32 bits must agree; only +/-0 is
// equivalent. Persisted bytes, gates and untouched sentinels use Exact instead.
void ValueExact(const std::vector<float>& expected,
                const std::vector<float>& actual, const char* label) {
  Require(expected.size() == actual.size(), "F32 comparison size");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Require(std::isfinite(expected[i]) && std::isfinite(actual[i]),
            "nonfinite F32 operator output");
    if ((expected[i] == 0 && actual[i] == 0) ||
        std::bit_cast<std::uint32_t>(expected[i]) ==
            std::bit_cast<std::uint32_t>(actual[i]))
      continue;
    throw std::runtime_error(std::string(label) + " at " + std::to_string(i));
  }
}
std::uint32_t Word(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}
std::vector<float> Values(std::size_t count, std::uint32_t seed,
                          float scale = 1.0F) {
  std::vector<float> values(count);
  for (auto& x : values)
    x = scale * (static_cast<int>(Word(seed) & 65535U) - 32768) / 32768.0F;
  return values;
}
void CpuRotate(std::span<float> values, Kind kind) {
  Require(values.size() % kDim == 0, "CPU head geometry");
  for (std::size_t off = 0; off < values.size(); off += kDim)
    qfn::RotateKvHead(values.subspan(off, kDim), kind);
}
std::vector<std::byte> Encode(const std::vector<float>& values,
                              Type type = kRotated) {
  Require(values.size() % kKvWidth == 0, "encode row geometry");
  std::vector<std::byte> bytes(values.size() / kKvWidth * kRow);
  for (std::size_t t = 0; t < values.size() / kKvWidth; ++t)
    qfn::EncodeKvRow(std::span(values).subspan(t * kKvWidth, kKvWidth), type,
                     std::span(bytes).subspan(t * kRow, kRow));
  return bytes;
}
std::vector<__half> Half(const std::vector<float>& values) {
  std::vector<__half> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(),
                 [](float x) { return __float2half(x); });
  return result;
}
std::vector<__half> Decode(const std::vector<std::byte>& bytes,
                           Type type = kRotated) {
  std::vector<float> values(bytes.size() / kRow * kKvWidth);
  for (std::size_t t = 0; t < bytes.size() / kRow; ++t)
    qfn::DecodeKvRow(std::span(bytes).subspan(t * kRow, kRow), type,
                     std::span(values).subspan(t * kKvWidth, kKvWidth));
  // Decode already rounded to F16 once: this conversion is exact, not another
  // independently quantized F16 producer. This is the operand-exact control.
  return Half(values);
}
std::vector<float> Guarded(const std::vector<float>& values) {
  std::vector<float> result(values.size() + 2 * kGuard, kSentinel);
  std::copy(values.begin(), values.end(), result.begin() + kGuard);
  return result;
}
void Guards(const std::vector<float>& values) {
  for (std::size_t i = 0; i < kGuard; ++i)
    Require(values[i] == kSentinel &&
                values[values.size() - kGuard + i] == kSentinel,
            "F32 sentinel overwritten");
}

void CheckRotation(std::vector<float> input, std::uint32_t n,
                   std::uint32_t heads, Kind kind) {
  Require(input.size() == std::size_t(n) * heads * kDim, "rotation fixture");
  auto expected = input;
  CpuRotate(expected, kind);
  Buffer<float> device(input.size() + 2 * kGuard);
  device.Put(Guarded(input));
  Require(q::RotateKvRows(device.get() + kGuard, n, heads, kDim, kind, nullptr),
          "RotateKvRows refused supported geometry");
  auto actual = device.Host();
  Guards(actual);
  ValueExact(Guarded(expected), actual, "FWHT nonzero bits/zero equivalence");
  // A second transform uses the SAME rounded CPU sequence, not an assumed
  // exact arbitrary-input round trip. Covers the output-inverse operation.
  CpuRotate(expected, kind);
  Require(q::RotateKvRows(device.get() + kGuard, n, heads, kDim, kind, nullptr),
          "inverse refused");
  actual = device.Host();
  Guards(actual);
  ValueExact(Guarded(expected), actual, "second FWHT bits");
}
void CheckTransforms() {
  Require(std::fegetround() == FE_TONEAREST && (_mm_getcsr() & 0x8040U) == 0,
          "CPU oracle requires RNE with FTZ/DAZ clear");
  for (auto kind : {kQK, kVO}) {
    const unsigned width = kind == kQK ? 256 : 64;
    const float scale = kind == kQK ? 0.0625F : 0.125F;
    std::vector<float> bases(256 * kDim);
    for (unsigned basis = 0; basis < 256; ++basis) {
      bases[basis * kDim + basis] = 1;
      std::array<float, kDim> cpu{};
      cpu[basis] = 1;
      qfn::RotateKvHead(cpu, kind);
      for (unsigned d = 0; d < kDim; ++d) {
        const float expected =
            d / width != basis / width                         ? 0.0F
            : std::popcount((d % width) & (basis % width)) % 2 ? -scale
                                                               : scale;
        Require(cpu[d] == expected, "independent Sylvester basis oracle");
      }
    }
    CheckRotation(bases, 128, 2, kind);  // every basis and all four H64 blocks
    for (auto heads : {1U, 2U, 24U})
      for (auto n : {1U, 3U, 8U, 33U})
        CheckRotation(Values(std::size_t(n) * heads * kDim, 123 + n + heads), n,
                      heads, kind);
    std::vector<float> edges(8 * kDim);
    std::uint32_t seed = 0x6d2b79f5;
    for (unsigned d = 0; d < kDim; ++d) {
      edges[d] = -0.0F;
      edges[kDim + d] = d % 2 ? -0.0F : 0.0F;
      edges[2 * kDim + d] =
          std::bit_cast<float>((Word(seed) & 0x807fffffU) | ((d % 138) << 23));
      edges[3 * kDim + d] =
          std::bit_cast<float>((d % 2 ? 0x80000000U : 0U) | (d % 17));
      edges[4 * kDim + d] = std::numeric_limits<float>::min();
      edges[5 * kDim + d] = d % 2 ? -1e-7F : 1e-7F;
      edges[6 * kDim + d] = std::numeric_limits<float>::max() / 32;
      edges[7 * kDim + d] = d % 2 ? -4096.0F : 4096.0F;
    }
    CheckRotation(edges, 4, 2, kind);
    std::array<float, kDim> zero;
    zero.fill(-0.0F);
    qfn::RotateKvHead(zero, kind);
    for (unsigned d = 0; d < kDim; ++d)
      Require(std::bit_cast<std::uint32_t>(zero[d]) ==
                  (d % width == 0 ? 0x80000000U : 0U),
              "CPU strict signed-zero oracle");
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
      auto head = Values(kDim, 77);
      head[127] = bad;
      const auto before = head;
      Reject([&] { qfn::RotateKvHead(head, kind); });
      Exact(before, head, "transactional nonfinite CPU rejection");
    }
    std::vector<float> overflow(kDim, std::numeric_limits<float>::max());
    const auto before = overflow;
    Reject([&] { qfn::RotateKvHead(overflow, kind); });
    Exact(before, overflow, "transactional CPU transform overflow");
  }
  std::array<float, kDim> head{};
  for (auto size : {0U, 1U, 64U, 255U})
    Reject([&] { qfn::RotateKvHead(std::span(head).first(size), kQK); });
  Reject([&] { qfn::RotateKvHead(head, static_cast<Kind>(99)); });
}

void PoisonBlock(std::span<const std::byte> block) {
  const auto scale = std::to_integer<unsigned>(block[0]) |
                     (std::to_integer<unsigned>(block[1]) << 8);
  Require((scale & 0x7c00U) == 0x7c00U && (scale & 0x03ffU) != 0,
          "invalid GPU input did not NaN-poison scale");
  Require(std::all_of(block.begin() + 2, block.end(),
                      [](std::byte x) { return x == std::byte{0}; }),
          "invalid GPU block did not zero codes");
}
void CheckRotatedStore() {
  constexpr unsigned n = 8, start = 3;
  for (auto kind : {kQK, kVO}) {
    auto input = Values(n * kKvWidth, 297);
    std::fill_n(input.begin(), kKvWidth, 0.0F);
    std::fill_n(input.begin() + kKvWidth, kKvWidth, -0.0F);
    for (unsigned d = 0; d < kKvWidth; ++d) {
      input[2 * kKvWidth + d] = d % 2 ? -1e-7F : 1e-7F;
      input[3 * kKvWidth + d] = std::numeric_limits<float>::min();
      input[4 * kKvWidth + d] =
          std::bit_cast<float>((d % 2 ? 0x80000000U : 0U) | (d % 17));
      input[5 * kKvWidth + d] = kind == kQK ? 4000.0F : 8000.0F;
    }
    std::fill_n(input.begin() + 6 * kKvWidth, kKvWidth, 0.0F);
    input[6 * kKvWidth + 31] = 1;
    input[6 * kKvWidth + 64] = -1;
    input[6 * kKvWidth + kDim + 255] = 2;
    auto rotated = input;
    CpuRotate(rotated, kind);
    const auto bytes = Encode(rotated);
    Buffer<float> src(input.size());
    Buffer<std::byte> dst((start + n + 2) * kRow);
    Buffer<std::uint32_t> pos(1);
    std::vector<std::byte> expected(dst.bytes(), std::byte{0xA5});
    std::copy(bytes.begin(), bytes.end(), expected.begin() + start * kRow);
    for (auto chunk : {1U, 3U, 8U}) {
      src.Put(input);
      Hip(hipMemset(dst.get(), 0xA5, dst.bytes()));
      for (unsigned off = 0; off < n; off += chunk) {
        const auto rows = std::min(chunk, n - off);
        pos.Put({start + off});
        Require(q::RotateKvRows(src.get() + std::size_t(off) * kKvWidth, rows,
                                kKvHeads, kDim, kind, nullptr),
                "rotated Store fixture refused");
        q::StoreKv(src.get() + std::size_t(off) * kKvWidth, dst.get(), rows,
                   kKvWidth, pos.get(), nullptr, kRotated);
      }
      Exact(expected, dst.Host(),
            "rotated Store canonical bytes/chunks/bounds");
      ValueExact(rotated, src.Host(),
                 "Store implicitly changed/rotated source");
    }
  }
}

void CheckPoison() {
  constexpr unsigned start = 2, n = 2;
  for (auto kind : {kQK, kVO}) {
    // Coherent finite values are VALID plain q8, yet rotation amplifies the
    // first coordinate above the finite-F16 reconstruction range (D7).
    const float coherent = kind == kQK ? 5000.0F : 10000.0F;
    for (float bad : {coherent, std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
      auto input = Values(n * kKvWidth, 991);
      if (std::isfinite(bad))
        std::fill_n(input.begin(), kDim, bad);
      else
        input[0] = bad;
      const auto companion =
          std::vector<float>(input.begin() + kKvWidth, input.end());
      auto expected_companion = companion;
      CpuRotate(expected_companion, kind);
      if (bad == coherent) {
        (void)Encode(input, kQ8);  // explicitly accepted BEFORE rotation
        auto rotated = input;
        CpuRotate(rotated, kind);
        Reject([&] { (void)Encode(rotated); });
      } else {
        auto head = std::vector<float>(input.begin(), input.begin() + kDim);
        Reject([&] { qfn::RotateKvHead(head, kind); });
      }
      Buffer<float> src(input.size());
      Buffer<std::byte> cache((start + n + 2) * kRow);
      Buffer<std::uint32_t> pos(1);
      src.Put(input);
      pos.Put({start});
      Hip(hipMemset(cache.get(), 0xA5, cache.bytes()));
      // Numeric invalidity is asynchronous payload signaling, NOT an expected
      // synchronous GPU exception or a false geometry return.
      Require(q::RotateKvRows(src.get(), n, kKvHeads, kDim, kind, nullptr),
              "numeric invalidity unexpectedly treated as geometry refusal");
      q::StoreKv(src.get(), cache.get(), n, kKvWidth, pos.get(), nullptr,
                 kRotated);
      const auto raw = cache.Host();
      const auto bad_row = std::span(raw).subspan(start * kRow, kRow);
      PoisonBlock(bad_row.first(34));
      Reject([&] { qfn::ValidateKvRow(bad_row, kKvWidth, kRotated); });
      std::vector<std::byte> expected(raw.size(), std::byte{0xA5});
      // Never Decode or attend to the poisoned row. Preserve it for the
      // companion/boundary comparison; the first poisoned block was checked.
      std::copy(bad_row.begin(), bad_row.end(),
                expected.begin() + start * kRow);
      auto good_head =
          std::vector<float>(input.begin() + kDim, input.begin() + kKvWidth);
      CpuRotate(good_head, kind);
      std::vector<std::byte> good_bytes(kRow / 2);
      qfn::EncodeKvRow(good_head, kRotated, good_bytes);
      std::copy(good_bytes.begin(), good_bytes.end(),
                expected.begin() + start * kRow + kRow / 2);
      const auto bytes = Encode(expected_companion);
      std::copy(bytes.begin(), bytes.end(),
                expected.begin() + (start + 1) * kRow);
      Exact(expected, raw, "poison companion/bounds");
    }
  }
  // Store is codec-only even for R: raw nonfinite/half-overflow inputs must
  // retain the qualified v2 poison behavior, without an implicit rotation.
  auto input = Values(kKvWidth, 44);
  for (unsigned b = 0; b < 4; ++b)
    input[b * 32] = std::array<float, 4>{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), 70000.0F, -70000.0F}[b];
  Reject([&] { (void)Encode(input); });
  Buffer<float> src(input.size());
  Buffer<std::byte> dst(kRow);
  Buffer<std::uint32_t> pos(1);
  src.Put(input);
  pos.Put({0});
  q::StoreKv(src.get(), dst.get(), 1, kKvWidth, pos.get(), nullptr, kRotated);
  const auto bytes = dst.Host();
  for (unsigned b = 0; b < 4; ++b)
    PoisonBlock(std::span(bytes).subspan(b * 34, 34));
}

void CheckPreparation(unsigned n, unsigned heads, const qfn::YarnRope& yarn) {
  constexpr unsigned start = 5;
  const unsigned width = heads * kDim;
  const unsigned stride = 2 * (width + kKvWidth) + 17;  // nontrivial row stride
  const std::size_t count = std::size_t(n) * width;
  const std::size_t cache_bytes = (start + n + 2) * kRow;
  Buffer<float> packed(std::size_t(n) * stride), gamma(2 * kDim);
  Buffer<std::uint32_t> pos(1);
  const auto input = Values(std::size_t(n) * stride, 17 + n + heads, 4);
  packed.Put(input);
  auto g = Values(2 * kDim, 37, 0.5F);
  for (auto& x : g)
    x += 1;
  gamma.Put(g);
  pos.Put({start});
  Buffer<float> qr(std::max<std::size_t>(count, 1)),
      gr(std::max<std::size_t>(count, 1));
  Buffer<float> kr(std::size_t(n) * kKvWidth), vr(std::size_t(n) * kKvWidth);
  // PREQUANT boundary: existing GPU Unpack/RMS/RoPE defines the identical
  // canonical F32 input. No F16/q8 prequantize -> rotate shortcut, and no
  // invented CPU projection/RMS/RoPE oracle.
  q::UnpackQGate(packed.get(), stride, heads ? qr.get() : nullptr,
                 heads ? gr.get() : nullptr, kr.get(), vr.get(), n, heads, kDim,
                 kKvWidth, nullptr);
  if (heads) {
    q::RmsNormRows(qr.get(), gamma.get(), qr.get(), n * heads, kDim, 1, 1e-6F,
                   nullptr);
    q::Rope(qr.get(), n, heads, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
            yarn);
  }
  q::RmsNormRows(kr.get(), gamma.get() + kDim, kr.get(), n * kKvHeads, kDim, 1,
                 1e-6F, nullptr);
  q::Rope(kr.get(), n, kKvHeads, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
          yarn);
  auto reference_q = heads ? qr.Host() : std::vector<float>{};
  const auto reference_g = heads ? gr.Host() : std::vector<float>{};
  auto reference_k = kr.Host(), reference_v = vr.Host();
  CpuRotate(reference_q, kQK);
  CpuRotate(reference_k, kQK);
  CpuRotate(reference_v, kVO);
  const auto kb = Encode(reference_k), vb = Encode(reference_v);
  std::vector<std::byte> ek(cache_bytes, std::byte{0xA5}), ev = ek;
  std::copy(kb.begin(), kb.end(), ek.begin() + start * kRow);
  std::copy(vb.begin(), vb.end(), ev.begin() + start * kRow);
  Buffer<float> qo(count + 2 * kGuard), go(count + 2 * kGuard);
  Buffer<std::byte> ko(cache_bytes), vo(cache_bytes);
  const auto reset = [&] {
    qo.Put(std::vector<float>(count + 2 * kGuard, kSentinel));
    go.Put(std::vector<float>(count + 2 * kGuard, kSentinel));
    Hip(hipMemset(ko.get(), 0xA5, ko.bytes()));
    Hip(hipMemset(vo.get(), 0xA5, vo.bytes()));
  };
  const auto check = [&] {
    const auto queries = qo.Host(), gates = go.Host();
    Guards(queries);
    Guards(gates);
    ValueExact(Guarded(reference_q), queries, "Prepare rotated Q");
    Exact(Guarded(reference_g), gates, "Prepare unchanged gates");
    Exact(ek, ko.Host(), "Prepare rotated K bytes/bounds");
    Exact(ev, vo.Host(), "Prepare rotated V bytes/bounds");
    Exact(input, packed.Host(), "Prepare input/stride padding changed");
  };
  for (bool prefill : {false, true}) {
    reset();
    Require(q::PrepareAttention(
                packed.get(), stride, heads ? gamma.get() : nullptr,
                gamma.get() + kDim, heads ? qo.get() + kGuard : nullptr,
                heads ? go.get() + kGuard : nullptr, ko.get(), vo.get(), n,
                heads, kKvHeads, kDim, 64, pos.get(), 1e7F, 1e-6F, nullptr,
                nullptr, prefill, yarn, kRotated),
            "rotated Prepare required, including cache-only heads=0");
    check();
  }
  // Same explicit YaRN parameters, including heads=0. This is NOT the known
  // batched-MTP-versus-sequential catch-up YaRN oracle (that bug is out of
  // scope).
  for (auto chunk : {1U, 3U, 8U}) {
    if (n <= 8)
      continue;
    reset();
    for (unsigned off = 0; off < n; off += chunk) {
      pos.Put({start + off});
      Require(
          q::PrepareAttention(
              packed.get() + std::size_t(off) * stride, stride,
              heads ? gamma.get() : nullptr, gamma.get() + kDim,
              heads ? qo.get() + kGuard + std::size_t(off) * width : nullptr,
              heads ? go.get() + kGuard + std::size_t(off) * width : nullptr,
              ko.get(), vo.get(), std::min(chunk, n - off), heads, kKvHeads,
              kDim, 64, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, false, yarn,
              kRotated),
          "chunked rotated Prepare refused");
    }
    check();
  }
  // Separate rotate + codec-only Store must produce the SAME bytes from the
  // same canonical F32 boundary. It must not rotate again inside StoreKv.
  pos.Put({start});
  reset();
  Require(q::RotateKvRows(kr.get(), n, kKvHeads, kDim, kQK, nullptr) &&
              q::RotateKvRows(vr.get(), n, kKvHeads, kDim, kVO, nullptr),
          "separate producer rotation refused");
  q::StoreKv(kr.get(), ko.get(), n, kKvWidth, pos.get(), nullptr, kRotated);
  q::StoreKv(vr.get(), vo.get(), n, kKvWidth, pos.get(), nullptr, kRotated);
  Exact(ek, ko.Host(), "separate producer K bytes/bounds");
  Exact(ev, vo.Host(), "separate producer V bytes/bounds");
}

// Attention tests deliberately stop at the pre-output-projection boundary.
// Direct q8 and its EXACT decoded-half operands are compared within each
// algorithm/split, never scalar versus WMMA or F16-producer versus q8.
void CheckAttention(unsigned start, unsigned n, bool sparse, bool rotated) {
  const auto context = start + n;
  const auto words = ((context + kRatio - 1) / kRatio + 31) / 32;
  const std::size_t count = std::size_t(n) * kQWidth;
  auto k = Values(std::size_t(context) * kKvWidth, 891);
  auto v = Values(std::size_t(context) * kKvWidth, 347);
  auto queries = Values(count, 412, 4);
  if (rotated) {
    CpuRotate(k, kQK);
    CpuRotate(v, kVO);
    CpuRotate(queries, kQK);
  }
  const auto type = rotated ? kRotated : kQ8;
  const auto kb = Encode(k, type), vb = Encode(v, type);
  Buffer<std::byte> keys(kb.size()), values(vb.size());
  Buffer<__half> kh(k.size()), vh(v.size());
  Buffer<float> query(count), gate(count), out(count + 2 * kGuard);
  Buffer<float> partials(std::size_t(n) * kHeads * 8 * (kDim + 2));
  Buffer<std::uint32_t> masks(std::size_t(n) * words), pos(1);
  // Q8_1 is two F16 fields + 32 signed codes, 36 bytes per 32 channels.
  Buffer<std::byte> activation(count / 32 * 36);
  keys.Put(kb);
  values.Put(vb);
  kh.Put(Decode(kb, type));
  vh.Put(Decode(vb, type));
  query.Put(queries);
  // Nonuniform, sign-changing gates expose the inverse/gate ordering hazard.
  const auto gates = Values(count, 721, 6);
  gate.Put(gates);
  pos.Put({start});
  std::vector<std::uint32_t> mask(std::size_t(n) * words);
  for (unsigned t = 0; t < n; ++t)
    for (unsigned b = 0; b < (start + t + 1) / kRatio; ++b)
      if ((b + t) % 5 != 0)
        mask[std::size_t(t) * words + b / 32] |= 1U << (b % 32);
  masks.Put(mask);
  const auto selection = sparse ? masks.get() : nullptr;
  const auto reset = [&] {
    out.Put(std::vector<float>(count + 2 * kGuard, kSentinel));
    Hip(hipMemset(activation.get(), 0xA5, activation.bytes()));
  };
  const auto scalar = [&](bool decoded, unsigned splits) {
    reset();
    q::Attention(
        query.get(), decoded ? static_cast<void*>(kh.get()) : keys.get(),
        decoded ? static_cast<void*>(vh.get()) : values.get(), selection, words,
        out.get() + kGuard, splits ? partials.get() : nullptr, splits, n,
        pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr,
        decoded ? kF16 : type);
    auto raw = out.Host();
    Guards(raw);
    return raw;
  };
  // Invert only initialized rows, AFTER Attention's merge or WMMA sublaunches,
  // and BEFORE either existing gate kernel. Also compare GPU inverse against
  // CPU FWHT on the actual raw context, not just two uses of the GPU helper.
  const auto finish = [&](const std::vector<float>& raw, unsigned first,
                          bool quantize, bool apply_gate = true) {
    out.Put(raw);
    auto expected = raw;
    if (rotated) {
      CpuRotate(
          std::span(expected).subspan(kGuard + std::size_t(first) * kQWidth,
                                      std::size_t(n - first) * kQWidth),
          kVO);
      Require(q::RotateKvRows(out.get() + kGuard + std::size_t(first) * kQWidth,
                              n - first, kHeads, kDim, kVO, nullptr),
              "context inverse refused");
    }
    ValueExact(expected, out.Host(), "pre-gate canonical inverse");
    if (apply_gate) {
      const auto offset = std::size_t(first) * kQWidth;
      if (quantize) {
        Require(q::SigmoidMulQ8_1(out.get() + kGuard + offset,
                                  gate.get() + offset, count - offset,
                                  activation.get() + offset / 32 * 36, nullptr),
                "Q8_1 gate refused");
      } else {
        q::SigmoidMul(out.get() + kGuard + offset, gate.get() + offset,
                      count - offset, nullptr);
      }
    }
    auto result = out.Host();
    Guards(result);
    Exact(std::vector<float>(
              raw.begin(), raw.begin() + kGuard + std::size_t(first) * kQWidth),
          std::vector<float>(result.begin(), result.begin() + kGuard +
                                                 std::size_t(first) * kQWidth),
          "inverse/gate touched unset rows");
    return result;
  };
  for (auto splits : {0U, 1U, 3U, 8U}) {
    const auto reference = scalar(true, splits);
    ValueExact(reference, scalar(false, splits), "narrow decoded-half control");
    const auto direct = scalar(false, splits);
    for (bool quantize : {false, true}) {
      const auto expected = finish(reference, 0, quantize);
      const auto expected_q8 = activation.Host();
      ValueExact(expected, finish(direct, 0, quantize),
                 "narrow inverse then gate");
      if (quantize)
        Exact(expected_q8, activation.Host(), "narrow inverse/gate Q8_1 bytes");
    }
  }
  const auto wmma = [&](bool decoded, bool last, const float* fused_gate) {
    reset();
    Require(q::WmmaCausalAttention(
                query.get(), fused_gate,
                decoded ? static_cast<void*>(kh.get()) : keys.get(),
                decoded ? static_cast<void*>(vh.get()) : values.get(),
                selection, words, out.get() + kGuard, n, start, kHeads,
                kKvHeads, kDim, kRatio, nullptr, last, decoded ? kF16 : type),
            "WMMA refused supported geometry/null gate");
    auto raw = out.Host();
    Guards(raw);
    return raw;
  };
  const auto full_wmma = wmma(false, false, nullptr);
  for (bool last : {false, true}) {
    // Last-only retains the final query tile: dense16/sparse4.
    const unsigned first = !last    ? 0
                           : sparse ? (n - 1) / 4 * 4
                                    : (n - 1) / 16 * 16;
    const auto reference = wmma(true, last, nullptr);
    const auto direct = wmma(false, last, nullptr);
    auto last_expected = full_wmma;
    std::fill(last_expected.begin() + kGuard,
              last_expected.begin() + kGuard + std::size_t(first) * kQWidth,
              kSentinel);
    ValueExact(last_expected, direct,
               "WMMA last-only active rows/unset sentinels");
    ValueExact(reference, direct, "WMMA null-gate decoded-half control");
    ValueExact(direct, wmma(false, last, nullptr), "WMMA null-gate repeat");
    const auto ungated = finish(reference, first, false, false);
    ValueExact(ungated, finish(direct, first, false, false),
               "WMMA inverse no gate");
    for (bool quantize : {false, true}) {
      const auto expected = finish(reference, first, quantize);
      const auto expected_q8 = activation.Host();
      ValueExact(expected, finish(direct, first, quantize),
                 "WMMA inverse then gate");
      if (quantize)
        Exact(expected_q8, activation.Host(), "WMMA inverse/gate Q8_1 bytes");
    }
    if (rotated && !last) {
      const auto correct = finish(direct, 0, false);
      // The old WMMA fused gate is the WRONG place for rotated coordinates.
      // Show this with real nonuniform gates, rather than merely commenting it.
      const auto wrong_raw = wmma(false, false, gate.get());
      const auto wrong = finish(wrong_raw, 0, false, false);
      bool different = false;
      for (std::size_t i = kGuard; i < kGuard + count; ++i)
        different |= correct[i] != wrong[i];
      Require(different,
              "gate-hazard fixture failed to distinguish wrong order");
    }
  }
  // Operator analogue of E1: both WMMA sublaunches finish BEFORE the one
  // full-row inverse/gate. This checks the range primitive, not executor
  // checkpoint planning, persistence, projection or early-return placement.
  for (auto cut : {1U, 3U, 7U, 8U, 16U}) {
    if (cut >= n)
      continue;
    std::vector<float> packed_joined, packed_finished;
    for (bool decoded : {false, true}) {
      reset();
      for (auto range : {std::array<unsigned, 2>{0, cut}, {cut, n - cut}}) {
        const auto first = range[0], rows = range[1];
        const auto offset = std::size_t(first) * kQWidth;
        Require(
            q::WmmaCausalAttention(
                query.get() + offset, nullptr,
                decoded ? static_cast<void*>(kh.get()) : keys.get(),
                decoded ? static_cast<void*>(vh.get()) : values.get(),
                sparse ? masks.get() + std::size_t(first) * words : nullptr,
                words, out.get() + kGuard + offset, rows, start + first, kHeads,
                kKvHeads, kDim, kRatio, nullptr, false, decoded ? kF16 : type),
            "WMMA checkpoint-range primitive refused");
      }
      const auto joined = out.Host();
      const auto finished = finish(joined, 0, false);
      if (!decoded) {
        packed_joined = joined;
        packed_finished = finished;
      } else {
        // Same four-query grouping in both representations. A
        // different split can legitimately change key-union tile rounding.
        ValueExact(packed_joined, joined, "WMMA split decoded-half control");
        ValueExact(packed_finished, finished,
                   "WMMA inverse/gate after both sublaunches");
      }
    }
  }
  // Native F16 and rotated-F16 producer controls: repeatability only. Their
  // operands differ from q8-decoded operands, so NO q8/F16 equality or quality
  // threshold is asserted. These five arms are not a model-quality campaign.
  kh.Put(Half(k));
  vh.Put(Half(v));
  const auto native = scalar(true, 0);
  ValueExact(native, scalar(true, 0), "F16-producer repeat");
  const auto native_finished = finish(native, 0, false);
  ValueExact(native_finished, finish(scalar(true, 0), 0, false),
             "F16-producer inverse/gate repeat");
  Exact(kb, keys.Host(), "attention mutated packed K");
  Exact(vb, values.Host(), "attention mutated packed V");
  Exact(queries, query.Host(), "attention mutated Q");
  Exact(gates, gate.Host(), "attention mutated gate");
  Exact(mask, masks.Host(), "attention mutated selection mask");
}

void CheckRefusals(const qfn::YarnRope& yarn) {
  Buffer<float> data(2 * kGuard);
  Buffer<std::byte> cache(2 * kRow);
  Buffer<std::uint32_t> pos(1);
  const std::vector<float> sentinel(2 * kGuard, kSentinel);
  const std::vector<std::byte> bytes(2 * kRow, std::byte{0xA5});
  data.Put(sentinel);
  cache.Put(bytes);
  pos.Put({0});
  const auto prepare = [&](hipStream_t stream, Type type, unsigned heads,
                           unsigned kv_heads, unsigned d, unsigned rotary,
                           unsigned stride) {
    return q::PrepareAttention(data.get(), stride, data.get(), data.get(),
                               data.get(), data.get(), cache.get(), cache.get(),
                               1, heads, kv_heads, d, rotary, pos.get(), 1e7F,
                               1e-6F, stream, nullptr, false, yarn, type);
  };
  const auto fused = [&](hipStream_t stream, Type type, unsigned n) {
    // All pointers are nonnull, so refusal cannot be satisfied by a null-input
    // guard. These undersized buffers are legal ONLY inside NoLaunch capture;
    // a nonempty graph fails and is destroyed without replay or device OOB.
    return q::AttentionF16Gemm(
        cache.get(), reinterpret_cast<const __half*>(cache.get()), data.get(),
        data.get(), data.get(), data.get(), cache.get(), cache.get(), n,
        pos.get(), 1e7F, 1e-6F, stream, nullptr, yarn, type);
  };
  for (auto unknown : {3U, 42U, std::numeric_limits<unsigned>::max()}) {
    const auto type = static_cast<Type>(unknown);
    NoLaunch check;
    Require(!prepare(check.stream(), type, kHeads, kKvHeads, kDim, 64, 13312),
            "unknown type accepted by Prepare");
    Require(!fused(check.stream(), type, 1024), "unknown fused type");
    Reject([&] {
      q::StoreKv(data.get(), cache.get(), 1, kKvWidth, pos.get(),
                 check.stream(), type);
    });
    Reject([&] {
      q::Attention(data.get(), cache.get(), cache.get(), pos.get(), 1,
                   data.get(), nullptr, 0, 1, pos.get(), kHeads, kKvHeads, kDim,
                   kRatio, check.stream(), type);
    });
    Require(
        !q::WmmaCausalAttention(data.get(), nullptr, cache.get(), cache.get(),
                                nullptr, 0, data.get(), 1, 0, kHeads, kKvHeads,
                                kDim, kRatio, check.stream(), false, type),
        "unknown WMMA type");
    check.Check();
  }
  for (unsigned n : {1U, 1024U}) {
    NoLaunch check;
    // Rotated fused projection is deliberately unsupported, including an
    // otherwise eligible wide row count. Executor must Dense + rotated Prepare.
    Require(!fused(check.stream(), kRotated, n), "rotated fused launch");
    check.Check();
  }
  for (auto geometry : {std::array<unsigned, 5>{23, 2, 256, 64, 13312},
                        {24, 1, 256, 64, 13312},
                        {24, 2, 128, 64, 13312},
                        {24, 2, 256, 32, 13312},
                        {24, 2, 256, 64, 13311},
                        {0, 2, 256, 64, 1023}}) {
    NoLaunch check;
    Require(!prepare(check.stream(), kRotated, geometry[0], geometry[1],
                     geometry[2], geometry[3], geometry[4]),
            "bad rotated Prepare geometry");
    check.Check();
  }
  for (auto type : {kF16, kQ8, kRotated}) {
    NoLaunch check;
    Require(!prepare(check.stream(), type, kHeads, kKvHeads, 257, 64, 20000),
            "Prepare accepted oversized head");
    // Legacy F16/plain wrappers accept nonzero row counts (the executor,
    // not this wrapper, chooses the wide route). Only zero is refused here.
    Require(!fused(check.stream(), type, 0),
            "fused projection accepted empty input");
    if (type != kF16)
      Reject([&] {
        q::StoreKv(data.get(), cache.get(), 1, 31, pos.get(), check.stream(),
                   type);
      });
    // Store is intentionally a codec-only API: 32/64/128-wide rows ARE valid
    // for R. D256 admission belongs to the model/transform, not storage.
    if (type == kRotated) {
      for (auto geometry :
           {std::array<unsigned, 3>{23, 2, 256}, {24, 1, 256}, {24, 2, 128}})
        Reject([&] {
          q::Attention(data.get(), cache.get(), cache.get(), pos.get(), 1,
                       data.get(), nullptr, 0, 1, pos.get(), geometry[0],
                       geometry[1], geometry[2], kRatio, check.stream(), type);
        });
    } else {
      q::Attention(data.get(), cache.get(), cache.get(), nullptr, 0, data.get(),
                   nullptr, 0, 1, pos.get(), kHeads, kKvHeads, 128, kRatio,
                   check.stream(), type);
    }
    for (auto geometry : {std::array<unsigned, 4>{23, 2, 256, 4},
                          {24, 1, 256, 4},
                          {24, 2, 128, 4},
                          {24, 2, 256, 1}}) {
      Require(!q::WmmaCausalAttention(data.get(), nullptr, cache.get(),
                                      cache.get(), nullptr, 0, data.get(), 1, 0,
                                      geometry[0], geometry[1], geometry[2],
                                      geometry[3], check.stream(), false, type),
              "bad WMMA geometry");
      Require(q::WmmaCausalAttentionContextLimit(geometry[0], geometry[1],
                                                 geometry[2], geometry[3]) == 0,
              "unsupported model geometry has WMMA context limit");
    }
    // Cheap wrapper-refusal coverage only. The actual E3 executor fallback
    // (>262144 sparse context -> narrow unsplit -> inverse -> gate) remains
    // a required INTEGRATION case, not an operator pass or silently skipped
    // gate.
    Require(!q::WmmaCausalAttention(data.get(), nullptr, cache.get(),
                                    cache.get(), pos.get(), 2049, data.get(), 1,
                                    262144, kHeads, kKvHeads, kDim, kRatio,
                                    check.stream(), false, type),
            "WMMA over-limit sparse launch");
    check.Check();
  }
  {
    NoLaunch check;
    for (auto d : {0U, 64U, 128U, 255U, 257U})
      Require(!q::RotateKvRows(data.get(), 1, 2, d, kQK, check.stream()),
              "rotation accepted wrong D");
    for (auto heads : {0U, 25U})
      Require(!q::RotateKvRows(data.get(), 1, heads, kDim, kVO, check.stream()),
              "rotation accepted wrong heads");
    Require(!q::RotateKvRows(nullptr, 1, 2, kDim, kQK, check.stream()),
            "rotation accepted null nonempty input");
    Require(!q::RotateKvRows(data.get(), 1, 2, kDim, static_cast<Kind>(99),
                             check.stream()),
            "rotation accepted unknown kind");
    Require(!q::RotateKvRows(data.get(), std::numeric_limits<unsigned>::max(),
                             kHeads, kDim, kVO, check.stream()),
            "rotation grid overflow");
    Require(q::RotateKvRows(nullptr, 0, 2, kDim, kVO, check.stream()),
            "rotation refused valid empty input");
    check.Check();
  }
  for (unsigned missing = 0; missing < 6; ++missing) {
    NoLaunch check;
    Require(!q::PrepareAttention(
                missing == 0 ? nullptr : data.get(), 13312, nullptr, nullptr,
                missing == 1 ? nullptr : data.get(),
                missing == 2 ? nullptr : data.get(),
                missing == 3 ? nullptr : cache.get(),
                missing == 4 ? nullptr : cache.get(), 1, kHeads, kKvHeads, kDim,
                64, missing == 5 ? nullptr : pos.get(), 1e7F, 1e-6F,
                check.stream(), nullptr, false, yarn, kRotated),
            "Prepare accepted missing required pointer");
    check.Check();
  }
  Exact(sentinel, data.Host(), "refused wrapper touched F32 sentinel");
  Exact(bytes, cache.Host(), "refused wrapper touched packed sentinel");
}
}  // namespace

void CheckKvCacheOnlyProducers(const char* model_path, const char* mtp_path);

int main(int argc, char** argv) {
  try {
    // Explicit, coordinator-owned real-model mode. No default model discovery,
    // download, native CTest registration or missing-model success.
    if (argc != 1) {
      if (argc != 6 || std::string_view(argv[1]) != "--model" ||
          std::string_view(argv[3]) != "--mtp-model" ||
          std::string_view(argv[5]) != "--producer-only") {
        std::cerr << "Usage: kv_rotation_ops_test [--model FIRST.gguf "
                     "--mtp-model MTP-Q8.gguf --producer-only]\n";
        return 77;
      }
      CheckKvCacheOnlyProducers(argv[2], argv[4]);
      return 0;
    }
    CheckTransforms();
    std::cout << "ROTATED_KV_V3_OPS: FWHT CPU/GPU nonzero-bit exact (+/-0 "
                 "equivalent)\n";
    CheckRotatedStore();
    CheckPoison();
    const auto yarn = qfn::MakeYarnRope(
        {.factor = 1.5625F, .original_context = 262144}, 64, 1e7F);
    for (const auto& mode : {qfn::YarnRope{}, yarn}) {
      CheckRefusals(mode);
      for (auto heads : {0U, 24U})
        for (auto n : {1U, 3U, 8U, 32U, 33U, 64U})
          CheckPreparation(n, heads, mode);
    }
    std::cout
        << "ROTATED_KV_V3_OPS: prequant Prepare/separate bytes, heads24/0, "
           "YaRN off/on, poison, no-launch refusals exact\n";
    for (auto n : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 17U, 33U}) {
      CheckAttention(37, n, false, true);
      CheckAttention(2412, n, true, true);
    }
    for (auto start : {2040U, 2044U, 2048U})
      CheckAttention(start, 8, start >= 2048, true);
    CheckAttention(2044, 8, true,
                   true);  // sparse companion at the 2048 boundary
    CheckAttention(37, 8, false, false);
    CheckAttention(2412, 8, true, false);
    Require(live_bytes == 0, "device scratch leak");
    std::cout
        << "ROTATED_KV_V3_OPS: narrow widths1..8/splits/masks; WMMA "
           "dense/sparse/null-gate/last-only; inverse-before-gate/Q8_1 exact\n"
        << "ROTATED_KV_V3_OPS: controls=plain-F16,rotated-F16,plain-q8-v2,"
           "rotated-q8,rotated-q8-decoded-F16; no F16-vs-q8 quality verdict\n"
        << "ROTATED_KV_V3_OPS: explicit device scratch peak=" << peak_bytes
        << '\n'
        << "ROTATED_KV_V3_OPS: OPERATOR_CHECKS_COMPLETE; integration NOT "
           "covered: "
           "E1 checkpoint executor, E3 >262144 fallback, constructor "
           "admission, "
           "indexer state, trunk/MTP/rollback/persistence, D6 legacy "
           "ISA/forward, "
           "full-model quality. Legacy kv_q8_ops_test must run unchanged.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ROTATED_KV_V3_OPS: FAIL: " << error.what() << '\n';
    return 1;
  }
}
