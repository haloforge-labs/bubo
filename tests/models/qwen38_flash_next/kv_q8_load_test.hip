#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kv_q8_load.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"

// Compile this TU as HIP/C++20 for gfx1151 with the production fast-math
// options. Link the portable kv_cache.cpp (or its existing owning library)
// and the HIP runtime; no production kernels/model/weights are needed.
namespace qfn = gufo::models::qwen38_flash_next;
namespace detail = gufo::models::qwen38_flash_next::rocm::detail;
namespace {
constexpr std::uint32_t kCodes = 255;
constexpr std::uint32_t kScales = 0x6008;
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kBatchScales = 256;
constexpr std::uint32_t kPositions = 64;
constexpr std::uint64_t kPairs = std::uint64_t{kCodes} * kScales;
static_assert(kPairs == 6268920);
static_assert(sizeof(qfn::KvBlockQ8_0) == 34);
static_assert(alignof(qfn::KvBlockQ8_0) == 2);

struct Result {
  std::uint64_t mismatch_mask;
  std::uint32_t first_actual;
};

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

template<typename T>
class Buffer {
public:
  explicit Buffer(std::size_t count) {
    void* p = nullptr;
    Hip(hipMalloc(&p, count * sizeof(T)));
    p_ = static_cast<T*>(p);
  }
  ~Buffer() { (void)hipFree(p_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() const { return p_; }

private:
  T* p_{};
};

__host__ __device__ std::uint16_t CanonicalZero(std::uint16_t bits) {
  return (bits & 0x7fffU) == 0 ? 0 : bits;
}

// Only 510 blocks, shared by every scale. For each fixed position, the 255
// records permute ALL codes: every multiplier is coprime to 255. Different
// multipliers/shifts vary the packed neighbors, including mixed signs/zeros.
// Two adjacent real blocks retain the 34-byte stride. Across records/groups,
// code pointers exercise every even alignment modulo 16. The last group's
// eighth byte is the allocation's last byte (no oversized tail fixture).
std::vector<qfn::KvBlockQ8_0> Fixtures() {
  constexpr std::array<unsigned, 8> kMultipliers = {1, 2, 4, 7, 8, 11, 13, 14};
  std::vector<qfn::KvBlockQ8_0> blocks(2 * kCodes);
  for (unsigned pos = 0; pos < kPositions; ++pos) {
    std::array<bool, kCodes> seen{};
    for (unsigned record = 0; record < kCodes; ++record) {
      const unsigned code =
          (record * kMultipliers[pos % 8] + 37 * pos + 11 * (pos / 8)) % kCodes;
      blocks[2 * record + pos / 32].values[pos % 32] =
          static_cast<std::int8_t>(static_cast<int>(code) - 127);
      seen[code] = true;
    }
    Require(std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }),
            "fixture does not exhaust codes at every position");
  }
  // The helper takes scale separately. These fixture headers are unused;
  // the actual original F16 scale bits are supplied by each kernel thread.
  return blocks;
}

// Independent CPU codec oracle, never a second GPU implementation. Decode
// performs exact F32 q*scale followed by portable integer RNE-to-half. The
// final conversion just recovers those already-rounded F16 bits. No host
// _Float16 arithmetic or GPU conversion/denormal behavior defines expected.
std::vector<std::uint16_t> Reference(std::uint64_t& subnormal_pairs) {
  std::array<qfn::KvBlockQ8_0, 8> row{};
  std::array<float, 256> decoded{};
  for (unsigned i = 0; i < kCodes; ++i)
    row[i / 32].values[i % 32] =
        static_cast<std::int8_t>(static_cast<int>(i) - 127);
  std::vector<std::uint16_t> reference(kPairs);
  for (unsigned scale = 0; scale < kScales; ++scale) {
    for (auto& block : row)
      block.scale = static_cast<std::uint16_t>(scale);
    qfn::DecodeKvRow(std::as_bytes(std::span(row)), qfn::KvCacheType::kQ8_0,
                     decoded);
    for (unsigned code = 0; code < kCodes; ++code) {
      const auto bits = CanonicalZero(qfn::KvFloatToHalfBits(decoded[code]));
      Require((bits & 0x7c00U) != 0x7c00U, "nonfinite CPU oracle result");
      reference[scale * kCodes + code] = bits;
      if ((bits & 0x7c00U) == 0 && (bits & 0x03ffU) != 0)
        ++subnormal_pairs;
    }
  }
  Require(reference[kCodes + 128] == 0x0001 &&
              reference[kCodes + 126] == 0x8001 && subnormal_pairs != 0,
          "CPU oracle lost smallest signed F16 subnormals");
  return reference;
}

__launch_bounds__(kThreads) __global__
    void CheckPacked(const qfn::KvBlockQ8_0* fixtures,
                     const std::uint16_t* reference, std::uint32_t first_scale,
                     std::uint32_t count, Result* results) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const std::uint32_t scale_bits = first_scale + index / kCodes;
  const std::uint32_t record = index % kCodes;
  const _Float16 scale =
      __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(scale_bits));
  Result result{};
#pragma unroll
  for (unsigned parity = 0; parity < 2; ++parity) {
    const auto& block = fixtures[2 * record + parity];
#pragma unroll
    for (unsigned group = 0; group < 4; ++group) {
      // This is the ACTUAL candidate header, not a copy of its arithmetic.
      const uint4 packed =
          detail::LoadQ8Kv8Packed(block.values + 8 * group, scale);
      const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (unsigned lane = 0; lane < 8; ++lane) {
        const unsigned pos = parity * 32 + group * 8 + lane;
        const int code = block.values[group * 8 + lane];
        const auto expected = reference[scale_bits * kCodes + code + 127];
        const auto actual = CanonicalZero(
            static_cast<std::uint16_t>(words[lane / 2] >> (16 * (lane % 2))));
        if (actual != expected) {
          if (result.mismatch_mask == 0)
            result.first_actual = actual;
          result.mismatch_mask |= std::uint64_t{1} << pos;
        }
      }
    }
  }
  results[index] = result;
}
}  // namespace

int main() {
  try {
    std::uint64_t subnormal_pairs = 0;
    const auto reference = Reference(subnormal_pairs);
    const auto fixtures = Fixtures();
    Buffer<std::uint16_t> d_reference(reference.size());
    Buffer<qfn::KvBlockQ8_0> d_fixtures(fixtures.size());
    Buffer<Result> d_results(kBatchScales * kCodes);
    std::vector<Result> results(kBatchScales * kCodes);
    Hip(hipMemcpy(d_reference.get(), reference.data(),
                  reference.size() * sizeof(reference[0]),
                  hipMemcpyHostToDevice));
    Hip(hipMemcpy(d_fixtures.get(), fixtures.data(),
                  fixtures.size() * sizeof(fixtures[0]),
                  hipMemcpyHostToDevice));

    // CPU+GPU tables/fixtures/batched results total <27 MiB (excluding HIP
    // runtime overhead); no 401-million-element output allocation or atomics.
    std::uint64_t mismatches = 0;
    unsigned reported = 0;
    for (unsigned first = 0; first < kScales; first += kBatchScales) {
      const unsigned count = std::min(kBatchScales, kScales - first) * kCodes;
      hipLaunchKernelGGL(CheckPacked, dim3((count + kThreads - 1) / kThreads),
                         dim3(kThreads), 0, 0, d_fixtures.get(),
                         d_reference.get(), first, count, d_results.get());
      Hip(hipGetLastError());
      Hip(hipDeviceSynchronize());
      Hip(hipMemcpy(results.data(), d_results.get(), count * sizeof(Result),
                    hipMemcpyDeviceToHost));
      for (unsigned i = 0; i < count; ++i) {
        const auto mask = results[i].mismatch_mask;
        mismatches += std::popcount(mask);
        if (mask == 0 || reported >= 8)
          continue;
        ++reported;
        const unsigned scale = first + i / kCodes;
        const unsigned record = i % kCodes;
        const unsigned pos = std::countr_zero(mask);
        const int code = fixtures[2 * record + pos / 32].values[pos % 32];
        std::cerr << "mismatch scale=0x" << std::hex << scale << " expected=0x"
                  << reference[scale * kCodes + code + 127] << " actual=0x"
                  << results[i].first_actual << std::dec << " q=" << code
                  << " parity=" << pos / 32 << " offset=" << (pos % 32) / 8 * 8
                  << " lane=" << pos % 8 << '\n';
      }
    }
    std::cout << (mismatches == 0 ? "PASS" : "FAIL")
              << " kv_q8_load: " << kPairs << " scale/code pairs, "
              << kPairs * kPositions
              << " comparisons (all 8 lanes x 4 offsets x 2 block parities), "
              << subnormal_pairs << " nonzero-subnormal pairs, " << mismatches
              << " mismatches; only signed zeros canonicalized\n";
    return mismatches == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL kv_q8_load: " << error.what() << '\n';
    return 1;
  }
}
