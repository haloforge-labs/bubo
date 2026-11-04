// Operator-only raw PLE comparison and cold-weight timing.
#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen38_flash_next/ple_test_data.hpp"

namespace q = gufo::models::qwen38_flash_next;
using q::test::Require;
hipError_t ExpertBenchEvict(void* buffer, std::size_t bytes);

namespace {
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
struct Buffers {
  std::uint8_t* host{nullptr};
  std::uint8_t* raw{nullptr};
  float* out{nullptr};
  void* evict{nullptr};
  hipEvent_t begin{nullptr}, end{nullptr};
  ~Buffers() {
    (void)hipDeviceSynchronize();
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    (void)hipFree(out);
    (void)hipFree(evict);
    (void)hipHostFree(host);
  }
};

void CompareRows(q::test::RealPleTable& real, Buffers& b) {
  const auto& config = real.config;
  auto& table = *real.table;
  std::mt19937 random(711);
  std::array<std::int32_t, 9> tokens{};
  for (auto& token : tokens)
    token = static_cast<std::int32_t>(random() % real.vocab);
  for (const bool prompt : {false, true}) {
    for (const std::size_t count : {1U, 2U, 8U, 9U}) {
      const auto chain = prompt ? std::span(real.prompt).first(count)
                                : std::span(tokens).first(count);
      q::NgramHistory before;
      q::PleReadAhead ahead;
      ahead.Begin(before);
      for (const auto token : chain)
        Require(ahead.Append(table, config, token, {}),
                "raw submission failed");
      const auto rows = count * config.ple_heads;
      Require(ahead.CopyRows(count, std::span(b.host, rows * table.RowBytes())),
              "raw gather failed");
      std::vector<std::uint32_t> ids(rows);
      q::HashNgramRows(config, before, chain, ids);
      std::vector<float> reference(rows * table.RowDim());
      Require(table.Read(ids, reference), "CPU reference read failed");
      Check(hipMemset(b.out, 0xA5, (reference.size() + 16) * sizeof(float)));
      q::rocm::DecodePleRows(
          b.raw, b.out, static_cast<std::uint32_t>(rows), table.RowDim(),
          table.Type() == gufo::core::GgmlType::kIQ4_NL, nullptr);
      Check(hipGetLastError());
      std::vector<float> result(reference.size() + 16);
      Check(hipMemcpy(result.data(), b.out, result.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      Require(std::memcmp(result.data(), reference.data(),
                          reference.size() * sizeof(float)) == 0,
              "GPU raw decode differs from CPU F32 bytes");
      for (std::size_t i = reference.size(); i < result.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &result[i], sizeof(bits));
        Require(bits == 0xA5A5A5A5U, "GPU decode wrote beyond the row count");
      }
    }
  }
  std::puts(
      "CPU/GPU bytes identical for random and real-prompt tokens; counts "
      "1,2,8,9");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--bench") {
    std::puts("usage: ple_rows_bench --bench first-production-shard.gguf");
    return 2;
  }
  try {
    auto real = q::test::OpenRealPle(std::filesystem::u8path(argv[2]));
    const auto& table = *real.table;
    constexpr std::size_t rows = 16384, calls = 32;
    constexpr std::size_t evict_bytes = 64U << 20;
    const auto slice_bytes = rows * table.RowBytes();
    const auto out_bytes = rows * table.RowDim() * sizeof(float);
    Buffers b;
    Check(hipHostMalloc(reinterpret_cast<void**>(&b.host), calls * slice_bytes,
                        hipHostMallocMapped | hipHostMallocCoherent));
    Check(hipHostGetDevicePointer(reinterpret_cast<void**>(&b.raw), b.host, 0));
    Check(hipMalloc(reinterpret_cast<void**>(&b.out), out_bytes));
    Check(hipMalloc(&b.evict, evict_bytes));
    Check(hipMemset(b.evict, 0, evict_bytes));
    Check(hipEventCreate(&b.begin));
    Check(hipEventCreate(&b.end));
    CompareRows(real, b);

    // Populate distinct mapped weight addresses with real compressed rows.
    // The cache sweep below also removes any lines touched by these copies.
    std::array<std::uint32_t, 512> ids{};
    std::mt19937 random(173);
    for (auto& id : ids)
      id = random() % table.Rows();
    auto source = real.table->ReadAhead(ids);
    Require(source && source->Wait(), "bench weight read failed");
    const auto raw = source->RawRows();
    for (std::size_t offset = 0; offset < calls * slice_bytes;) {
      const auto count = std::min(raw.size(), calls * slice_bytes - offset);
      std::memcpy(b.host + offset, raw.data(), count);
      offset += count;
    }
    const bool iq4 = table.Type() == gufo::core::GgmlType::kIQ4_NL;
    q::rocm::DecodePleRows(b.raw + (calls - 1) * slice_bytes, b.out, rows,
                           table.RowDim(), iq4, nullptr);
    Check(hipDeviceSynchronize());
    double total_ms = 0;
    unsigned kept = 0;
    for (std::size_t call = 0; call < calls; ++call) {
      Check(ExpertBenchEvict(b.evict, evict_bytes));
      Check(hipDeviceSynchronize());
      Check(hipEventRecord(b.begin));
      q::rocm::DecodePleRows(b.raw + call * slice_bytes, b.out, rows,
                             table.RowDim(), iq4, nullptr);
      Check(hipGetLastError());
      Check(hipEventRecord(b.end));
      Check(hipEventSynchronize(b.end));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, b.begin, b.end));
      const double gbps = (slice_bytes + out_bytes) / (ms * 1e6);
      if (!(ms > 0) || !(gbps <= 256)) {
        std::puts("Discarded a timed call above the 256 GB/s ceiling.");
        continue;
      }
      total_ms += ms;
      ++kept;
    }
    if (kept)
      std::printf(
          "raw_gpu_calls=%u rows_per_call=%zu mean_ms=%.4f "
          "read_write_GB/s=%.3f\n",
          kept, rows, total_ms / kept,
          kept * (slice_bytes + out_bytes) / (total_ms * 1e6));
    else
      std::puts("No retained timing.");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
  return 0;
}
