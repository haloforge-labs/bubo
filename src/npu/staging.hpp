// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_STAGING_HPP_
#define GUFO_NPU_STAGING_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "guard.hpp"
#include "session.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::npu {
struct Knobs {
  std::filesystem::path artifacts;
  unsigned slices = 6, gate_slices = 0;
  bool shared_expert = true;
  bool hal_monitor = true;  // GUFO_NPU_HAL_MONITOR, detection only
  bool chip_power = true;   // GUFO_NPU_CHIP_POWER, logging only
  bool mem_trace = true;    // GUFO_NPU_MEM_TRACE, logging only
  GuardMode guard = GuardMode::kOn;  // GUFO_NPU_GUARD on|off|log
  GuardOverrides guard_test;         // GUFO_NPU_GUARD_TEST (internal)
  // GUFO_NPU_TEST_FAIL_CHUNK (internal): NPU chunk of a request to report as
  // invalid, counted from 0.
  std::optional<std::uint64_t> test_fail_chunk;
  Timing timing;
};
bool IsKnob(std::string_view name) noexcept;
// Absence selects the ON default; a present empty or malformed value fails.
// This boundary itself requires the immutable token, before interpreting DIR.
Knobs ParseKnobs(const core::ExperimentalAuthorization& authorization,
                 const std::map<std::string, std::string>& values);

inline constexpr std::uint32_t kTokens = 4096, kK = 2560, kN = 1024;
enum class Lane { LinearZ, QueryGate, SharedGate, SharedUp };
struct Q8Matrix {
  core::GgmlType type = core::GgmlType::kQ8_0;
  std::uint64_t rows = 0, cols = 0;
  // CPU bytes only, never a HIP device pointer. Remains owned by the caller
  // during qualification/copy; the staged result owns independent bytes.
  std::span<const std::byte> bytes;
};
struct WeightSlice {
  Lane lane;
  std::uint32_t row0 = 0, valid_rows = 0;
  std::uint32_t first_head = 0, head_count = 0;
};
std::vector<WeightSlice> LinearPlan(const Q8Matrix& weights, unsigned slices);
std::vector<WeightSlice> GatePlan(const Q8Matrix& weights, unsigned slices,
                                  unsigned query_heads, unsigned kv_heads,
                                  unsigned head_dim);
std::vector<WeightSlice> SharedPlan(const Q8Matrix& gate, const Q8Matrix& up,
                                    bool enabled);
// Packed Q8_0 rows, zero-padded to 1024 x 2560, NOT an encoded XRT B BO.
// BFP encoding and session fences protect the staged bytes.
std::vector<std::byte> StageQ8(const Q8Matrix& weights,
                               const WeightSlice& slice);
bool QualifiedPrefill(std::uint64_t tokens,
                      std::uint64_t registered_groups) noexcept;
// Whether a pass of `tokens` rows can take an NPU lane at all (token count
// only; the weights, the mode and the guard decide the rest): whole 4096
// groups, at most the executor batch capacity and at most 4 groups.
bool QualifiedChunk(unsigned tokens, unsigned capacity) noexcept;
}  // namespace gufo::npu
#endif
