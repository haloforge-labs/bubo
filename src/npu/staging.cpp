// SPDX-License-Identifier: MIT
// Lane geometry selectively adapted from the MIT fork's device_model.cpp.
#include "staging.hpp"

#include <windows.h>

#include <array>
#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "chunk_retry.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::npu {
namespace {
using Values = std::map<std::string, std::string>;
std::string_view Value(const Values& values, const char* name) {
  auto it = values.find(name);
  if (it == values.end())
    return {};
  if (it->second.empty())
    throw std::invalid_argument(std::string(name) + ": empty value");
  return it->second;
}
unsigned Number(const Values& values, const char* name, unsigned fallback,
                unsigned low, unsigned high) {
  auto text = Value(values, name);
  if (text.empty())
    return fallback;
  unsigned result = 0;
  auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      result < low || result > high)
    throw std::invalid_argument(std::string(name) + ": outside strict bounds");
  return result;
}
bool Switch(const Values& values, const char* name, bool fallback) {
  auto text = Value(values, name);
  if (text.empty())
    return fallback;
  if (text != "0" && text != "1")
    throw std::invalid_argument(std::string(name) + ": expected literal 0|1");
  return text == "1";
}
constexpr std::size_t kRowBytes = kK / 32 * sizeof(quant::block_q8_0);
void Matrix(const Q8Matrix& weights) {
  if (weights.type != core::GgmlType::kQ8_0 || weights.cols != kK ||
      weights.rows == 0 ||
      weights.rows > std::numeric_limits<std::size_t>::max() / kRowBytes ||
      weights.bytes.size() != weights.rows * kRowBytes ||
      weights.bytes.data() == nullptr)
    throw std::invalid_argument(
        "NPU requires complete CPU Q8_0 rows at K=2560");
}
void SliceCount(unsigned slices, bool zero) {
  if (slices > 12 || (!zero && slices == 0))
    throw std::invalid_argument("NPU slice count outside qualified bounds");
}
}  // namespace

bool IsKnob(std::string_view name) noexcept {
  return name == "GUFO_NPU_DIR" || name == "GUFO_NPU_SLICES" ||
         name == "GUFO_NPU_GATE_SLICES" || name == "GUFO_NPU_SHEXP" ||
         name == "GUFO_NPU_TIMEOUT_MS" || name == "GUFO_NPU_KEEPALIVE" ||
         name == "GUFO_NPU_KEEPALIVE_MS" || name == "GUFO_NPU_HAL_MONITOR" ||
         name == "GUFO_NPU_CHIP_POWER" || name == "GUFO_NPU_MEM_TRACE" ||
         name == "GUFO_NPU_GUARD" || name == "GUFO_NPU_GUARD_TEST" ||
         name == "GUFO_NPU_TEST_FAIL_CHUNK";
}

Knobs ParseKnobs(const core::ExperimentalAuthorization& authorization,
                 const Values& values) {
  authorization.Require(core::ExperimentalFeature::Prefill);
  Knobs knobs;
  auto directory = Value(values, "GUFO_NPU_DIR");
  if (directory.find('\0') != std::string_view::npos)
    throw std::invalid_argument("GUFO_NPU_DIR: invalid artifact path");
  if (directory.empty()) {
    std::array<wchar_t, 32768> executable{};
    const DWORD length = GetModuleFileNameW(nullptr, executable.data(),
                                            static_cast<DWORD>(executable.size()));
    if (!length || length >= executable.size())
      throw std::invalid_argument("NPU executable directory unavailable");
    knobs.artifacts = (std::filesystem::path(executable.data()).parent_path() /
                       ".." / "tools" / "npu").lexically_normal();
  } else {
    std::u8string utf8;
    for (const auto byte : directory)
      utf8.push_back(static_cast<char8_t>(static_cast<unsigned char>(byte)));
    knobs.artifacts = std::filesystem::path(utf8);
  }
  knobs.slices = Number(values, "GUFO_NPU_SLICES", 6, 1, 12);
  knobs.gate_slices = Number(values, "GUFO_NPU_GATE_SLICES", 0, 0, 12);
  knobs.shared_expert = Switch(values, "GUFO_NPU_SHEXP", true);
  knobs.hal_monitor = Switch(values, "GUFO_NPU_HAL_MONITOR", true);
  knobs.chip_power = Switch(values, "GUFO_NPU_CHIP_POWER", true);
  knobs.mem_trace = Switch(values, "GUFO_NPU_MEM_TRACE", true);
  if (const auto text = Value(values, "GUFO_NPU_GUARD"); !text.empty()) {
    if (!ParseGuardMode(std::string(text), &knobs.guard))
      throw std::invalid_argument(
          "GUFO_NPU_GUARD: expected literal on|off|log");
  } else if (values.contains("GUFO_NPU_GUARD")) {
    throw std::invalid_argument("GUFO_NPU_GUARD: expected literal on|off|log");
  }
  if (const auto text = Value(values, "GUFO_NPU_GUARD_TEST"); !text.empty()) {
    std::string error;
    if (!ParseGuardOverrides(std::string(text), &knobs.guard_test, &error))
      throw std::invalid_argument("GUFO_NPU_GUARD_TEST: " + error);
  }
  if (const auto text = Value(values, "GUFO_NPU_TEST_FAIL_CHUNK");
      !text.empty()) {
    std::uint64_t chunk = 0;
    std::string error;
    if (!ParseTestFailChunk(text, &chunk, &error))
      throw std::invalid_argument("GUFO_NPU_TEST_FAIL_CHUNK: " + error);
    knobs.test_fail_chunk = chunk;
  }
  (void)Switch(values, "GUFO_NPU_KEEPALIVE", true);
  knobs.timing = ParseTiming(Value(values, "GUFO_NPU_TIMEOUT_MS"),
                             Value(values, "GUFO_NPU_KEEPALIVE"),
                             Value(values, "GUFO_NPU_KEEPALIVE_MS"));
  return knobs;
}

std::vector<WeightSlice> LinearPlan(const Q8Matrix& weights, unsigned slices) {
  SliceCount(slices, false);
  Matrix(weights);
  if (weights.rows != 16384)
    throw std::invalid_argument("NPU linear z requires 16384 rows");
  std::vector<WeightSlice> plan;
  const auto start = 16384 - slices * kN;
  for (unsigned i = 0; i < slices; ++i)
    plan.push_back({Lane::LinearZ, start + i * kN, kN});
  return plan;
}

std::vector<WeightSlice> GatePlan(const Q8Matrix& weights, unsigned slices,
                                  unsigned query_heads, unsigned kv_heads,
                                  unsigned head_dim) {
  SliceCount(slices, true);
  if (slices == 0)
    return {};
  Matrix(weights);
  // Exactly the fork's qualified 24 query heads, paired query/gate rows;
  // the final two K/V heads remain GPU work, never staged as query rows.
  if (weights.rows != 13312 || query_heads != 24 || kv_heads != 2 ||
      head_dim != 256 || slices * 2 > query_heads)
    throw std::invalid_argument("NPU query/gate head geometry not qualified");
  std::vector<WeightSlice> plan;
  for (unsigned i = 0; i < slices; ++i)
    plan.push_back({Lane::QueryGate, i * kN, kN, i * 2, 2});
  return plan;
}

std::vector<WeightSlice> SharedPlan(const Q8Matrix& gate, const Q8Matrix& up,
                                    bool enabled) {
  if (!enabled)
    return {};
  Matrix(gate);
  Matrix(up);
  if (gate.rows != up.rows || gate.rows > kN)
    throw std::invalid_argument(
        "NPU shared gate/up rows must match in 1..1024");
  return {{Lane::SharedGate, 0, static_cast<std::uint32_t>(gate.rows)},
          {Lane::SharedUp, 0, static_cast<std::uint32_t>(up.rows)}};
}

std::vector<std::byte> StageQ8(const Q8Matrix& weights,
                               const WeightSlice& slice) {
  Matrix(weights);
  // Revalidate public slice descriptors; forged/overflowing row offsets must
  // never reach pointer arithmetic, allocation or copy.
  if (slice.valid_rows == 0 || slice.valid_rows > kN ||
      slice.row0 > weights.rows || slice.valid_rows > weights.rows - slice.row0)
    throw std::invalid_argument("NPU Q8_0 slice outside source rows");
  const auto shared =
      slice.lane == Lane::SharedGate || slice.lane == Lane::SharedUp;
  if ((!shared && slice.valid_rows != kN) ||
      (shared && (slice.row0 != 0 || weights.rows != slice.valid_rows)) ||
      (slice.lane == Lane::LinearZ &&
       (weights.rows != 16384 || slice.row0 % kN != 0 || slice.row0 < 4096)) ||
      (slice.lane == Lane::QueryGate &&
       (weights.rows != 13312 || slice.row0 % kN != 0 || slice.row0 >= 12288 ||
        slice.first_head != slice.row0 / 512 || slice.head_count != 2)) ||
      (slice.lane != Lane::LinearZ && slice.lane != Lane::QueryGate && !shared))
    throw std::invalid_argument("NPU Q8_0 slice lane not qualified");
  const auto input = weights.bytes.subspan(slice.row0 * kRowBytes,
                                           slice.valid_rows * kRowBytes);
  for (std::size_t i = 0; i < input.size(); i += sizeof(quant::block_q8_0)) {
    std::uint16_t scale;
    std::memcpy(&scale, input.data() + i, sizeof(scale));
    if ((scale & 0x7c00) == 0x7c00)
      throw std::invalid_argument("NPU Q8_0 scale is not finite");
  }
  std::vector<std::byte> output(kN * kRowBytes, std::byte{0});
  std::memcpy(output.data(), input.data(), input.size());
  return output;
}

bool QualifiedPrefill(std::uint64_t tokens,
                      std::uint64_t registered_groups) noexcept {
  return tokens >= kTokens && tokens % kTokens == 0 &&
         tokens / kTokens <= registered_groups;
}

bool QualifiedChunk(unsigned tokens, unsigned capacity) noexcept {
  return QualifiedPrefill(tokens, capacity / kTokens) && tokens <= capacity &&
         tokens <= 4 * kTokens;
}
}  // namespace gufo::npu
