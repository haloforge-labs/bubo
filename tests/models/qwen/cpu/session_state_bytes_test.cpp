// CPU arithmetic regression for the arena extraction; no model or GPU loads.
#include "src/core/session_state_bytes.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "src/models/qwen/hip/execution_policy.hpp"
// Reuse the existing in-memory GGUF writer; it writes metadata only here.
#include "tests/models/gemma4/gguf_builder.hpp"

namespace {
namespace core = gufo::core;
using namespace gufo::hip;
std::size_t checks = 0;
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
  ++checks;
}

// clang-format off
// Copied verbatim from 3ff74dce src/models/qwen/hip/arena.cpp:
// checked allocation helpers, required locals and request-state calculation.
std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Qwen GPU allocation size overflows");
  }
  return left * right;
}

std::size_t CheckedSum(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen GPU allocation total overflows");
  }
  return left + right;
}

void CheckedAdd(std::size_t value, std::size_t* total) {
  *total = CheckedSum(*total, value);
}

void AddAllocation(std::size_t elements, std::size_t element_bytes,
                   std::size_t* total) {
  CheckedAdd(CheckedMultiply(elements, element_bytes), total);
}

std::size_t OldStateBytes(const core::ModelConfig& config, std::uint32_t max_context,
                          QwenExecutionPolicy policy) {
  const std::size_t context = std::max(max_context, 1U);
  const std::size_t kv =
      CheckedMultiply(config.num_key_value_heads, config.head_dim);
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t time_step = config.ssm_time_step_rank;
  struct { std::size_t request_state_bytes = 0; } usage;
  auto& state = usage.request_state_bytes;
  const std::size_t total_kv = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.FullAttentionLayerCount(), kv),
                      context),
      1);
  AddAllocation(
      CheckedMultiply(total_kv, 2),
      policy.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float),
      &state);

  const std::size_t total_conv = CheckedMultiply(
      CheckedMultiply(config.SsmLayerCount(), ssm_qkv), config.ssm_conv_kernel);
  AddAllocation(total_conv, sizeof(float), &state);
  const std::size_t total_deltanet = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.SsmLayerCount(), time_step),
                      config.ssm_state_size),
      config.SsmValueSize());
  AddAllocation(total_deltanet,
                QwenRecurrentStateElementBytes(policy.recurrent_state_storage),
                &state);
  return usage.request_state_bytes;
}

// clang-format on

core::ModelConfig Config(bool flash) {
  // Flash geometry: tests/models/qwen38_flash_next/config_test.cpp,
  // ValidMetadata (shared by Huihui and base). Dense 27B geometry:
  // tests/models/qwen/cpu/ssm_test.cpp. These are independent fixtures,
  // not values obtained from the state-byte function being tested.
  const std::string arch = flash ? "qwen4exp" : "qwen35";
  gemma4_test::Metadata fields{{"general.architecture", arch}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", flash ? 48U : 64U},
           {"embedding_length", flash ? 2560U : 5120U},
           {"feed_forward_length", 17408},
           {"attention.head_count", 24},
           {"attention.head_count_kv", flash ? 2U : 4U},
           {"attention.key_length", 256},
           {"context_length", 262144},
           {"full_attention_interval", 4},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"rope.dimension_count", 64}})
    fields[arch + "." + key] = value;
  gemma4_test::GgufImage image(fields);
  Require(image.reader() != nullptr, image.error());
  std::string error;
  const auto config = image.reader()->ExtractModelConfig(&error);
  Require(config.has_value(), error);
  return *config;
}
}  // namespace

int main() {
  try {
    for (const auto name :
         {"FlashNext-Huihui", "FlashNext-base", "Qwen3.6-27B"}) {
      const auto config = Config(std::string_view(name) != "Qwen3.6-27B");
      std::size_t cases = 0;
      for (const auto context : {1U, 4096U, 32768U, 65536U, 262144U})
        for (const auto sessions : {1U, 2U, 4U, 8U})
          for (const auto kv :
               {QwenKvCacheStorage::kFp16, QwenKvCacheStorage::kFp32})
            for (const auto recurrent : {QwenRecurrentStateStorage::kFp32,
                                         QwenRecurrentStateStorage::kBf16}) {
              QwenExecutionPolicy policy;
              policy.kv_cache_storage = kv;
              policy.recurrent_state_storage = recurrent;
              const auto old_bytes = OldStateBytes(config, context, policy);
              const auto new_bytes = core::QwenDenseSessionStateBytes(
                  config, context, policy.UsesFp16AttentionKv() ? 2 : 4,
                  QwenRecurrentStateElementBytes(recurrent));
              Require(old_bytes == new_bytes &&
                          old_bytes * sessions == new_bytes * sessions,
                      std::string(name) +
                          " context=" + std::to_string(context) +
                          " sessions=" + std::to_string(sessions));
              ++cases;
            }
      std::cout << "PASS: " << name
                << " 5 contexts x 4 session counts x 4 storage modes = "
                << cases << " byte-equal cases\n";
    }
    // Flash-Next's actual runtime has indexer/rollback/PLE state and uses
    // Executor::SessionBytes, not QwenDenseSessionStateBytes. Its geometry
    // above checks this extracted function only, not that runtime's claim.
    std::cout << "PASS: " << checks << " checks; 240 byte-equal cases\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
