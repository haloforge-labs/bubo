// Metadata-only old/new loader and request-state equivalence. No HIP or
// weights.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "src/core/session_state_bytes.hpp"
#include "src/models/qwen36_a3b/config.hpp"
#include "tests/models/gemma4/gguf_builder.hpp"

namespace {
namespace a3b = gufo::models::qwen36_a3b;
// Independent architecture spelling from 3ff74dce engine.cpp.
constexpr std::string_view kArch = "qwen35moe";
std::size_t checks = 0;
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
  ++checks;
}
void Fail(std::string* error, std::string message) {
  if (error && error->empty())
    *error = std::move(message);
}

// clang-format off
// Copied from 3ff74dce engine.hpp; only the type name is test-local.
struct OldConfig {
  std::uint32_t layers{0};
  std::uint32_t hidden{0};
  std::uint32_t vocab{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t rotary_dim{0};
  float rope_theta{0};
  float eps{0};
  std::uint32_t full_attention_interval{0};
  std::uint32_t ssm_conv_kernel{0};
  std::uint32_t ssm_head_dim{0};
  std::uint32_t ssm_k_heads{0};
  std::uint32_t ssm_v_heads{0};
  std::uint32_t experts{0};
  std::uint32_t experts_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};
  std::uint32_t nextn{0};  // MTP blocks after the trunk (not in `layers`)

  bool IsLinear(std::uint32_t layer) const {
    return (layer + 1) % full_attention_interval != 0;
  }
  std::uint32_t ConvChannels() const {
    return (2 * ssm_k_heads + ssm_v_heads) * ssm_head_dim;
  }
  std::uint32_t SsmValueDim() const { return ssm_v_heads * ssm_head_dim; }
  std::uint32_t QDim() const { return heads * head_dim; }
  std::uint32_t KvDim() const { return kv_heads * head_dim; }
};

// Body copied verbatim from 3ff74dce src/models/qwen36_a3b/engine.cpp,
// Engine::Bind. Only the wrapper supplies its GGUF and Config explicitly.
bool OldReadConfig(const gufo::core::GgufReader& g, OldConfig& c_,
                   std::string* error) {
  if (g.GetMetadataString("general.architecture") != kArch) {
    Fail(error, "GGUF architecture is not qwen35moe");
    return false;
  }
  bool ok = true;
  const auto u32 = [&](std::string_view key) {
    const auto v =
        g.GetMetadataUint32(std::string(kArch) + "." + std::string(key));
    if (!v) {
      Fail(error, "missing GGUF key " + std::string(key));
      ok = false;
    }
    return v.value_or(0);
  };
  const auto f32 = [&](std::string_view key) {
    const auto v =
        g.GetMetadataFloat32(std::string(kArch) + "." + std::string(key));
    if (!v) {
      Fail(error, "missing GGUF key " + std::string(key));
      ok = false;
    }
    return v.value_or(0.0F);
  };
  c_.layers = u32("block_count");
  c_.hidden = u32("embedding_length");
  c_.heads = u32("attention.head_count");
  c_.kv_heads = u32("attention.head_count_kv");
  c_.head_dim = u32("attention.key_length");
  c_.rotary_dim = u32("rope.dimension_count");
  c_.rope_theta = f32("rope.freq_base");
  c_.eps = f32("attention.layer_norm_rms_epsilon");
  c_.full_attention_interval = u32("full_attention_interval");
  c_.ssm_conv_kernel = u32("ssm.conv_kernel");
  c_.ssm_head_dim = u32("ssm.state_size");
  c_.ssm_k_heads = u32("ssm.group_count");
  c_.ssm_v_heads = u32("ssm.time_step_rank");
  c_.experts = u32("expert_count");
  c_.experts_used = u32("expert_used_count");
  c_.expert_ff = u32("expert_feed_forward_length");
  c_.shared_ff = u32("expert_shared_feed_forward_length");
  if (!ok)
    return false;
  // MTP files append their nextn block(s) after the trunk.
  c_.nextn = g.GetMetadataUint32(std::string(kArch) + ".nextn_predict_layers")
                 .value_or(0);
  if (c_.nextn >= c_.layers) {
    Fail(error, "bad nextn_predict_layers");
    return false;
  }
  c_.layers -= c_.nextn;
  // The FN kernels this runtime borrows fix these geometry points.
  if (c_.head_dim != 256 || c_.ssm_head_dim != 128 || c_.ssm_conv_kernel != 4 ||
      c_.heads % c_.kv_heads != 0 || c_.ssm_v_heads % c_.ssm_k_heads != 0 ||
      c_.experts > 512 || c_.full_attention_interval == 0 ||
      c_.hidden % 32 != 0) {
    Fail(error, "unsupported qwen35moe geometry");
    return false;
  }
  return true;
}

// Allocation loop copied verbatim from 3ff74dce engine.cpp. CPU no-op
// allocation sinks preserve the old formula without allocating weights/state.
std::size_t OldStateBytes(const OldConfig& c_, std::uint32_t context) {
  struct { std::uint32_t max_context; } options_{context};
  struct Layer { bool linear; };
  std::vector<Layer> layers_;
  for (std::uint32_t i = 0; i < c_.layers; ++i)
    layers_.push_back({(i + 1) % c_.full_attention_interval != 0});
  const std::size_t ch = c_.ConvChannels();
  using __half = std::uint16_t;  // CPU spelling of the old two-byte HIP type.
  std::vector<float*> conv_state_(c_.layers), ssm_state_(c_.layers);
  std::vector<__half*> k_cache_(c_.layers), v_cache_(c_.layers), vt_cache_(c_.layers);
  const auto f32 = [](std::size_t) -> float* { return nullptr; };
  const auto alloc = [](std::size_t) -> void* { return nullptr; };
  std::size_t state_bytes = 0;
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (layers_[i].linear) {
      const std::size_t conv = (c_.ssm_conv_kernel - 1) * ch;
      const std::size_t st = static_cast<std::size_t>(c_.ssm_v_heads) *
                             c_.ssm_head_dim * c_.ssm_head_dim;
      conv_state_[i] = f32(conv);
      ssm_state_[i] = f32(st);
      state_bytes += (conv + st) * sizeof(float);
    } else {
      const std::size_t kv =
          static_cast<std::size_t>(options_.max_context) * c_.KvDim();
      k_cache_[i] = static_cast<__half*>(alloc(kv * sizeof(__half)));
      v_cache_[i] = static_cast<__half*>(alloc(kv * sizeof(__half)));
      vt_cache_[i] = static_cast<__half*>(alloc(kv * sizeof(__half)));
      state_bytes += 3 * kv * sizeof(__half);
    }
  }
  return state_bytes;
}

// clang-format on

auto Fields(const auto& c) {
  return std::tie(c.layers, c.hidden, c.vocab, c.heads, c.kv_heads, c.head_dim,
                  c.rotary_dim, c.rope_theta, c.eps, c.full_attention_interval,
                  c.ssm_conv_kernel, c.ssm_head_dim, c.ssm_k_heads,
                  c.ssm_v_heads, c.experts, c.experts_used, c.expert_ff,
                  c.shared_ff, c.nextn);
}
gemma4_test::Metadata Metadata(std::uint32_t nextn) {
  // Same keys/geometry as tests/cli/contract/models_test.cpp::WriteA3b;
  // production trunk count 40 from docs/models/qwen3.6-35b-a3b/README.md.
  // Cyber-Tiel's MTP key is present, base's optional key can be absent.
  gemma4_test::Metadata fields{{"general.architecture", std::string(kArch)}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 40 + nextn},
           {"embedding_length", 2048},
           {"attention.head_count", 16},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"rope.dimension_count", 64},
           {"full_attention_interval", 4},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 32},
           {"expert_count", 256},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 512},
           {"expert_shared_feed_forward_length", 512},
           {"context_length", 262144}})
    fields[std::string(kArch) + "." + key] = value;
  fields["qwen35moe.rope.freq_base"] = 10000000.0F;
  fields["qwen35moe.attention.layer_norm_rms_epsilon"] = 1e-6F;
  if (nextn)
    fields["qwen35moe.nextn_predict_layers"] = nextn;
  return fields;
}
void CheckLoad(const gemma4_test::Metadata& fields, bool expected) {
  gemma4_test::GgufImage image(fields);
  Require(image.reader() != nullptr, image.error());
  OldConfig old_config;
  a3b::Config new_config;
  std::string old_error, new_error;
  const bool old_ok = OldReadConfig(*image.reader(), old_config, &old_error);
  const bool new_ok = a3b::ReadConfig(*image.reader(), new_config, &new_error);
  Require(old_ok == expected && new_ok == old_ok, "loader acceptance differs");
  Require(Fields(old_config) == Fields(new_config), "loader fields differ");
  Require(old_error == new_error, "loader diagnostics differ");
}
}  // namespace

int main() {
  try {
    for (const auto nextn : {0U, 1U}) {
      const auto fields = Metadata(nextn);
      const auto* name = nextn ? "Cyber-Tiel-MTP" : "Qwen3.6-35B-A3B";
      CheckLoad(fields, true);
      gemma4_test::GgufImage image(fields);
      OldConfig old_config;
      a3b::Config config;
      std::string error;
      Require(a3b::ReadConfig(*image.reader(), config, &error) &&
                  OldReadConfig(*image.reader(), old_config, &error),
              error);
      Require(config.layers == 40 && config.nextn == nextn,
              "nextn is excluded from the trunk");
      std::size_t cases = 0;
      for (const auto context : {1U, 4096U, 32768U, 65536U, 262144U})
        for (const auto sessions : {1U, 2U, 4U, 8U}) {
          const auto old_bytes = OldStateBytes(old_config, context);
          const auto new_bytes = gufo::core::A3bSessionStateBytes(
              config.layers, config.full_attention_interval, config.KvDim(),
              config.ssm_conv_kernel, config.ConvChannels(), config.ssm_v_heads,
              config.ssm_head_dim, context);
          // Multiple session counts check arithmetic, not runtime support:
          // this engine currently serves one session.
          Require(old_bytes == new_bytes &&
                      old_bytes * sessions == new_bytes * sessions,
                  std::string(name) + " state bytes differ");
          ++cases;
        }
      std::cout << "PASS: " << name
                << " 19 loader fields equal; 5 contexts x 4 session counts = "
                << cases << " byte-equal cases\n";
      for (const auto& [key, ignored] : fields) {
        if (key == "qwen35moe.context_length" ||
            key == "qwen35moe.nextn_predict_layers")
          continue;
        auto missing = fields;
        missing.erase(key);
        CheckLoad(missing, false);
      }
      auto malformed = fields;
      malformed["qwen35moe.nextn_predict_layers"] = 41U;
      CheckLoad(malformed, false);
      malformed = fields;
      malformed["qwen35moe.attention.key_length"] = 128U;
      CheckLoad(malformed, false);
    }
    std::cout << "PASS: " << checks
              << " checks; 40 byte-equal cases; loader rejection parity\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
