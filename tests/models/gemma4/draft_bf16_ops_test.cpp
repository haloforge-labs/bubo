// GPU-only regression: real upload and two drafter steps at both target
// widths, including the full 1024 x 262144 BF16 vocabulary allocation.
// Sparse BF16 weights and zero transformer branches give an independent
// CPU oracle without running a full target model or a CPU transformer.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/gguf_builder.hpp"

namespace g4 = gufo::models::gemma4;
namespace gpu = g4::rocm;
using gemma4_test::Require;
using gemma4_test::TensorSpec;
using gufo::core::GgmlType;
using gufo::models::qwen38_flash_next::MtpCandidateLogits;

namespace {

constexpr std::uint32_t kHidden = 1024, kVocab = 262144, kLayers = 4;

std::uint32_t Heads(std::uint32_t target) {
  return target == 2816 ? 16 : 32;
}

// WeightUpload needs a real file descriptor, even for a synthetic GGUF.
struct FixtureFile {
  std::filesystem::path directory, path;

  explicit FixtureFile(const gemma4_test::GgufImage& image) {
    directory =
        std::filesystem::temp_directory_path() /
        ("gufo-gemma4-bf16-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    Require(std::filesystem::create_directory(directory),
            "cannot create private GGUF fixture directory");
    path = directory / "head.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(image.reader()->GetData()),
              static_cast<std::streamsize>(image.reader()->GetSize()));
    out.close();
    Require(bool(out), "cannot write BF16 GGUF fixture");
  }

  ~FixtureFile() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
};

TensorSpec Bf16(std::uint32_t cols, std::uint32_t rows) {
  return {{cols, rows},
          GgmlType::kBF16,
          std::vector<std::uint8_t>(std::size_t{cols} * rows * 2, 0)};
}

void Set(TensorSpec& t, std::uint32_t row, std::uint32_t col, float value) {
  const auto bits =
      static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
  Require(std::bit_cast<float>(std::uint32_t{bits} << 16) == value,
          "fixture value must be exactly representable in BF16");
  std::memcpy(t.data.data() + (row * t.dims[0] + col) * 2, &bits, 2);
}

TensorSpec Ones(std::uint32_t n) {
  TensorSpec t{{n}, GgmlType::kF32, std::vector<std::uint8_t>(n * 4)};
  const float one = 1.0F;
  for (std::uint32_t i = 0; i < n; ++i) {
    std::memcpy(t.data.data() + i * 4, &one, 4);
  }
  return t;
}

g4::TensorRef Tensor(const gufo::core::GgufReader& reader, const char* name) {
  const auto* t = reader.FindTensor(name);
  Require(t != nullptr, std::string("missing fixture tensor ") + name);
  g4::TensorRef ref;
  ref.data = t->data;
  ref.type = t->type;
  ref.cols = t->dimensions[0];
  ref.rows = t->dimensions.size() > 1 ? t->dimensions[1] : 1;
  ref.name = t->name;
  ref.file_offset = static_cast<const std::uint8_t*>(t->data) -
                    static_cast<const std::uint8_t*>(reader.GetData());
  return ref;
}

float Read(const g4::TensorRef& t, std::size_t row, std::size_t col) {
  std::uint16_t bits;
  std::memcpy(
      &bits,
      static_cast<const std::uint8_t*>(t.data) + (row * t.cols + col) * 2, 2);
  return std::bit_cast<float>(std::uint32_t{bits} << 16);
}

gemma4_test::Metadata Metadata(std::uint32_t target) {
  const std::string p = "gemma4-assistant.";
  return {
      {"general.architecture", std::string("gemma4-assistant")},
      {p + "block_count", kLayers},
      {p + "embedding_length", kHidden},
      {p + "embedding_length_out", target},
      {p + "context_length", std::uint32_t{16}},
      {p + "feed_forward_length", std::uint32_t{128}},
      {p + "attention.head_count", Heads(target)},
      {p + "attention.head_count_kv",
       gemma4_test::I32Array{{int(Heads(target) / 2), int(Heads(target) / 2),
                              int(Heads(target) / 2), int(Heads(target) / 8)}}},
      {p + "attention.sliding_window_pattern",
       gemma4_test::BoolArray{{true, true, true, false}}},
      {p + "attention.key_length", std::uint32_t{512}},
      {p + "attention.value_length", std::uint32_t{512}},
      {p + "attention.key_length_swa", std::uint32_t{256}},
      {p + "attention.value_length_swa", std::uint32_t{256}},
      {p + "rope.dimension_count", std::uint32_t{512}},
      {p + "rope.dimension_count_swa", std::uint32_t{256}},
      {p + "rope.freq_base", 1000000.0F},
      {p + "rope.freq_base_swa", 10000.0F},
      {p + "attention.sliding_window", std::uint32_t{16}},
      {p + "attention.layer_norm_rms_epsilon", 1e-6F},
      {p + "attention.shared_kv_layers", kLayers},
      {p + "nextn_predict_layers", kLayers},
  };
}

// Only tokens 0..63 are embedded by this fixture. The target graph is never
// run; sliding/global zero KV caches supply the drafter's shared attention.
// Header geometry matches the local Huihui heads: 16/32 query heads,
// KV [8,8,8,2]/[16,16,16,4], hd256 sliding and hd512 global.
void CheckTargetWidth(std::uint32_t target) {
  gemma4_test::Tensors tensors;
  auto& embedding = tensors["target_embedding"] = Bf16(target, 64);
  for (std::uint32_t r = 0; r < 64; ++r) {
    for (std::uint32_t c = 0; c < target; ++c) {
      Set(embedding, r, c, 0.25F + float((r + c) % 17) / 64.0F);
    }
  }
  tensors["target_norm"] = Ones(target);
  tensors["target_k_norm_swa"] = Ones(256);
  tensors["target_k_norm_global"] = Ones(512);
  auto& pre = tensors["nextn.pre_projection.weight"] =
      Bf16(2 * target, kHidden);
  for (std::uint32_t r = 0; r < kHidden; ++r) {
    Set(pre, r, r, 0.5F);
    Set(pre, r, target + r, 0.25F);
  }
  auto& post = tensors["nextn.post_projection.weight"] = Bf16(kHidden, target);
  for (std::uint32_t r = 0; r < target; ++r) {
    Set(post, r, r % kHidden, 0.5F + float(r % 3) / 8.0F);
  }
  auto& vocab = tensors["token_embd.weight"] = Bf16(kHidden, kVocab);
  // Exactly 64 positive logits; the remaining vocabulary rows are zero.
  for (std::uint32_t r = 0; r < 64; ++r) {
    Set(vocab, r, r, float(r + 1) / 8.0F);
  }
  tensors["output_norm.weight"] = Ones(kHidden);
  tensors["rope_freqs.weight"] = Ones(256);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    for (const char* norm : {"attn_norm.weight", "post_attention_norm.weight",
                             "ffn_norm.weight", "post_ffw_norm.weight"}) {
      tensors[p + norm] = Ones(kHidden);
    }
    const std::uint32_t dim = l == 3 ? 512 : 256;
    const std::uint32_t q_dim = Heads(target) * dim;
    tensors[p + "attn_q_norm.weight"] = Ones(dim);
    tensors[p + "attn_q.weight"] = Bf16(kHidden, q_dim);
    tensors[p + "attn_output.weight"] = Bf16(q_dim, kHidden);
    tensors[p + "ffn_gate.weight"] = Bf16(kHidden, 128);
    tensors[p + "ffn_up.weight"] = Bf16(kHidden, 128);
    tensors[p + "ffn_down.weight"] = Bf16(128, kHidden);
  }
  gemma4_test::GgufImage image(Metadata(target), tensors);
  tensors.clear();
  Require(image.reader() != nullptr, image.error());
  FixtureFile file(image);
  std::string error;
  const auto disk_reader = gufo::core::GgufReader::OpenFile(file.path, &error);
  Require(disk_reader != nullptr, "open BF16 GGUF fixture: " + error);
  const auto& reader = *disk_reader;
  g4::ModelWeights weights;
  auto& c = weights.config;
  c.num_layers = 2;
  c.hidden_size = target;
  c.ffn_size = 128;
  c.context_length = 16;
  c.num_heads = Heads(target);
  c.kv_heads = {Heads(target) / 2, Heads(target) / 8};
  c.sliding = {1, 0};
  c.head_dim_global = c.rope_dim_global = 512;
  c.head_dim_sliding = c.rope_dim_sliding = 256;
  c.rope_theta_global = 1000000.0F;
  c.rope_theta_sliding = 10000.0F;
  c.sliding_window = 16;
  weights.vocab_size = kVocab;
  weights.token_embd = Tensor(reader, "target_embedding");
  weights.output = weights.token_embd;
  weights.output_norm = Tensor(reader, "target_norm");
  weights.rope_factors.values.assign(256, 1.0F);
  weights.layers.resize(2);
  weights.layers[0].attn_k_norm = Tensor(reader, "target_k_norm_swa");
  weights.layers[1].attn_k_norm = Tensor(reader, "target_k_norm_global");
  const auto draft = g4::DraftWeights::Bind(reader, weights, &error);
  Require(draft.has_value(), "bind BF16 head: " + error);
  const auto device =
      gpu::DeviceModel::Upload(weights, reader, &*draft, &reader, &error);
  Require(device != nullptr, "upload BF16 head: " + error);
  for (const auto* t :
       {&device->draft().pre_projection, &device->draft().post_projection}) {
    Require(t->type == GgmlType::kF16, "BF16 head was not converted");
  }
  const auto& head = device->draft().token_embd;
  Require(head.type == GgmlType::kQ4_K, "BF16 vocabulary must use Q4_K");
  constexpr std::size_t row_bytes = kHidden / 256 * 144;
  std::vector<std::uint8_t> encoded(64 * row_bytes);
  HIP_CHECK(hipMemcpy(encoded.data(), head.data, encoded.size(),
                      hipMemcpyDeviceToHost));
  std::vector<float> decoded(64 * kHidden);
  for (std::uint32_t r = 0; r < 64; ++r) {
    auto* out = decoded.data() + r * kHidden;
    const auto* in = encoded.data() + r * row_bytes;
    gufo::quant::DequantizeQ4_K(in, out, kHidden);
    for (std::uint32_t col = 0; col < kHidden; ++col) {
      const double original = Read(draft->token_embd, r, col);
      // This sparse fixture has one nonzero weight per row. Q8_0 stores
      // it at level 127; Q4_K adds one half-scale rounding at level 15.
      // Combined relative scale error is below 0.0015 (not a dense-matrix
      // quality guarantee). Fitting a Q4_K offset can add FP32 noise to
      // zeros in the one live block; every wholly zero block stays zero.
      Require(std::isfinite(out[col]) &&
                  std::abs(out[col] - original) <=
                      0.0015 * std::abs(original) + 2e-7 * double(r + 1) / 8,
              "sparse vocabulary exceeded the quantization error bound");
    }
  }
  std::vector<std::uint8_t> tail(4096, 1);
  HIP_CHECK(hipMemcpy(
      tail.data(),
      static_cast<const std::uint8_t*>(head.data) + row_bytes * kVocab,
      tail.size(), hipMemcpyDeviceToHost));
  Require(std::all_of(tail.begin(), tail.end(), [](auto v) { return v == 0; }),
          "vocabulary allocation must retain its zeroed 4 KiB tail");
  Require(device->token_embd().type == GgmlType::kBF16 &&
              device->output().data == device->token_embd().data &&
              device->draft().output_norm.type == GgmlType::kF32,
          "target embedding tie and draft norm must remain untouched");
  gpu::Executor executor(*device, 8, 1, 16);
  for (const std::uint32_t rows : {1U, 2U, 8U}) {
    std::vector<std::unique_ptr<gpu::KvCache>> caches;
    std::vector<gpu::Executor::DraftJob> jobs;
    std::vector<std::vector<std::int32_t>> proposals(rows);
    std::vector<std::vector<float>> hidden(rows, std::vector<float>(target));
    std::vector<std::uint32_t> calls(rows, 0);
    for (std::uint32_t j = 0; j < rows; ++j) {
      auto cache = executor.CreateCache(16, &error);
      Require(cache != nullptr, error);
      for (std::uint32_t i = 0; i < target; ++i) {
        hidden[j][i] = 0.125F + float((i + 3 * j) % 23) / 64.0F;
      }
      HIP_CHECK(hipMemsetAsync(cache->allocation, 0, cache->bytes,
                               executor.stream()));
      HIP_CHECK(hipMemcpyAsync(cache->hidden, hidden[j].data(), target * 4,
                               hipMemcpyHostToDevice, executor.stream()));
      auto propose = [&, j](const MtpCandidateLogits& logits) {
        const std::uint32_t token = calls[j] == 0 ? j : 7U;
        // The branches add zero to the pre-projection residual; its final
        // state is just RMSNorm(pre * [scaled embedding; previous hidden]).
        std::vector<double> h(kHidden);
        double squares = 0.0;
        const float scale = std::sqrt(float(target));
        for (std::uint32_t i = 0; i < kHidden; ++i) {
          const float e = Read(weights.token_embd, token, i) * scale;
          h[i] =
              double(Read(draft->pre_projection, i, i)) * e +
              double(Read(draft->pre_projection, i, target + i)) * hidden[j][i];
          squares += h[i] * h[i];
        }
        const double rms = 1.0 / std::sqrt(squares / kHidden + c.rms_eps);
        for (double& value : h) {
          value *= rms;
        }
        Require(logits.size == 64, "must return all top-64 logits");
        std::vector<bool> seen(64, false);
        for (std::size_t i = 0; i < logits.size; ++i) {
          const auto id = logits.ids[i];
          Require(id < 64 && !seen[id], "wrong top-64 candidate set");
          seen[id] = true;
          double want = 0;
          for (std::uint32_t col = 0; col < kHidden; ++col) {
            want += decoded[id * kHidden + col] * h[col];
          }
          Require(std::isfinite(logits.logits[i]) &&
                      std::abs(logits.logits[i] - want) <=
                          3e-5 * std::max(1.0, std::abs(want)),
                  "draft logit differs from CPU dot of uploaded vocabulary");
          const double original = Read(draft->token_embd, id, id) * h[id];
          Require(std::abs(logits.logits[i] - original) <=
                      0.0015 * std::abs(original) +
                          3e-5 * std::max(1.0, std::abs(original)),
                  "draft logit exceeded the BF16 reference quantization bound");
        }
        for (std::uint32_t i = 0; i < target; ++i) {
          hidden[j][i] = static_cast<float>(
              Read(draft->post_projection, i, i % kHidden) * h[i % kHidden]);
        }
        ++calls[j];
        return gpu::DraftProposal{7, false};
      };
      jobs.push_back({cache.get(), static_cast<std::int32_t>(j), 1, 2, propose,
                      &proposals[j]});
      caches.push_back(std::move(cache));
    }
    executor.DraftChains(jobs);
    for (std::uint32_t j = 0; j < rows; ++j) {
      Require(
          calls[j] == 2 && proposals[j] == std::vector<std::int32_t>({7, 7}),
          "both steps must execute, including the post-projection");
    }
  }
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckTargetWidth(2816);
    CheckTargetWidth(5376);
  });
}
