#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/iq_decode_bits.hpp"
#include "src/models/qwen38_flash_next/native_dispatch.hpp"
#include "src/models/qwen38_flash_next/native_gather.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using Type = gufo::core::GgmlType;

namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}

// Exercise the exact helper used by the GPU gather on one real weight row.
void CompareRow(const qfn::TensorRef& t, bool& same_bytes, float& max_error) {
  const auto row = std::min<std::uint64_t>(17, t.rows - 1);
  const auto* src = t.Expert(0) + row * t.RowBytes();
  std::vector<float> old(t.cols), now(t.cols), x(t.cols);
  qfn::cpu::DequantizeRow(t, 0, row, old.data());
  Require(std::any_of(old.begin(), old.end(), [](float v) { return v != 0; }),
          "selected real row is all zero: " + std::string(t.name));
  for (std::size_t b = 0; b < t.cols / 256; ++b) {
    float* dst = now.data() + b * 256;
    if (t.type == Type::kIQ3_S) {
      const auto& block = reinterpret_cast<const gufo::quant::block_iq3_s*>(src)[b];
      for (int lane = 0; lane < 32; ++lane)
        qfn::native::Iq3S(block, gufo::quant::Fp16ToFloat(block.d), dst, lane,
                          gufo::quant::Iq3sGrid());
    } else if (t.type == Type::kIQ4_XS) {
      const auto& block = reinterpret_cast<const gufo::quant::block_iq4_xs*>(src)[b];
      for (int lane = 0; lane < 32; ++lane)
        qfn::native::Iq4Xs(block, gufo::quant::Fp16ToFloat(block.d), dst, lane,
                           gufo::quant::kValuesIq4Nl);
    } else {
      const auto& block = reinterpret_cast<const gufo::quant::block_q6_K*>(src)[b];
      for (int lane = 0; lane < 64; ++lane)
        qfn::native::Q6K(block, gufo::quant::Fp16ToFloat(block.d), dst, lane);
    }
  }
  double dot = 0;
  for (std::size_t i = 0; i < t.cols; ++i) {
    x[i] = 0.25F * std::sin(static_cast<float>(i) * 0.031F);
    Require(std::isfinite(now[i]), "nonfinite native gather value");
    max_error = std::max(max_error, std::fabs(now[i] - old[i]));
    dot += static_cast<double>(now[i]) * x[i];
  }
  const float reference = qfn::cpu::DotRow(t, 0, row, x);
  Require(std::fabs(static_cast<float>(dot) - reference) <=
              1e-6F * std::max(1.0F, std::fabs(reference)),
          "native gather dot differs from CPU reference: " + std::string(t.name));
  same_bytes &= std::memcmp(old.data(), now.data(), now.size() * sizeof(float)) == 0;
  std::printf("real-row tensor=%.*s type=%.*s row=%llu cols=%llu dot=%.9g\n",
              static_cast<int>(t.name.size()), t.name.data(),
              static_cast<int>(gufo::core::ToString(t.type).size()),
              gufo::core::ToString(t.type).data(),
              static_cast<unsigned long long>(row),
              static_cast<unsigned long long>(t.cols), reference);
}

// Every dense type needs a storage geometry and both kernel routes; a type in
// kNativeDenseTypes that FormatOf does not know binds as "unsupported format".
void CheckDenseTypeTables() {
  for (const auto type : qfn::kNativeDenseTypes) {
    const std::string name(gufo::core::ToString(type));
    qfn::TensorRef t;
    t.type = type;
    t.cols = 256;
    Require(t.RowBytes() != 0, "no storage geometry for dense type " + name);
    Require(qfn::NativeDenseKernel(type, true) != nullptr &&
                qfn::NativeDenseKernel(type, false) != nullptr,
            "no native kernel for dense type " + name);
  }
  std::printf(
      "%zu dense types: storage geometry and both kernel routes present\n",
      qfn::kNativeDenseTypes.size());
}

// The reference dequantizer must decode Q3_K and must not zero-fill a type it
// does not know. Q3_K keeps its fp16 scale in the last two bytes of a block.
void CheckQ3KRow() {
  constexpr std::size_t kBlocks = 4, kBlockBytes = 110;
  std::vector<std::uint8_t> bytes(kBlocks * kBlockBytes);
  std::uint32_t state = 12345;
  for (auto& b : bytes) {
    state = state * 1664525U + 1013904223U;
    b = static_cast<std::uint8_t>(state >> 24);
  }
  for (std::size_t b = 0; b < kBlocks; ++b) {
    bytes[b * kBlockBytes + 108] = 0x00;  // fp16 0.5
    bytes[b * kBlockBytes + 109] = 0x38;
  }
  qfn::TensorRef t;
  t.data = bytes.data();
  t.type = Type::kQ3_K;
  t.cols = kBlocks * 256;
  t.rows = 1;
  t.experts = 1;
  Require(t.RowBytes() == bytes.size(), "Q3_K row bytes");
  std::vector<float> now(t.cols), expected(t.cols);
  qfn::cpu::DequantizeRow(t, 0, 0, now.data());
  gufo::quant::DequantizeQ3_K(bytes.data(), expected.data(), t.cols);
  Require(std::any_of(now.begin(), now.end(), [](float v) { return v != 0; }),
          "Q3_K row decoded to zeros");
  Require(
      std::memcmp(now.data(), expected.data(), now.size() * sizeof(float)) == 0,
      "Q3_K DequantizeRow differs from DequantizeQ3_K");
  t.type = Type::kQ4_0;
  bool threw = false;
  try {
    qfn::cpu::DequantizeRow(t, 0, 0, now.data());
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find("type=Q4_0") != std::string::npos;
  }
  Require(threw, "unknown type must throw an error naming the type");
  std::puts("Q3_K DequantizeRow equals DequantizeQ3_K; unknown type throws");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    CheckDenseTypeTables();
    CheckQ3KRow();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  if (argc < 2) {
    std::puts("usage: native_cpu_test [--mtp sidecar.gguf] <raw-shard-1.gguf> [...] (CPU only)");
    std::puts("no model given: type-table checks only");
    return 0;
  }
  try {
    bool same_bytes = true;
    float max_error = 0;
    std::set<Type> compared;
    std::unique_ptr<gufo::core::GgufReader> mtp;
    int first = 1;
    if (std::string_view(argv[1]) == "--mtp" && argc > 3) {
      std::string error;
      mtp = gufo::core::GgufReader::OpenFile(argv[2], &error);
      Require(mtp != nullptr, error);
      first = 3;
    }
    Require(!qfn::iq_decode::ActiveExpert(-1) &&
                !qfn::iq_decode::ActiveExpert(-2147483647) &&
                qfn::iq_decode::ActiveExpert(0) && qfn::iq_decode::ActiveExpert(511),
            "negative expert guard changed");
    for (int arg = first; arg < argc; ++arg) {
      std::string error;
      auto reader = gufo::core::GgufReader::OpenFile(argv[arg], &error);
      Require(reader != nullptr, error);
      auto model = qfn::ModelWeights::Bind(*reader, &error);
      Require(model.has_value(), error);
      const bool native = std::any_of(model->layers.begin(), model->layers.end(),
          [](const qfn::LayerWeights& l) { return l.router.type == Type::kBF16; });
      if (mtp) {
        // The draft block binds with the sidecar's own config: the 512-expert
        // sidecar drafts for a pruned 256-expert trunk too.
        const auto bound = qfn::MtpWeights::Bind(*mtp, model->config, &error);
        Require(bound.has_value(), error);
        Require(bound->config.num_experts == bound->block.ffn_gate_exps.experts,
                "the draft block keeps its own expert count");
      }
      std::set<std::string> paths;
      std::size_t weight_bytes = 0;
      for (const auto& t : reader->GetTensors()) {
        if (t.name != "per_layer_token_embd.weight")
          weight_bytes += t.size_bytes;
        auto role = t.name;
        if (role.starts_with("blk."))
          role.remove_prefix(role.find('.', 4) + 1);
        const std::string label = std::string(role) + " " +
                                  std::string(gufo::core::ToString(t.type));
        const bool stacked = role == "ffn_gate_inp.weight" ||
                             role == "ffn_gate_inp_shexp.weight" ||
                             role == "ssm_alpha.weight" || role == "ssm_beta.weight";
        const auto device_type = stacked ? qfn::StackedProjectionType(t.type, native) : t.type;
        if (role == "per_layer_token_embd.weight") {
          paths.insert(label + " ngram=disk-gather");
        } else if (role == "token_embd.weight") {
          paths.insert(label + " trunk/draft=native-embedding-gather");
        } else if (role.ends_with("_exps.weight")) {
          paths.insert(label + " D/V=routed-mmvq P=routed-wmma/mmq");
        } else if (role.find("norm") != std::string_view::npos ||
                   role == "ssm_a" || role == "ssm_dt.bias" ||
                   role == "ssm_conv1d.weight") {
          paths.insert(label + " direct-f32-consumer");
        } else if (role == "ple_conv1d.weight") {
          paths.insert(label + " D/V/P=ple-conv-typed");
        } else if (const char* kernel = qfn::NativeDenseKernel(device_type, false)) {
          paths.insert(label + " device=" + std::string(gufo::core::ToString(device_type)) +
                       " D/V=" + kernel + " P=" + qfn::NativeDenseKernel(device_type, true));
        } else {
          throw std::runtime_error("tensor " + std::string(t.name) + " type=" +
                                   std::string(gufo::core::ToString(t.type)) +
                                   " pass=decode/verify/prefill: no native route");
        }
        if (t.type != Type::kIQ3_S && t.type != Type::kIQ4_XS && t.type != Type::kQ6_K)
          continue;
        if (compared.contains(t.type) || t.dimensions.size() != 2)
          continue;
        qfn::TensorRef row;
        row.data = t.data;
        row.name = t.name;
        row.type = t.type;
        row.cols = t.dimensions[0];
        row.rows = t.dimensions[1];
        CompareRow(row, same_bytes, max_error);
        compared.insert(t.type);
      }
      std::printf("bound %s tensors=%llu layers=%u experts=%u (header/config/binder)\n",
                  argv[arg], static_cast<unsigned long long>(reader->GetTensorCount()),
                  model->config.num_layers, model->config.num_experts);
      std::printf("encoded_resident_weight_bytes=%zu mtp=%s\n", weight_bytes,
                  mtp && model->config.num_experts == 512 ? "admitted" : "none");
      for (const auto& path : paths)
        std::printf("route %s\n", path.c_str());
    }
    Require(!compared.empty(), "no real native gather rows compared");
    Require(max_error == 0, "native gather differs from reference dequantization");
    Require(same_bytes, "old/new real-weight output bytes differ");
    std::printf("CPU reference dequant+dot: equal, max_abs_error=%.9g, types=%zu\n"
                "old/new real-weight output bytes: equal\n"
                "IQ1_M/IQ2_XXS/IQ2_S/Q2_0 shared entry-point negative-id guard: skip\n",
                max_error, compared.size());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
