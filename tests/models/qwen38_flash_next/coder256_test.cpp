// Failure cases and layout evidence: decode-coder256/NOTES.md, recorded
// before this test. Synthetic bytes only; no model files or GPU APIs.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/reference.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Spec {
  std::string name;
  std::vector<std::uint64_t> shape;
  GgmlType type;
  unsigned block{32}, block_bytes{34};
};
struct Fixture {
  qfn::Config config;
  std::vector<Spec> specs;
  std::vector<std::uint8_t> bytes;
  std::unique_ptr<gufo::core::GgufReader> reader;

  explicit Fixture(unsigned experts) {
    config.hidden_size = config.expert_ff = config.shared_expert_ff = 32;
    config.num_experts = experts;
    config.num_experts_used = 10;
    specs = {
        {"blk.0.ffn_gate_inp.weight", {32, experts}, GgmlType::kF32},
        {"blk.0.ffn_gate_exps.weight", {32, 32, experts}, GgmlType::kQ8_0},
        {"blk.0.ffn_up_exps.weight", {32, 32, experts}, GgmlType::kQ8_0},
        {"blk.0.ffn_down_exps.weight", {32, 32, experts}, GgmlType::kQ8_0}};
  }
  template<class T>
  void Put(T value) {
    const auto offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  }
  void Text(const std::string& text) {
    Put(std::uint64_t(text.size()));
    bytes.insert(bytes.end(), text.begin(), text.end());
  }
  std::size_t Size(const Spec& s) {
    const auto count = std::accumulate(s.shape.begin(), s.shape.end(),
                                       std::uint64_t{1}, std::multiplies<>());
    return s.type == GgmlType::kF32 ? count * 4
                                    : count / s.block * s.block_bytes;
  }
  void Open() {
    bytes.clear();
    Put(std::uint32_t{0x46554747});
    Put(std::uint32_t{3});
    Put(std::uint64_t(specs.size()));
    Put(std::uint64_t{0});
    std::size_t offset = 0;
    for (const auto& s : specs) {
      Text(s.name);
      Put(std::uint32_t(s.shape.size()));
      for (auto d : s.shape)
        Put(d);
      Put(static_cast<std::uint32_t>(s.type));
      Put(std::uint64_t(offset));
      offset += (Size(s) + 31) / 32 * 32;
    }
    bytes.resize((bytes.size() + 31) / 32 * 32 + offset);
    std::string error;
    reader =
        gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
    Require(reader != nullptr, error.c_str());
  }
  std::optional<qfn::RoutedWeights> Bind(std::string& error) {
    error.clear();
    return qfn::RoutedWeights::Bind(*reader, config, 0, &error);
  }
};

void CheckBinder() {
  std::size_t previous_bytes = 0;
  for (unsigned experts : {256U, 512U}) {
    Fixture f(experts);
    f.Open();
    const auto unchanged = f.bytes;
    std::string error;
    auto w = f.Bind(error);
    Require(w && error.empty(), "compact bind failed");
    Require(w->router.rows == experts && w->gate.experts == experts &&
                w->up.experts == experts && w->down.experts == experts,
            "bound extent changed");
    const auto total =
        w->gate.SizeBytes() + w->up.SizeBytes() + w->down.SizeBytes();
    Require(total == std::size_t(experts) * 3 * 32 * 34,
            "expert resident bytes must follow descriptor extent");
    if (previous_bytes)
      Require(total == previous_bytes * 2, "512/256 expert memory ratio");
    previous_bytes = total;
    Require(f.bytes == unchanged, "binding mutated fixture bytes");
    for (unsigned tensor = 0; tensor < 4; ++tensor) {
      Fixture bad(experts);
      bad.specs[tensor].shape.back() = experts == 256 ? 512 : 256;
      bad.Open();
      Require(!bad.Bind(error), "mismatched expert extent accepted");
      if (experts == 256 && tensor == 0)
        Require(error.find("512-row masked router") != std::string::npos,
                "masked-router case must be named");
    }
    for (unsigned tensor = 0; tensor < 4; ++tensor) {
      Fixture bad(experts);
      bad.specs[tensor].shape[0] = 64;
      bad.Open();
      Require(!bad.Bind(error), "wrong reduction axis accepted");
      bad.specs[tensor].shape[0] = 32;
      bad.specs[tensor].shape.push_back(2);
      bad.Open();
      Require(!bad.Bind(error), "wrong tensor rank accepted");
    }
    Fixture missing(experts);
    missing.specs.pop_back();
    missing.Open();
    Require(!missing.Bind(error) &&
                error.find("missing tensor") != std::string::npos,
            "missing expert tensor accepted");
  }
}

// IQ1_M's enum/parser is introduced by iq-kernel, not this compact-layout
// patch. Once present, ALL three added admissions are mandatory: resolving
// the moved-list merge conflict by dropping any one must fail this matrix.
template<class Type = GgmlType>
void CheckAdmittedTypes() {
  struct Encoding {
    Type type;
    unsigned block, bytes;
  };
  std::vector<Encoding> types = {
      {Type::kQ4_K, 256, 144},  {Type::kQ5_K, 256, 176},
      {Type::kQ6_K, 256, 210},  {Type::kQ3_K, 256, 110},
      {Type::kQ5_1, 32, 24},    {Type::kQ8_0, 32, 34},
      {Type::kIQ3_S, 256, 110}, {Type::kIQ4_XS, 256, 136},
      {Type::kIQ4_NL, 32, 18},  {Type::kIQ3_XXS, 256, 98},
      {Type::kIQ2_S, 256, 82},  {Type::kQ2_0, 64, 18}};
  if constexpr (requires { Type::kIQ1_M; }) {
    types.push_back({Type::kIQ2_XXS, 256, 66});
    types.push_back({Type::kIQ2_XS, 256, 74});
    types.push_back({Type::kIQ1_M, 256, 56});
  }
  for (auto encoding : types) {
    for (unsigned experts : {256U, 512U}) {
      Fixture f(experts);
      f.config.hidden_size = f.config.expert_ff = 256;
      f.specs[0].shape[0] = 256;
      for (unsigned tensor = 1; tensor < 4; ++tensor) {
        f.specs[tensor] = {f.specs[tensor].name,
                           {256, 256, experts},
                           encoding.type,
                           encoding.block,
                           encoding.bytes};
      }
      f.Open();
      std::string error;
      const auto w = f.Bind(error);
      Require(w.has_value(), error.c_str());
      for (const auto* t : {&w->gate, &w->up, &w->down}) {
        Require(t->type == encoding.type &&
                    t->SizeBytes() == std::size_t(256) * 256 * experts /
                                          encoding.block * encoding.bytes,
                "expert type or encoded byte extent changed");
      }
    }
  }
  std::cout << types.size()
            << " admitted expert formats bound in all three "
               "projections at 256 and 512 experts\n";
}

// A type outside the expert list must fail naming tensor, type and pass.
void CheckUnsupportedExpertType() {
  Fixture f(256);
  f.config.hidden_size = f.config.expert_ff = 256;
  f.specs[0].shape[0] = 256;
  for (unsigned tensor = 1; tensor < 4; ++tensor)
    f.specs[tensor] = {
        f.specs[tensor].name, {256, 256, 256}, GgmlType::kQ4_0, 32, 18};
  f.Open();
  std::string error;
  Require(!f.Bind(error), "Q4_0 experts accepted");
  Require(error.find("blk.0.ffn_gate_exps.weight") != std::string::npos &&
              error.find("type=Q4_0") != std::string::npos &&
              error.find("pass=") != std::string::npos,
          error.c_str());
  std::cout << "unsupported expert type rejected: " << error << '\n';
}

std::vector<unsigned> Brute(std::span<const float> logits, unsigned e) {
  std::vector<unsigned> ids(e);
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  });
  ids.resize(10);
  return ids;
}
// F32 storage for the tiny graph's non-expert operators. deque retains data
// addresses while more tensors are appended.
struct Dense {
  std::deque<std::vector<float>> data;
  qfn::TensorRef Add(unsigned cols, unsigned rows, float value = 0) {
    auto& v = data.emplace_back(cols * rows, value);
    qfn::TensorRef t;
    t.data = v.data();
    t.cols = cols;
    t.rows = rows;
    return t;
  }
  qfn::HcMixer Mixer() {
    return {Add(32, 1, 1), Add(32, 1), Add(1, 32), Add(32, 1)};
  }
};
void CheckReference(unsigned experts) {
  Fixture f(experts);
  f.Open();
  std::string error;
  auto bound = f.Bind(error);
  Require(bound.has_value(), "reference fixture bind");
  auto* router =
      const_cast<float*>(static_cast<const float*>(bound->router.data));
  std::vector<float> logits(experts);
  for (unsigned e = 0; e < experts; ++e) {
    // Distinct logits, shuffled stack order; highest index must be selected.
    logits[e] = float((e * 73) % experts) / experts;
    router[e * 32] = logits[e];
  }
  router[(experts - 1) * 32] = logits.back() = 2;
  for (const auto* t : {&bound->gate, &bound->up, &bound->down}) {
    auto* bytes =
        const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(t->data));
    for (unsigned e = 0; e < experts; ++e) {
      for (unsigned r = 0; r < 32; ++r) {
        auto* row = bytes + (e * 32 + r) * 34;
        row[0] = 0;
        row[1] = 0x3c;  // half scale 1
        row[2] = t == &bound->down ? (e % 97) + 1 : (r == 0 ? 1 : 0);
      }
    }
  }
  qfn::ModelWeights w;
  w.config = f.config;
  auto& c = w.config;
  c.num_layers = c.num_layers_all = c.full_attention_interval = 1;
  c.hc_count = c.hc_low_rank = c.num_heads = c.num_kv_heads = c.indexer_heads =
      1;
  c.head_dim = c.rotary_dim = c.indexer_head_dim = 2;
  c.context_length = 4;
  c.vocab_size = 1;
  c.compress_ratio = 4;
  c.indexer_top_k = 4;
  c.ple_conv_kernel = 1;
  c.rope_theta = 10000;
  Dense d;
  w.token_embd = d.Add(32, 1, 1);
  w.output = d.Add(32, 1);
  w.hc_head = d.Mixer();
  auto& l = w.layers.emplace_back();
  l.router = bound->router;
  l.ffn_gate_exps = bound->gate;
  l.ffn_up_exps = bound->up;
  l.ffn_down_exps = bound->down;
  l.hc_attn = d.Mixer();
  l.hc_ffn = d.Mixer();
  l.attn_q = d.Add(32, 4);
  l.attn_k = d.Add(32, 2);
  l.attn_v = d.Add(32, 2);
  l.attn_out = d.Add(2, 32);
  l.attn_q_norm = d.Add(2, 1, 1);
  l.attn_k_norm = d.Add(2, 1, 1);
  l.indexer_q = d.Add(32, 2);
  l.indexer_k = d.Add(32, 2);
  l.indexer_q_norm = d.Add(2, 1, 1);
  l.indexer_k_norm = d.Add(2, 1, 1);
  l.shexp_gate_inp = d.Add(32, 1);
  l.shexp_gate = d.Add(32, 32);
  l.shexp_up = d.Add(32, 32);
  l.shexp_down = d.Add(32, 32);
  qfn::ReferenceModel model(w, nullptr, 4);
  auto expected_ids = Brute(logits, experts);
  std::vector<float> residual(32);
  Require(model.Step(0, {}, residual, &error), error.c_str());
  // Zero attention, identity residual, zero mixer logits: MoE input =
  // .5 * RMSNorm(ones). Independent closed form for the full one-layer Step.
  const double x = .5 / std::sqrt(1.0 + c.rms_eps);
  double denom = 0, sum = 0;
  for (auto e : expected_ids) {
    const double p = std::exp(double(logits[e]) * x);
    denom += p;
    sum += p * ((e % 97) + 1);
  }
  const double forward = 1 + x * x / (1 + std::exp(-x)) * sum / denom;
  for (float value : residual)
    Require(std::abs(value - forward) < 3e-5,
            "one-layer scalar forward mismatch");
  std::cout << experts
            << " experts: compact binder and one-layer Step passed\n";
}
}  // namespace
int main() {
  try {
    CheckBinder();
    CheckAdmittedTypes();
    CheckUnsupportedExpertType();
    CheckReference(256);
    CheckReference(512);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
