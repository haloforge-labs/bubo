#include "src/models/diffusion_gemma/config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/core/json.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

json::Value ReadJson(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return json::parse(text.str());
}

const json::Value& Member(const json::Value& object, const char* key) {
  const auto* value = object.find(key);
  if (value == nullptr)
    throw std::runtime_error(std::string("config.json is missing ") + key);
  return *value;
}

void Expect(bool condition, const std::string& what) {
  if (!condition)
    throw std::runtime_error("unsupported DiffusionGemma config: " + what);
}

}  // namespace

bool IsDiffusionGemmaDirectory(const std::filesystem::path& root) {
  std::error_code ec;
  if (!std::filesystem::is_directory(root, ec) ||
      !std::filesystem::is_regular_file(root / "config.json", ec))
    return false;
  try {
    const auto config = ReadJson(root / "config.json");
    const auto* type = config.find("model_type");
    return type != nullptr && type->get_str() == "diffusion_gemma";
  } catch (const std::exception&) {
    return false;
  }
}

Config LoadConfig(const std::filesystem::path& root) {
  const auto top = ReadJson(root / "config.json");
  Expect(Member(top, "model_type").get_str() == "diffusion_gemma",
         "model_type is not diffusion_gemma");
  const auto& text = Member(top, "text_config");
  Config c;
  c.hidden = static_cast<int>(Member(text, "hidden_size").as_size());
  c.layers = static_cast<int>(Member(text, "num_hidden_layers").as_size());
  c.heads = static_cast<int>(Member(text, "num_attention_heads").as_size());
  c.sliding_head_dim = static_cast<int>(Member(text, "head_dim").as_size());
  c.sliding_kv_heads =
      static_cast<int>(Member(text, "num_key_value_heads").as_size());
  c.global_head_dim =
      static_cast<int>(Member(text, "global_head_dim").as_size());
  c.global_kv_heads =
      static_cast<int>(Member(text, "num_global_key_value_heads").as_size());
  c.intermediate =
      static_cast<int>(Member(text, "intermediate_size").as_size());
  c.experts = static_cast<int>(Member(text, "num_experts").as_size());
  c.top_k = static_cast<int>(Member(text, "top_k_experts").as_size());
  c.expert_intermediate =
      static_cast<int>(Member(text, "moe_intermediate_size").as_size());
  c.vocab = static_cast<int>(Member(text, "vocab_size").as_size());
  c.sliding_window = static_cast<int>(Member(text, "sliding_window").as_size());
  c.max_positions =
      static_cast<int>(Member(text, "max_position_embeddings").as_size());
  c.rms_eps = static_cast<float>(Member(text, "rms_norm_eps").as_double());
  c.final_softcap =
      static_cast<float>(Member(text, "final_logit_softcapping").as_double());
  c.canvas = static_cast<int>(Member(top, "canvas_length").as_size());
  c.pad_id = static_cast<int>(Member(text, "pad_token_id").as_size());
  c.bos_id = static_cast<int>(Member(text, "bos_token_id").as_size());

  const auto& types = Member(text, "layer_types");
  Expect(static_cast<int>(types.size()) == c.layers, "layer_types length");
  for (std::size_t i = 0; i < types.size(); ++i) {
    const auto kind = types.items()[i].get_str();
    Expect(kind == "sliding_attention" || kind == "full_attention",
           "layer type " + kind);
    c.is_global.push_back(kind == "full_attention");
  }
  const auto& rope = Member(text, "rope_parameters");
  const auto& sliding = Member(rope, "sliding_attention");
  const auto& global = Member(rope, "full_attention");
  Expect(Member(sliding, "rope_type").get_str() == "default",
         "sliding rope type");
  Expect(Member(global, "rope_type").get_str() == "proportional",
         "global rope type");
  c.sliding_theta =
      static_cast<float>(Member(sliding, "rope_theta").as_double());
  c.global_theta = static_cast<float>(Member(global, "rope_theta").as_double());
  c.global_rotary_fraction =
      static_cast<float>(Member(global, "partial_rotary_factor").as_double());

  // The kernels are specialised for exactly this model.
  Expect(c.hidden == 2816 && c.heads == 16 && c.sliding_head_dim == 256 &&
             c.global_head_dim == 512 && c.sliding_kv_heads == 8 &&
             c.global_kv_heads == 2,
         "attention shape");
  Expect(c.intermediate == 2112 && c.expert_intermediate == 704 &&
             c.experts == 128 && c.top_k == 8,
         "feed-forward shape");
  Expect(c.canvas == 256, "canvas length");
  Expect(c.vocab == 262144, "vocabulary size");

  if (std::filesystem::exists(root / "generation_config.json")) {
    const auto generation = ReadJson(root / "generation_config.json");
    if (const auto* eos = generation.find("eos_token_id")) {
      for (std::size_t i = 0; i < eos->size(); ++i)
        c.eos_ids.push_back(static_cast<int>(eos->items()[i].as_size()));
    }
  }
  if (c.eos_ids.empty())
    c.eos_ids = {1, 106};
  return c;
}

DiffusionDefaults LoadDiffusionDefaults(const std::filesystem::path& root) {
  DiffusionDefaults d;
  const auto path = root / "generation_config.json";
  if (!std::filesystem::exists(path))
    return d;
  const auto g = ReadJson(path);
  if (const auto* v = g.find("max_denoising_steps"))
    d.max_denoising_steps = static_cast<int>(v->as_size());
  if (const auto* v = g.find("t_min"))
    d.t_min = static_cast<float>(v->as_double());
  if (const auto* v = g.find("t_max"))
    d.t_max = static_cast<float>(v->as_double());
  if (const auto* v = g.find("stability_threshold"))
    d.stability_threshold = static_cast<int>(v->as_size());
  if (const auto* v = g.find("confidence_threshold"))
    d.confidence_threshold = static_cast<float>(v->as_double());
  if (const auto* s = g.find("sampler_config")) {
    if (const auto* v = s->find("entropy_bound"))
      d.entropy_bound = static_cast<float>(v->as_double());
  }
  return d;
}

}  // namespace gufo::models::diffusion_gemma
