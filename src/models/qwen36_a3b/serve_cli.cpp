// gufo serve for Qwen3.6-35B-A3B (see serve_cli.hpp).
#include "src/models/qwen36_a3b/serve_cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen36_a3b/serve.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

constexpr const char* kDraftVocab = "C:/custom_llama/draft-vocab-ascii.txt";

/// on / off / auto -> true / false / unset; false on a bad value.
bool ParseMode(const std::string& value, std::optional<bool>* out) {
  if (value == "on") *out = true;
  else if (value == "off") *out = false;
  else if (value == "auto") out->reset();
  else return false;
  return true;
}

}  // namespace

bool IsA3bServeModel(std::span<const char* const> args) {
  std::string model;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if ((a == "-m" || a == "--model") && i + 1 < args.size()) {
      model = args[i + 1];
    } else if (a.starts_with("--model=")) {
      model = std::string(a.substr(8));
    }
  }
  if (model.empty() || !std::filesystem::is_regular_file(model)) return false;
  const auto reader = core::GgufReader::OpenFile(model);
  if (!reader) return false;
  const auto arch = reader->GetMetadataString("general.architecture");
  return arch.has_value() && *arch == "qwen35moe";
}

int RunGufoServe(std::span<const char* const> args) {
  (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)std::setvbuf(stderr, nullptr, _IONBF, 0);
  ServeOptions so;
  // gufo serve's host / port environment defaults.
  for (const char* key : {"HOST", "GUFO_HOST"})
    if (const char* v = std::getenv(key); v != nullptr && *v != '\0') {
      so.host = v;
      break;
    }
  for (const char* key : {"PORT", "GUFO_PORT"})
    if (const char* v = std::getenv(key); v != nullptr && *v != '\0') {
      so.port = std::atoi(v);
      break;
    }
  so.sampling.temperature = 1.0F;  // Qwen3.6's thinking-mode recommendation
  so.sampling.top_k = 20;
  so.sampling.top_p = 0.95F;

  std::string model;
  std::size_t context = 131072;
  std::size_t max_tokens = so.max_tokens;
  std::string think = "auto", preserve = "auto", speculative = "mtp";
  std::string dflash_model, draft_vocab = "auto";
  int draft_tokens = -1, lookup = 12;
  double survival = -1.0;
  bool verbose = false;
  std::string ignored;  // gufo serve flags without an A3B meaning

  cli::ArgParser p("gufo serve (Qwen3.6-35B-A3B)",
                   "OpenAI-compatible server on the A3B engine.");
  p.AddOption("", "--host", "HOST", "Bind address", "Server", &so.host);
  p.AddOption("", "--port", "PORT", "Port", "Server", &so.port);
  p.AddOption("", "--api-key", "KEY", "Require Bearer authorization",
              "Server", &so.api_key);
  p.AddOption("", "--max-connections", "N", "HTTP connections", "Server",
              &so.max_connections);
  p.AddOption("", "--max-request-bytes", "N", "Request body limit", "Server",
              &so.max_request_bytes);
  p.AddFlag("-v", "--verbose", "Verbose (no effect)", "Server", &verbose);
  p.AddOption("-m", "--model", "PATH", "Qwen3.6-35B-A3B GGUF", "Model", &model);
  p.AddOption("", "--served-model-name", "ID", "Model id in the API", "Model",
              &so.model_id);
  p.AddOption("-c", "--context", "N", "Context tokens (default 131072)",
              "Model", &context);
  p.AddOption("-n", "--max-tokens", "N", "Default max new tokens (32768)",
              "Sampling Defaults", &max_tokens);
  cli::RegisterSamplingOptions(p, &so.sampling, "Sampling Defaults");
  p.AddOption("", "--think", "MODE", "Default reasoning: on, off, auto",
              "Reasoning Defaults", &think);
  p.AddOption("", "--preserve-thinking", "MODE",
              "Replay prior reasoning: on, off, auto (default off)",
              "Reasoning Defaults", &preserve);
  p.AddOption("", "--speculative", "MODE", "mtp (default), dflash2 or off",
              "Speculative", &speculative);
  p.AddOption("", "--dflash-model", "PATH", "DFlash2 draft GGUF",
              "Speculative", &dflash_model);
  p.AddOption("-d", "--draft-tokens", "N",
              "Max drafts per step (MTP 6, DFlash2 7)", "Speculative",
              &draft_tokens);
  p.AddOption("", "--survival", "F",
              "Stop drafting below this survival (MTP 0.6, DFlash2 0.2)",
              "Speculative", &survival);
  p.AddOption("", "--lookup", "N",
              "Prompt lookup min match (default 12, 0 = off)", "Speculative",
              &lookup);
  p.AddOption("", "--draft-vocab", "PATH",
              "Draft head token list, or off (default: the ASCII list)",
              "Speculative", &draft_vocab);
  for (const char* flag :
       {"--sessions", "--mmproj", "--reasoning-effort", "--draft-policy",
        "--dspark-model", "--mtp-model", "--min-draft-tokens", "--prefill-chunk",
        "--max-pending", "--max-pending-per-client", "--request-timeout-ms",
        "--max-output-bytes", "--max-buffered-output-bytes",
        "--max-buffered-output-total", "--cache-disk", "--cache-disk-bytes",
        "--cache-disk-staging-bytes"})
    p.AddOption("", flag, "X", "Accepted and ignored (no A3B meaning)",
                "Ignored", &ignored);
  p.SetPositionalHandler([](std::string_view arg, std::string* error) {
    if (arg == "llm") return true;  // gufo serve llm ...
    if (error != nullptr) *error = "Unexpected argument: " + std::string(arg);
    return false;
  });
  std::string error;
  if (!p.Parse(args, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 2;
  }
  if (p.IsHelpRequested()) {
    p.PrintHelp();
    return 0;
  }
  std::optional<bool> think_mode, preserve_mode;
  if (!ParseMode(think, &think_mode) || !ParseMode(preserve, &preserve_mode)) {
    std::cerr << "Error: --think / --preserve-thinking take on, off or auto\n";
    return 2;
  }
  so.think = think_mode;
  so.preserve_thinking = preserve_mode;
  so.max_tokens = max_tokens;
  so.lookup = lookup;
  if (speculative == "dflash2" || speculative == "dflash") {
    if (dflash_model.empty()) {
      std::cerr << "Error: --speculative dflash2 needs --dflash-model\n";
      return 2;
    }
    so.mtp = 0;
    so.dflash_n = draft_tokens > 0 ? draft_tokens : 7;
    so.survival = survival >= 0 ? survival : 0.2;
  } else if (speculative == "mtp") {
    so.mtp = draft_tokens >= 0 ? draft_tokens : 6;
    so.survival = survival >= 0 ? survival : 0.6;
  } else if (speculative == "off") {
    so.mtp = 0;
  } else {
    std::cerr << "Error: --speculative must be mtp, dflash2 or off\n";
    return 2;
  }

  Options options;
  options.max_context = static_cast<std::uint32_t>(context);
  if (draft_vocab == "auto") {
    if (std::filesystem::is_regular_file(kDraftVocab))
      options.draft_vocab = kDraftVocab;
  } else if (draft_vocab != "off") {
    options.draft_vocab = draft_vocab;
  }
  auto engine = Engine::Load(model, options, &error);
  if (!engine) {
    std::cerr << "Error loading model '" << model << "': " << error << "\n";
    return 1;
  }
  if (so.dflash_n > 0 && !engine->LoadDFlash(dflash_model, &error)) {
    std::cerr << "Error loading DFlash2 draft: " << error << "\n";
    return 1;
  }
  auto tok = tokenization::QwenTokenizer::CreateFromGguf(engine->gguf(), &error);
  if (!tok) {
    std::cerr << "Error: tokenizer: " << error << "\n";
    return 1;
  }
  std::fprintf(stderr,
               "a3b: gufo serve: %s drafts %d, survival %.2f, lookup %d, draft "
               "vocab %s; sampling temp %.2f top-k %d top-p %.2f\n",
               so.dflash_n > 0 ? "dflash2" : so.mtp > 0 ? "mtp" : "off",
               so.dflash_n > 0 ? so.dflash_n : so.mtp, so.survival, so.lookup,
               options.draft_vocab.empty() ? "off" : "on",
               so.sampling.temperature, so.sampling.top_k, so.sampling.top_p);
  return RunServe(*engine, *tok, so);
}

}  // namespace gufo::models::qwen36_a3b
