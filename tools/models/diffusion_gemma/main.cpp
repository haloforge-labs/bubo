// gufo-diffusion-gemma: tokenizer and chat-template checks, one-shot
// generation, the prefill/generation context sweep and reference logits for
// the DiffusionGemma engine.
//
//   gufo-diffusion-gemma tokenize --model DIR [--text TEXT]
//   gufo-diffusion-gemma render --model DIR --file REQUEST.json
//   gufo-diffusion-gemma generate --model DIR --prompt TEXT [--max-tokens N]
//                                 [--seed S] [--runs N]
//   gufo-diffusion-gemma bench --model DIR --corpus FILE
//                              [--contexts 1024,4096,...] [--gen-tokens 512]
//                              [--reps 3] [--json OUT]
//   gufo-diffusion-gemma logits --model DIR --prompt-ids F --canvas-ids F
//                               --canvas2-ids F --out FILE
//   gufo-diffusion-gemma throughput --model DIR --corpus FILE
//                                   [--context 2048] [--sessions 4]
//                                   [--requests 16] [--gen-tokens 1024]
//
// Engine commands also take [--max-context N] [--chunk N] [--sessions N]
// and [--profile] (per-phase GPU time on stderr).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/diffusion_gemma/chat.hpp"
#include "src/models/diffusion_gemma/engine.hpp"
#include "src/models/diffusion_gemma/tokenizer.hpp"

namespace dg = gufo::models::diffusion_gemma;

namespace {

std::map<std::string, std::string> ParseFlags(int argc, char** argv) {
  std::map<std::string, std::string> flags;
  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0)
      throw std::invalid_argument("unexpected argument " + key);
    key = key.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
      flags[key] = argv[++i];
    else
      flags[key] = "1";
  }
  return flags;
}

std::string Flag(const std::map<std::string, std::string>& flags,
                 const std::string& key, const std::string& fallback = "") {
  const auto found = flags.find(key);
  if (found != flags.end())
    return found->second;
  if (fallback.empty())
    throw std::invalid_argument("--" + key + " is required");
  return fallback;
}

std::vector<int> ParseList(const std::string& text) {
  std::vector<int> values;
  std::stringstream in(text);
  std::string item;
  while (std::getline(in, item, ','))
    values.push_back(std::stoi(item));
  return values;
}

std::string ReadText(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// Renders an OpenAI-style {"messages": [...], "tools": [...]} request the
// way the server does, for comparison with the reference chat template.
int Render(const std::map<std::string, std::string>& flags) {
  const auto request = gufo::json::parse(ReadText(Flag(flags, "file")));
  std::vector<dg::Message> messages;
  for (const auto& m : request.find("messages")->items()) {
    dg::Message message;
    message.role = m.find("role")->str();
    if (const auto* content = m.find("content");
        content && content->is_string())
      message.content = content->str();
    if (const auto* id = m.find("tool_call_id"))
      message.tool_call_id = id->str();
    if (const auto* name = m.find("name"))
      message.name = name->str();
    if (const auto* calls = m.find("tool_calls")) {
      for (const auto& call : calls->items()) {
        dg::ToolCall out;
        if (const auto* id = call.find("id"))
          out.id = id->str();
        const auto* function = call.find("function");
        out.name = function->find("name")->str();
        const auto* arguments = function->find("arguments");
        out.arguments = arguments->is_string()
                            ? gufo::json::parse(arguments->str())
                            : *arguments;
        message.tool_calls.push_back(std::move(out));
      }
    }
    messages.push_back(std::move(message));
  }
  std::vector<gufo::json::Value> tools;
  if (const auto* list = request.find("tools"))
    tools = list->items();
  std::cout << dg::RenderChat(messages, tools, flags.count("thinking") > 0);
  return 0;
}

int Tokenize(const std::map<std::string, std::string>& flags) {
  dg::Tokenizer tokenizer(Flag(flags, "model"));
  std::string text = Flag(flags, "text", "\x01");
  if (flags.count("file"))
    text = ReadText(flags.at("file"));
  const auto ids = tokenizer.Encode(text);
  for (std::size_t i = 0; i < ids.size(); ++i)
    std::cout << (i ? " " : "") << ids[i];
  std::cout << "\n";
  return 0;
}

dg::EngineOptions EngineOptionsFrom(
    const std::map<std::string, std::string>& flags) {
  dg::EngineOptions options;
  options.max_context = std::stoi(Flag(flags, "max-context", "131072"));
  options.prefill_chunk = std::stoi(
      Flag(flags, "chunk", std::to_string(dg::EngineOptions{}.prefill_chunk)));
  options.sessions = std::stoi(Flag(flags, "sessions", "1"));
  options.profile = flags.contains("profile");
  return options;
}

int Generate(const std::map<std::string, std::string>& flags) {
  const auto load_start = std::chrono::steady_clock::now();
  dg::Engine engine(Flag(flags, "model"), EngineOptionsFrom(flags));
  std::cerr << "loaded " << engine.weight_bytes() / (1 << 20) << " MiB in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_start)
                   .count()
            << " s\n";
  const std::string user = flags.count("prompt-file")
                               ? ReadText(flags.at("prompt-file"))
                               : Flag(flags, "prompt");
  dg::Message message;
  message.content = user;
  const auto prompt =
      engine.tokenizer().Encode(dg::RenderChat({message}, {}, false));
  dg::GenerationOptions options;
  options.diffusion = engine.diffusion_defaults();
  options.max_new_tokens = std::stoi(Flag(flags, "max-tokens", "256"));
  options.seed = std::stoull(Flag(flags, "seed", "1"));
  if (!flags.count("no-warmup"))
    std::cerr << "warmup " << engine.Warmup() / 1000.0 << " s\n";
  if (flags.count("steps"))
    options.diffusion.max_denoising_steps = std::stoi(flags.at("steps"));
  const int runs = std::stoi(Flag(flags, "runs", "1"));
  for (int run = 0; run < runs; ++run) {
    std::vector<std::int32_t> output;
    const auto stats = engine.Generate(
        prompt, options, [&](std::span<const std::int32_t> tokens) {
          output.insert(output.end(), tokens.begin(), tokens.end());
          return true;
        });
    const auto& tokenizer = engine.tokenizer();
    const auto visible =
        dg::StripThinkingTokens(output, tokenizer.TokenId("<|channel>"),
                                tokenizer.TokenId("<channel|>"));
    std::cout << tokenizer.Decode(visible, true) << "\n";
    std::fprintf(stderr,
                 "prompt=%d generated=%d blocks=%d steps=%d prefill_ms=%.1f "
                 "denoise_ms=%.1f commit_ms=%.1f first_block_ms=%.1f "
                 "ms_per_step=%.1f eos=%d\n",
                 stats.prompt_tokens, stats.generated_tokens, stats.blocks,
                 stats.denoising_steps, stats.prefill_ms, stats.denoise_ms,
                 stats.commit_ms, stats.first_block_ms,
                 stats.denoise_ms / std::max(1, stats.denoising_steps),
                 stats.stopped_by_eos ? 1 : 0);
  }
  return 0;
}

// Prompt of exactly `context` tokens: a user turn holding the corpus from
// `offset` and a continuation instruction, then the open model turn.
std::vector<std::int32_t> ContextPrompt(const dg::Tokenizer& tokenizer,
                                        const std::vector<std::int32_t>& corpus,
                                        int context, std::size_t offset = 0) {
  const auto head = tokenizer.Encode("<bos><|turn>user\n");
  const auto tail = tokenizer.Encode(
      "\n\nContinue the story above in the same style. Write at least 800 "
      "words of new narrative prose; do not summarise.<turn|>\n<|turn>model\n");
  const int body = context - static_cast<int>(head.size() + tail.size());
  if (body <= 0 || offset + body > corpus.size())
    throw std::invalid_argument("corpus too short for context " +
                                std::to_string(context));
  std::vector<std::int32_t> prompt = head;
  prompt.insert(prompt.end(), corpus.begin() + offset,
                corpus.begin() + offset + body);
  prompt.insert(prompt.end(), tail.begin(), tail.end());
  return prompt;
}

std::vector<std::int32_t> ReadIds(const std::string& path) {
  std::vector<std::int32_t> ids;
  std::istringstream in(ReadText(path));
  std::int32_t id;
  while (in >> id)
    ids.push_back(id);
  return ids;
}

// Writes both denoiser steps' softcapped logits for a reference comparison.
int DumpLogits(const std::map<std::string, std::string>& flags) {
  dg::Engine engine(Flag(flags, "model"), EngineOptionsFrom(flags));
  const auto prompt = ReadIds(Flag(flags, "prompt-ids"));
  const auto canvas = ReadIds(Flag(flags, "canvas-ids"));
  const auto canvas2 = ReadIds(Flag(flags, "canvas2-ids"));
  const auto logits = engine.DebugLogits(prompt, canvas, canvas2);
  std::ofstream out(Flag(flags, "out"), std::ios::binary);
  out.write(reinterpret_cast<const char*>(logits.data()),
            static_cast<std::streamsize>(logits.size() * sizeof(float)));
  return 0;
}

int Bench(const std::map<std::string, std::string>& flags) {
  const auto contexts =
      ParseList(Flag(flags, "contexts", "512,1024,2048,4096,8192,16384,32768"));
  const int gen_tokens = std::stoi(Flag(flags, "gen-tokens", "512"));
  const int reps = std::stoi(Flag(flags, "reps", "3"));
  const int max_context = *std::max_element(contexts.begin(), contexts.end());
  auto engine_options = EngineOptionsFrom(flags);
  engine_options.max_context =
      std::max(engine_options.max_context, max_context + gen_tokens + 256);
  dg::Engine engine(Flag(flags, "model"), engine_options);
  const auto& tokenizer = engine.tokenizer();
  const auto corpus = tokenizer.Encode(ReadText(Flag(flags, "corpus")));
  std::cerr << "corpus tokens: " << corpus.size() << "\n";
  const bool skip_generation = flags.count("prefill-only") > 0;

  std::cerr << "warmup " << engine.Warmup() / 1000.0 << " s\n";

  std::ostringstream json;
  json << "{\"gen_tokens\":" << gen_tokens << ",\"reps\":" << reps
       << ",\"chunk\":" << engine_options.prefill_chunk << ",\"results\":[";
  std::printf("%8s %12s %12s %10s %10s %10s %10s %10s %8s\n", "context",
              "prefill_tps", "prefill_s", "gen_tps", "denoise_tps", "steps/blk",
              "ms/step", "ttfb_s", "tokens");
  bool first_row = true;
  for (const int context : contexts) {
    const auto prompt = ContextPrompt(tokenizer, corpus, context);
    std::vector<double> prefill, gen_tps, denoise_tps, steps, ms_step, ttfb,
        produced;
    for (int rep = 0; rep < reps; ++rep) {
      if (skip_generation) {
        engine.Reset();
        prefill.push_back(engine.Encode(prompt));
        continue;
      }
      dg::GenerationOptions options;
      options.diffusion = engine.diffusion_defaults();
      options.max_new_tokens = gen_tokens;
      options.seed = 1000 + rep;
      engine.Reset();  // Measure the whole prompt, not a reused prefix.
      const auto stats = engine.Generate(prompt, options, {});
      prefill.push_back(stats.prefill_ms);
      const double decode_ms = stats.denoise_ms + stats.commit_ms;
      gen_tps.push_back(stats.generated_tokens * 1000.0 /
                        std::max(1.0, decode_ms));
      denoise_tps.push_back(stats.blocks * 256 * 1000.0 /
                            std::max(1.0, stats.denoise_ms));
      steps.push_back(static_cast<double>(stats.denoising_steps) /
                      std::max(1, stats.blocks));
      ms_step.push_back(stats.denoise_ms / std::max(1, stats.denoising_steps));
      ttfb.push_back(stats.first_block_ms / 1000.0);
      produced.push_back(stats.generated_tokens);
    }
    const double prefill_ms = Median(prefill);
    const auto med = [](const std::vector<double>& v) {
      return v.empty() ? 0.0 : Median(v);
    };
    std::printf("%8d %12.1f %12.3f %10.2f %10.2f %10.1f %10.1f %10.2f %8.0f\n",
                context, context * 1000.0 / prefill_ms, prefill_ms / 1000.0,
                med(gen_tps), med(denoise_tps), med(steps), med(ms_step),
                med(ttfb), med(produced));
    std::fflush(stdout);
    json << (first_row ? "" : ",") << "{\"context\":" << context
         << ",\"prefill_ms\":[";
    for (std::size_t i = 0; i < prefill.size(); ++i)
      json << (i ? "," : "") << prefill[i];
    json << "],\"gen_tps\":[";
    for (std::size_t i = 0; i < gen_tps.size(); ++i)
      json << (i ? "," : "") << gen_tps[i];
    json << "],\"steps_per_block\":[";
    for (std::size_t i = 0; i < steps.size(); ++i)
      json << (i ? "," : "") << steps[i];
    json << "],\"ms_per_step\":[";
    for (std::size_t i = 0; i < ms_step.size(); ++i)
      json << (i ? "," : "") << ms_step[i];
    json << "],\"first_block_s\":[";
    for (std::size_t i = 0; i < ttfb.size(); ++i)
      json << (i ? "," : "") << ttfb[i];
    json << "],\"generated\":[";
    for (std::size_t i = 0; i < produced.size(); ++i)
      json << (i ? "," : "") << produced[i];
    json << "]}";
    first_row = false;
  }
  json << "]}\n";
  if (flags.count("json")) {
    std::ofstream out(flags.at("json"));
    out << json.str();
  }
  return 0;
}

// Concurrent fixed-length requests (the end tokens are never sampled), each
// on its own corpus slice so that none reuses another's prompt: aggregate
// output throughput and per-request latency at a given session count.
int Throughput(const std::map<std::string, std::string>& flags) {
  const int context = std::stoi(Flag(flags, "context", "2048"));
  const int requests = std::stoi(Flag(flags, "requests", "16"));
  const int gen_tokens = std::stoi(Flag(flags, "gen-tokens", "1024"));
  auto engine_options = EngineOptionsFrom(flags);
  engine_options.max_context =
      std::max(engine_options.max_context, context + gen_tokens + 256);
  dg::Engine engine(Flag(flags, "model"), engine_options);
  const auto& tokenizer = engine.tokenizer();
  const auto corpus = tokenizer.Encode(ReadText(Flag(flags, "corpus")));
  std::cerr << "warmup " << engine.Warmup() / 1000.0 << " s\n";
  const std::size_t stride =
      (corpus.size() - static_cast<std::size_t>(context)) /
      std::max(1, requests);
  std::vector<std::vector<std::int32_t>> prompts;
  for (int i = 0; i < requests; ++i)
    prompts.push_back(ContextPrompt(tokenizer, corpus, context, i * stride));
  std::vector<dg::GenerationStats> stats(requests);
  std::atomic<int> next{0};
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> clients;
  for (int s = 0; s < engine_options.sessions; ++s)
    clients.emplace_back([&] {
      for (int i = next++; i < requests; i = next++) {
        dg::GenerationOptions options;
        options.diffusion = engine.diffusion_defaults();
        options.max_new_tokens = gen_tokens;
        options.suppress_end = true;
        options.reuse_prompt = false;
        options.seed = 1000 + i;
        stats[i] = engine.Generate(prompts[i], options, {});
      }
    });
  for (auto& client : clients)
    client.join();
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  double tokens = 0, ttfb = 0, steps = 0, blocks = 0, denoise = 0, latency = 0;
  for (const auto& s : stats) {
    tokens += s.generated_tokens;
    ttfb += s.first_block_ms / 1000.0;
    steps += s.denoising_steps;
    blocks += s.blocks;
    denoise += s.denoise_ms;
    latency += s.prefill_ms + s.denoise_ms + s.commit_ms;
  }
  std::printf("%8s %8s %8s %10s %10s %10s %10s %10s %10s\n", "sessions",
              "context", "requests", "wall_s", "agg_tps", "user_tps", "ttfb_s",
              "ms/step", "steps/blk");
  std::printf("%8d %8d %8d %10.2f %10.1f %10.1f %10.2f %10.1f %10.1f\n",
              engine_options.sessions, context, requests, wall, tokens / wall,
              tokens / std::max(1e-9, latency / 1000.0), ttfb / requests,
              denoise / std::max(1.0, steps), steps / std::max(1.0, blocks));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: gufo-diffusion-gemma "
                 "tokenize|render|generate|bench|logits|throughput ...\n";
    return 2;
  }
  try {
    const std::string command = argv[1];
    const auto flags = ParseFlags(argc, argv);
    if (command == "tokenize")
      return Tokenize(flags);
    if (command == "render")
      return Render(flags);
    if (command == "generate")
      return Generate(flags);
    if (command == "bench")
      return Bench(flags);
    if (command == "logits")
      return DumpLogits(flags);
    if (command == "throughput")
      return Throughput(flags);
    std::cerr << "unknown command " << command << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
