// Operator-only capture: one cold raw 32K prompt and 64 greedy tokens.
// The same source also builds against the pinned Stage B engine for parity.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "prefill_capture_options.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

namespace a3b = gufo::models::qwen36_a3b;
void Save(const std::string& path, const void* bytes, std::size_t size) {
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(path, std::ios::binary);
  out.write(static_cast<const char*>(bytes),
            static_cast<std::streamsize>(size));
  out.close();
}
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
      std::cout << kCaptureUsage << '\n';
      return 0;
    }
    const auto capture = ParseCaptureOptions(argc, argv);
    if (!capture.audit.empty()) {
#ifdef _WIN32
      if (_putenv_s("GUFO_PREFILL_EXPERTS_AUDIT", capture.audit.c_str()) != 0)
#else
      if (setenv("GUFO_PREFILL_EXPERTS_AUDIT", capture.audit.c_str(), 1) != 0)
#endif
        throw std::runtime_error("cannot set audit mode");
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open prompt");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad())
      throw std::runtime_error("prompt read failed");
    a3b::Options options;
    options.max_context = 65536;
    options.prefill_chunk = capture.chunk;
    std::string error;
    const auto check = [&](bool ok) {
      if (!ok)
        throw std::runtime_error(error);
    };
    auto engine = a3b::Engine::Load(argv[1], options, &error);
    check(engine != nullptr);
    auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromGguf(
        engine->gguf(), &error);
    check(tokenizer != nullptr);
    const auto encoded = tokenizer->Encode(text);
    if (encoded.size() != capture.tokens)
      throw std::invalid_argument(
          "raw prompt token count differs from --prompt-tokens");
    std::vector<std::int32_t> prompt(encoded.begin(), encoded.end());
    a3b::Candidates next;
    const auto began = std::chrono::steady_clock::now();
    check(engine->SpecPrefill(prompt, &next, &error));
    const double prefill_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - began)
                                  .count();
    std::vector<float> frontier(engine->config().vocab);
    check(engine->RowLogits(0, frontier.data(), &error));
    if (!std::all_of(frontier.begin(), frontier.end(),
                     [](float v) { return std::isfinite(v); }))
      throw std::runtime_error("nonfinite frontier logits");
    const std::string prefix(argv[3]);
    Save(prefix + ".f32", frontier.data(), frontier.size() * sizeof(float));
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> drafts;
    std::vector<a3b::Candidates> rows;
    for (int i = 0; i < 64; ++i) {
      const auto token = next.ids[0];
      ids.push_back(token);
      if (i == 63)
        break;
      check(engine->SpecStep(token, 0, &drafts, &rows, &error));
      check(engine->SpecCommit(1, &error));
      if (rows.size() != 1)
        throw std::runtime_error("unexpected greedy row count");
      next = rows.front();
    }
    Save(prefix + ".i32", ids.data(), ids.size() * sizeof(std::int32_t));
    std::vector<gufo::tokenization::TokenId> decoded(ids.begin(), ids.end());
    const auto answer = tokenizer->Decode(decoded);
    Save(prefix + ".answer.txt", answer.data(), answer.size());
    std::ofstream meta;
    meta.exceptions(std::ios::failbit | std::ios::badbit);
    meta.open(prefix + ".meta", std::ios::binary);
    meta << "prompt_tokens=" << capture.tokens
         << "\ngenerated_tokens=64\nvocab=" << frontier.size()
         << "\nprefill_ms=" << prefill_ms << '\n';
    meta.close();
    std::cout << "PASS capture prompt_tokens=" << capture.tokens
              << " generated_tokens=64 chunk=" << capture.chunk
              << " prefill_ms=" << prefill_ms
              << " tok_s=" << capture.tokens * 1000.0 / prefill_ms << '\n';
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
