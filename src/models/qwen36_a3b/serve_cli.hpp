#ifndef GUFO_MODELS_QWEN36_A3B_SERVE_CLI_HPP_
#define GUFO_MODELS_QWEN36_A3B_SERVE_CLI_HPP_

#include <span>

/// `gufo serve` for Qwen3.6-35B-A3B (qwen35moe GGUFs): gufo.exe's serve entry
/// hands its arguments here when --model names such a file, so the A3B
/// engine and server (serve.cpp) run from gufo.exe with gufo serve's flags.
namespace gufo::models::qwen36_a3b {

/// True when the `gufo serve` arguments name a qwen35moe GGUF (-m/--model).
bool IsA3bServeModel(std::span<const char* const> args);

/// Runs the A3B server with `gufo serve` flags (FN-only flags are accepted
/// and ignored, so a launch line only needs its model swapped).
int RunGufoServe(std::span<const char* const> args);

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_SERVE_CLI_HPP_
