#pragma once

#include <iostream>
#include <string_view>

#include "src/cli/brand.hpp"
#include "src/cli/model_families.hpp"

namespace gufo::cli {
inline void PrintGlobalVersion() {
  std::cout << gufo::cli::FormatVersion(
                   gufo::cli::kProductName, gufo::cli::kProductVersion,
                   gufo::cli::kRevision, gufo::cli::kEngineVersion)
            << '\n';
}

inline void PrintGlobalHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " <COMMAND> [OPTIONS]\n\n"
      << "Fast LLM & DiT video inference engine for AMD Strix Halo "
         "(gfx1151).\n\n"
      << "Commands:\n"
      << "  prepare        Prepare pinned runtime and model files\n"
      << "  serve          Start the OpenAI-compatible HTTP server\n"
      << "  prompt         Execute one prompt request and exit\n"
      << "  chat           Start an interactive terminal conversation\n"
      << "  bench          Benchmark prompt processing and token generation\n"
      << "  eval           Evaluate an OpenAI-compatible text server\n"
      << "  video          Generate MiniMax H3 text-to-video\n"
      << "  transcribe     Transcribe WAV audio with Qwen3-ASR-1.7B\n"
      << "  diagnose       Inspect system hardware, memory, and drivers\n"
      << "  help           Print help for a specific command\n\n"
      << "Run '" << program_name << " help <COMMAND>' or '" << program_name
      << " <COMMAND> --help' for details on a command\n\n"
      << "Options:\n"
      << "  -h, --help     Print help\n"
      << "  -v, --version  Print version\n\n"
      << "Model families (serve llm / asr / tts / image / video):\n";
  for (const auto family : gufo::cli::kModelFamilies)
    std::cout << "  " << family << '\n';
}

}  // namespace gufo::cli
