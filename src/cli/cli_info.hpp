#pragma once

#include <iomanip>
#include <iostream>
#include <string_view>

#include "src/cli/brand.hpp"
#include "src/cli/command_table.hpp"

namespace gufo::cli {
inline void PrintGlobalVersion() {
  std::cout << gufo::cli::FormatVersion(
                   gufo::cli::kProductName, gufo::cli::kProductVersion,
                   gufo::cli::kRevision, gufo::cli::kEngineVersion)
            << '\n';
}

inline void PrintGlobalHelp(std::string_view program_name) {
  std::cout << "Usage: " << program_name << " <COMMAND> [OPTIONS]\n\n"
            << "Local LLM server for AMD Strix Halo (gfx1151).\n"
            << "Start here: bubo prepare, then bubo serve.\n\n"
            << "Commands:\n";
  for (const auto& command : kCommands<CommandLayer::Metadata>)
    std::cout << "  " << std::left << std::setw(15) << command.name
              << command.summary << '\n';
  std::cout << '\n'
            << "Run '" << program_name << " help <COMMAND>' or '"
            << program_name << " <COMMAND> --help' for details on a command\n\n"
            << "Options:\n"
            << "  -h, --help     Print help\n"
            << "  -v, --version  Print version\n";
}

}  // namespace gufo::cli
