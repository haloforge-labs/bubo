#pragma once

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace gufo::cli {

struct CommandContext {
  std::function<int()> load_engine;
  std::function<int(std::span<const char* const>)> help;
};
using CommandRunner = int (*)(std::span<const char* const>,
                              const CommandContext&);
struct Command {
  std::string_view name, summary;
  bool engine_needed, machine_supported, help_topic;
  CommandRunner run;
};
enum class CommandLayer { Metadata, Bootstrap, Engine };

namespace commands {
#define BUBO_COMMAND(name, summary, engine, machine, help, runner) \
  int runner(std::span<const char* const>, const CommandContext&);
#include "src/cli/command_table.inc"
#undef BUBO_COMMAND
inline int LoadEngine(std::span<const char* const>,
                      const CommandContext& context) {
  return context.load_engine();
}
}  // namespace commands

// Bootstrap binds engine commands to its loader. The engine binds those same
// rows to the actual runners. Metadata-only consumers need no engine symbols.
template<CommandLayer Layer>
inline constexpr auto kCommands = std::to_array<Command>({
#define BUBO_COMMAND(name, summary, engine, machine, help, runner)         \
  Command{name, summary, engine, machine, help, []() -> CommandRunner {    \
            if constexpr (Layer == CommandLayer::Metadata)                 \
              return nullptr;                                              \
            else if constexpr (Layer == CommandLayer::Bootstrap && engine) \
              return &commands::LoadEngine;                                \
            else                                                           \
              return &commands::runner;                                    \
          }()},
#include "src/cli/command_table.inc"
#undef BUBO_COMMAND
});

template<CommandLayer Layer>
const Command* FindCommand(std::string_view name) {
  for (const auto& command : kCommands<Layer>)
    if (command.name == name)
      return &command;
  return nullptr;
}

// Engine-free variants of otherwise engine-backed commands, shared by both
// entry points. A disengaged result means normal table dispatch should run.
std::optional<int> RunCommandVariant(std::span<const char* const> options);
}  // namespace gufo::cli
