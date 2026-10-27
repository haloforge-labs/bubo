#include "src/cli/command_table.hpp"

#include <cassert>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "src/cli/cli_info.hpp"

int main() {
  using namespace gufo::cli;
  std::set<std::string_view> names;
  unsigned engine_routes = 0, help_calls = 0, checks = 0;
  CommandContext context;
  context.load_engine = [&] {
    ++engine_routes;
    return 71;
  };
  context.help = [&](auto args) {
    ++help_calls;
    return args.size() == 1 ? 73 : 2;
  };
  std::ostringstream help;
  auto* previous = std::cout.rdbuf(help.rdbuf());
  PrintGlobalHelp(kProductName);
  std::cout.rdbuf(previous);
  for (const auto& command : kCommands<CommandLayer::Bootstrap>) {
    assert(!command.name.empty() && !command.summary.empty() && command.run);
    assert(names.insert(command.name).second);
    assert(FindCommand<CommandLayer::Bootstrap>(command.name) == &command);
    const auto* metadata = FindCommand<CommandLayer::Metadata>(command.name);
    assert(metadata && metadata->summary == command.summary && !metadata->run);
    assert(help.str().find("  " + std::string(command.name) + " ") !=
           std::string::npos);
    checks += 5;
    if (command.engine_needed) {
      const std::string name(command.name);
      const char* args[] = {name.c_str(), "--help"};
      assert(command.run(args, context) == 71);
      ++checks;
    }
    std::cout << "PASS: command table " << command.name << '\n';
  }
  for (const auto* name : {"prepare", "status", "stop"}) {
    assert(!FindCommand<CommandLayer::Bootstrap>(name)->engine_needed);
    ++checks;
  }
  assert(FindCommand<CommandLayer::Bootstrap>("models")->engine_needed);
  assert(!FindCommand<CommandLayer::Bootstrap>("unknown-command"));
  const char* args[] = {"help"};
  assert(FindCommand<CommandLayer::Bootstrap>("help")->run(args, context) ==
         73);
  assert(help_calls == 1 && engine_routes > 0);
  checks += 4;
  std::cout << "Command table: " << names.size() << " commands, "
            << engine_routes << " engine routes, " << checks
            << " checks passed, 0 failed\n";
}
