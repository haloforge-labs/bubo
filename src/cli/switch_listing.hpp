#pragma once

#include <ostream>
#include <span>

#include "src/cli/arg_parser.hpp"
#include "src/cli/machine_output.hpp"

namespace bubo::cli {
inline Value SwitchListing() {
  namespace json = gufo::json;
  static const std::vector<Value> rows = {
#define GUFO_RELEASE_ROWS
#include "src/cli/serve/release_switches.inc"
#undef GUFO_RELEASE_ROWS
  };
  auto listing = Value::object();
  listing["schema_version"] = 1;
  listing["switches"] = Value::array();
  for (const auto& row : rows)
    listing["switches"].push_back(row);
  return listing;
}
inline bool ListSwitchesRequested(std::span<const char* const> args) {
  return gufo::cli::ArgParser::Contains(args, {"--list-switches"});
}
inline int RunListSwitches(std::span<const char* const> args,
                           std::ostream& out) {
  MachineWriter writer(out);
  if (args.size() != 2 || !ListSwitchesRequested(args) ||
      (std::string_view(args[0]) != "--json" &&
       std::string_view(args[1]) != "--json")) {
    writer.Write("error",
                 Error("invalid_arguments", "serve", "arguments", false,
                       "Use bubo serve --list-switches --json."));
    return 2;
  }
  return writer.Write("switches", SwitchListing()) ? 0 : 1;
}
}  // namespace bubo::cli
