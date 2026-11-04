#ifndef GUFO_CLI_REFUSAL_OPTIONS_HPP_
#define GUFO_CLI_REFUSAL_OPTIONS_HPP_

#include <optional>
#include <string>
#include <string_view>

#include "src/cli/arg_parser.hpp"
#include "src/models/qwen38_flash_next/refusal_direction.hpp"

namespace gufo::cli {

inline void RegisterRefusalOptions(
    ArgParser& parser, models::qwen38_flash_next::RefusalOverrides* options) {
  using Options = models::qwen38_flash_next::RefusalOverrides;
  const auto add = [&](const char* name, const char* hint, const char* help,
                       std::optional<std::string> Options::* member) {
    parser.AddCustomOption(
        "", name, hint, help, "Model",
        [options, member](std::string_view, std::string_view value,
                          std::string*) {
          options->*member = std::string(value);
          return true;
        });
  };
  add("--refusal-direction", "on|off",
      "Use the packaged direction (default: off)", &Options::enabled);
  add("--refusal-direction-file", "PATH",
      "Override the packaged direction file", &Options::file);
  add("--refusal-direction-strength", "NUMBER",
      "Projection strength from 0 to 4 (default: fitted value in file)",
      &Options::strength);
  add("--refusal-direction-mode", "sites|stream",
      "Project selected outputs or the residual stream (default: sites)",
      &Options::mode);
  add("--refusal-direction-layer-start", "N",
      "First stream layer, 1 through 47 (default: 4; ignored in sites mode)",
      &Options::first_layer);
  add("--refusal-direction-layer-end", "N",
      "Last stream layer, 1 through 47 (default: 44; ignored in sites mode)",
      &Options::last_layer);
}

}  // namespace gufo::cli
#endif  // GUFO_CLI_REFUSAL_OPTIONS_HPP_
