#include <array>
#include <iostream>
#include <sstream>

#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepare/vc_runtime.hpp"

int main() {
  using namespace bubo::prepare;
  VcRuntimeProbe missing{[] { return false; },
                         [](std::string_view) { return false; }};
  unsigned passed = 0;
  for (const auto* mode : {"--json", "--jsonl"}) {
    const std::array<const char*, 4> args{"model", "--output", "unused", mode};
    std::ostringstream output, errors;
    if (Run(args, "", output, errors, &missing) != 4)
      return 1;
    std::istringstream lines(output.str());
    std::string line, extra;
    if (!std::getline(lines, line) || std::getline(lines, extra) ||
        output.str().find('\r') != std::string::npos)
      return 1;
    const auto event = gufo::json::parse(line);
    if (event.member_str("type") != "error" ||
        event.find("payload")->member_str("code") != "runtime_missing")
      return 1;
    ++passed;
  }
  std::cout << "vc_runtime_missing_after_main_json: " << passed
            << " passed, 0 failed\n";
}
