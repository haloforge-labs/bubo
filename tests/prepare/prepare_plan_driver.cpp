// Prints the real prepare plan without a download: runs the production
// prepare::Run with the given arguments. Close stdin (or send a decline
// record) and the run ends at the approval step with exit 3 before any
// request or write. Example:
//   prepare_plan_driver model --model-set original --json --output D:\BuboData
// Not a test. Build target: prepare_plan_driver (EXCLUDE_FROM_ALL).
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/cli/prepare/prepare.hpp"

int main(int argc, char** argv) {
  std::vector<const char*> args(argv + 1, argv + argc);
  const char* local = std::getenv("LOCALAPPDATA");
  return bubo::prepare::Run(args, local ? local : "", std::cout, std::cerr);
}
