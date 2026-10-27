#include <iostream>

#include "src/cli/machine_exit.hpp"
#include "src/cli/prepare/prepare.hpp"

int main() {
  using namespace bubo::cli;
  using bubo::prepare::ExitCode;
  static_assert(static_cast<int>(ExitCode::Syntax) == 2);
  static_assert(static_cast<int>(ExitCode::Cancel) == 3);
  static_assert(static_cast<int>(ExitCode::MissingSource) == 4);
  static_assert(static_cast<int>(ExitCode::Transport) == 5);
  static_assert(static_cast<int>(ExitCode::Integrity) == 6);
  static_assert(static_cast<int>(ExitCode::Storage) == 7);
  static_assert(static_cast<int>(ExitCode::Locked) == 8);
  static_assert(static_cast<int>(ExitCode::Internal) == 9);
  static_assert(static_cast<int>(ExitCode::NotImplemented) == 10);
  unsigned count = 0;
  for (int code = 11; code <= 18; ++code) {
    if (LifecycleExit(true, static_cast<MachineExit>(code), 2) != code ||
        LifecycleExit(false, static_cast<MachineExit>(code), 2) != 2)
      return 1;
    count += 2;
  }
  std::cout << "Lifecycle exits: " << count << " passed, 0 failed\n";
}
