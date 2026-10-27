// CPU process fixture for the real native stdin reader. No engine/network.
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/prepare.hpp"

int main(int argc, char** argv) {
  using namespace bubo::prepare;
  using gufo::json::Value;
  const std::string target = argc == 2 ? argv[1] : "verify";
  Options options;
  options.machine = true;
  options.step = Step::Model;
  bool ready = target == "noop";
  bubo::cli::MachineWriter writer(std::cout, true);
  const PreparePhase phase = [&](const Options& o, std::istream&, std::ostream&,
                                 std::ostream&, PreparationPlan* preview,
                                 bool) {
    if (preview) {
      if (!ready && target == "preflight") {
        ReportProgress(o, "model", "verify", "fixture", 0, 3);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < deadline) {
          if (o.cancelled())
            return 9;
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
      }
      preview->ready = ready;
      auto plan = Value::object();
      plan["component"] = "model";
      plan["ready"] = ready;
      plan["sources"] = Value::array();
      plan["licenses"] = Value::array();
      plan["paths"] = Value::object();
      plan["volumes"] = Value::array();
      preview->machine = plan;
      return 0;
    }
    for (const auto* current :
         {"download", "copy", "verify", "extract", "promote"}) {
      ReportProgress(o, "model", current, "fixture", 0, 3);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(3);
      do {
        if (o.cancelled())
          return 9;
        if (target != current)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } while (std::chrono::steady_clock::now() < deadline);
    }
    ready = true;
    return 0;
  };
  return RunMachine(options, phase, {}, std::cin, writer, std::cerr);
}
