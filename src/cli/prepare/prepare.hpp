#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bubo::prepare {
enum class Step { All, Runtime, Model };
enum class ExitCode {
  Success = 0,
  Syntax = 2,
  Cancel = 3,
  MissingSource = 4,
  Transport = 5,
  Integrity = 6,
  Storage = 7,
  Locked = 8,
  Internal = 9,
  NotImplemented = 10
};
struct Options {
  Step step = Step::All;
  std::string root;
  std::string mtp_model;
  std::string mmproj;
  std::string cache_dir;
  std::string runtime_dir;
  bool resume = true;
  bool yes = false;
  bool verify = false;
  bool offline = false;
  bool help = false;
};
struct ParseResult {
  Options options;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};
struct ConsentVolume {
  std::string name;
  std::uint64_t free_bytes = 0, required_bytes = 0;
};
// Read-only preflight output. Internal orchestration displays both steps before
// one approval; this does not provide a CLI or catalogue override.
struct PreparationPlan {
  bool ready = false;
  std::string text;
  std::vector<ConsentVolume> volumes;
};
using PreparePhase =
    std::function<int(const Options&, std::istream&, std::ostream&,
                      std::ostream&, PreparationPlan*, bool)>;
int RunAll(const Options&, const PreparePhase& model,
           const PreparePhase& runtime, std::istream&, std::ostream&,
           std::ostream&, const std::function<bool()>& cancelled = {});
ParseResult Parse(std::span<const char* const> args,
                  std::string_view local_app_data);
std::string ModelRoot(const Options& options);
void PrintHelp(std::ostream& out);
int Run(std::span<const char* const> args, std::string_view local_app_data,
        std::ostream& out, std::ostream& err);
}  // namespace bubo::prepare
