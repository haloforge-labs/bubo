#pragma once

#include <iosfwd>
#include <span>
#include <string>
#include <string_view>

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
ParseResult Parse(std::span<const char* const> args,
                  std::string_view local_app_data);
std::string ModelRoot(const Options& options);
void PrintHelp(std::ostream& out);
int Run(std::span<const char* const> args, std::string_view local_app_data,
        std::ostream& out, std::ostream& err);
}  // namespace bubo::prepare
