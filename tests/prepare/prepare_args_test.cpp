#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/cli/prepare/prepare.hpp"

int main() {
  using namespace bubo::prepare;
  struct Case {
    std::vector<const char*> args;
    const char* local;
    int code;
    Step step;
  };
  const std::vector<Case> cases = {
      {{}, "C:\\Users\\fixture\\AppData\\Local", 10, Step::All},
      {{"all"}, "C:\\Local", 10, Step::All},
      {{"runtime"}, "C:\\Local", 10, Step::Runtime},
      {{"model"}, "C:\\Local", 10, Step::Model},
      {{"--output", "D:\\Bubo Data", "--yes", "--verify", "--offline"},
       "",
       10,
       Step::All},
      {{"runtime", "--output", "D:\\データ", "--offline"},
       "",
       10,
       Step::Runtime},
      {{"model", "--output", "D:\\Data", "--yes", "--yes"},
       "",
       10,
       Step::Model},
      {{"--help"}, "", 0, Step::All},
      {{"runtime", "--help"}, "", 0, Step::Runtime},
      {{"model", "--help"}, "", 0, Step::Model},
      {{"all", "--help"}, "", 0, Step::All},
      {{}, "", 2, Step::All},
      {{""}, "C:\\Local", 2, Step::All},
      {{"MODEL"}, "C:\\Local", 2, Step::All},
      {{"runtime", "model"}, "C:\\Local", 2, Step::All},
      {{"--yes", "model"}, "C:\\Local", 2, Step::All},
      {{"--output"}, "C:\\Local", 2, Step::All},
      {{"--output", ""}, "C:\\Local", 2, Step::All},
      {{"--output", "--yes"}, "C:\\Local", 2, Step::All},
      {{"--output", "D:\\A", "--output", "D:\\B"}, "", 2, Step::All},
      {{"--output=D:\\A"}, "", 2, Step::All},
      {{"--output", "bad\npath"}, "", 2, Step::All},
      {{"--help", "--force"}, "", 2, Step::All},
      {{"--source", "D:\\A"}, "C:\\Local", 2, Step::All},
      {{"--imatrix", "D:\\A"}, "C:\\Local", 2, Step::All},
      {{"--download-imatrix"}, "C:\\Local", 2, Step::All},
      {{"--url", "https://example.invalid"}, "C:\\Local", 2, Step::All},
      {{"--hash", "abc"}, "C:\\Local", 2, Step::All},
      {{"--help", ""}, "", 2, Step::All},
  };
  for (const auto& test : cases) {
    std::ostringstream out, err;
    assert(Run(test.args, test.local, out, err) == test.code);
    const auto parsed = Parse(test.args, test.local);
    if (test.code != 2) {
      assert(parsed && parsed.options.step == test.step);
      assert(err.str().empty());
      if (test.code == 10) {
        const bool runtime =
            out.str().find("prepare runtime: not implemented in I1\n") !=
            std::string::npos;
        const bool model =
            out.str().find("prepare model: not implemented in I1\n") !=
            std::string::npos;
        assert(runtime == (test.step != Step::Model));
        assert(model == (test.step != Step::Runtime));
      }
    } else {
      assert(!parsed && out.str().empty());
      assert(err.str().starts_with("prepare: E2 "));
    }
  }
  const std::array<const char*, 4> flags{"--yes", "--verify", "--offline",
                                         "--yes"};
  const auto parsed = Parse(flags, "C:\\Users\\fixture\\AppData\\Local");
  assert(parsed.options.yes && parsed.options.verify && parsed.options.offline);
  assert(std::filesystem::u8path(parsed.options.root) ==
         std::filesystem::u8path("C:\\Users\\fixture\\AppData\\Local") /
             "Bubo");
  assert(std::filesystem::u8path(ModelRoot(parsed.options)) ==
         std::filesystem::u8path(parsed.options.root) / "models" /
             "huihui-qwen38-flash-next-ud-q4-k-xl-v1");
  std::cout << "PASS: prepare argument matrix (" << cases.size()
            << " cases), UTF-8 paths, defaults, stubs and exit codes\n";
}
