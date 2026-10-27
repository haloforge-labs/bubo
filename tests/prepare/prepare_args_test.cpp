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
  std::vector<Case> cases = {
      {{}, "C:\\Users\\fixture\\AppData\\Local", 4, Step::All},
      {{"all"}, "C:\\Local", 4, Step::All},
      {{"runtime"}, "C:\\Local", 4, Step::Runtime},
      {{"model"}, "C:\\Local", 4, Step::Model},
      {{"--output", "D:\\Bubo Data", "--yes", "--verify", "--offline"},
       "",
       4,
       Step::All},
      {{"runtime", "--output", "D:\\データ", "--offline"},
       "",
       4,
       Step::Runtime},
      {{"model", "--output", "D:\\Data", "--yes", "--yes"}, "", 4, Step::Model},
      {{"model", "--no-resume"}, "C:\\Local", 4, Step::Model},
      {{"model", "--resume", "--no-resume"}, "C:\\Local", 4, Step::Model},
      {{"model", "--no-resume", "--resume"}, "C:\\Local", 4, Step::Model},
      {{"model", "--mtp-model", "D:\\既存\\mtp.gguf", "--mmproj",
        "D:\\mmproj.gguf", "--cache-dir", "D:\\Cache", "--output", "D:\\Data"},
       "",
       4,
       Step::Model},
      {{"runtime", "--runtime-dir", "D:\\ROCm", "--cache-dir", "D:\\Cache"},
       "C:\\Local",
       4,
       Step::Runtime},
      {{"all", "--mtp-model", "D:\\mtp.gguf", "--mmproj", "D:\\mmproj.gguf",
        "--runtime-dir", "D:\\ROCm"},
       "C:\\Local",
       4,
       Step::All},
      {{"runtime", "--mtp-model", "D:\\mtp.gguf"}, "C:\\Local", 2, Step::All},
      {{"runtime", "--mmproj", "D:\\mmproj.gguf"}, "C:\\Local", 2, Step::All},
      {{"model", "--runtime-dir", "D:\\ROCm"}, "C:\\Local", 2, Step::All},
      {{"model", "--mtp-model", "D:\\mtp.gguf"}, "", 2, Step::All},
      {{"--resume=true"}, "C:\\Local", 2, Step::All},
      {{"--no_resume"}, "C:\\Local", 2, Step::All},
      {{"--mtp-model=D:\\mtp.gguf"}, "C:\\Local", 2, Step::All},
      {{"--mmproj=D:\\mmproj.gguf"}, "C:\\Local", 2, Step::All},
      {{"--cache-dir=D:\\Cache"}, "C:\\Local", 2, Step::All},
      {{"--runtime-dir=D:\\ROCm"}, "C:\\Local", 2, Step::All},
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
  for (const auto* flag :
       {"--mtp-model", "--mmproj", "--cache-dir", "--runtime-dir"}) {
    cases.push_back({{flag}, "C:\\Local", 2, Step::All});
    cases.push_back({{flag, ""}, "C:\\Local", 2, Step::All});
    cases.push_back({{flag, "--yes"}, "C:\\Local", 2, Step::All});
    cases.push_back({{flag, "bad\npath"}, "C:\\Local", 2, Step::All});
    cases.push_back(
        {{flag, "D:\\A", flag, "D:\\B"}, "C:\\Local", 2, Step::All});
  }
  for (const auto& test : cases) {
    std::ostringstream out, err;
    const auto parsed = Parse(test.args, test.local);
    int expected_code = test.code;
    if (parsed && parsed.options.step == Step::Runtime &&
        !parsed.options.help) {
      // Runtime grammar is checked above using the original matrix. Execute
      // the real installation gate against a job-contained missing product;
      // no test may contact AMD or write to an example C:/D: path.
      const auto destination =
          (std::filesystem::current_path() / "j3-args-missing-install")
              .string();
      assert(!std::filesystem::exists(destination));
      const auto probe_root = destination + "-root";
      assert(!std::filesystem::exists(probe_root));
      const std::array<const char*, 6> runtime_probe{
          "runtime",           "--offline", "--runtime-dir",
          destination.c_str(), "--output",  probe_root.c_str()};
#ifdef _WIN32
      // --runtime-dir is a plain folder: no bubo.exe or engine copy is
      // required there (the old identity check is gone). An offline run with
      // an empty cache stops before any write, with a cache error.
      Run(runtime_probe, test.local, out, err);
      assert(err.str().find("identical bubo.exe") == std::string::npos);
      assert(err.str().find("different bubo") == std::string::npos);
      // A relative runtime folder is refused with a plain message.
      std::ostringstream relative_out, relative_err;
      const std::array<const char*, 7> relative_probe{
          "runtime", "--offline", "--runtime-dir",   "relative-runtime-dir",
          "--yes",   "--output",  probe_root.c_str()};
      assert(Run(relative_probe, test.local, relative_out, relative_err) == 2);
      assert(relative_err.str().find("absolute local path") !=
             std::string::npos);
      assert(!std::filesystem::exists("relative-runtime-dir"));
#else
      expected_code = 2;
      assert(Run(runtime_probe, test.local, out, err) == expected_code);
      assert(err.str().find("requires Windows") != std::string::npos);
#endif
      assert(!std::filesystem::exists(probe_root + "/runtime"));
      assert(!std::filesystem::exists(destination));
    } else if (test.code == 4) {
      // The model gate is closed in the release (notice bundle embedded), so
      // a real Run of an accepted line reaches the free-space check and, on a
      // large disk, the network. Execute a hermetic probe instead: --offline
      // into a job-contained directory fails with E4 before any disk or
      // network step. The argument grammar itself is asserted below.
      const auto root =
          (std::filesystem::current_path() / "args-offline-root").string();
      std::filesystem::create_directories(root);
      const std::array<const char*, 4> offline_probe{
          test.step == Step::Model ? "model" : "all", "--offline", "--output",
          root.c_str()};
      assert(Run(offline_probe, test.local, out, err) == expected_code);
      assert(err.str().find("offline: missing") != std::string::npos);
      assert(out.str().empty());
    } else
      assert(Run(test.args, test.local, out, err) == expected_code);
    if (test.code != 2) {
      assert(parsed && parsed.options.step == test.step);
      if (expected_code != 0) {
        assert(err.str().starts_with("prepare: E" +
                                     std::to_string(expected_code) + " "));
        assert(out.str().empty());
      } else {
        assert(err.str().empty());
      }
    } else {
      assert(!parsed && out.str().empty());
      assert(err.str().starts_with("prepare: E2 "));
    }
  }
  const std::array<const char*, 4> flags{"--yes", "--verify", "--offline",
                                         "--yes"};
  const auto parsed = Parse(flags, "C:\\Users\\fixture\\AppData\\Local");
  assert(parsed.options.yes && parsed.options.verify &&
         parsed.options.offline && parsed.options.resume);
  assert(std::filesystem::u8path(parsed.options.root) ==
         std::filesystem::u8path("C:\\Users\\fixture\\AppData\\Local") /
             "Bubo");
  assert(std::filesystem::u8path(ModelRoot(parsed.options)) ==
         std::filesystem::u8path(parsed.options.root) / "models" /
             "huihui-qwen38-flash-next-ud-q4-k-xl-v1");
  const std::array<const char*, 10> model_paths{
      "model",    "--no-resume",     "--mtp-model", "D:\\既存\\mtp.gguf",
      "--mmproj", "D:\\mmproj.gguf", "--cache-dir", "D:\\Cache",
      "--output", "D:\\Bubo Data"};
  const auto paths = Parse(model_paths, "");
  assert(paths && !paths.options.resume &&
         paths.options.mtp_model == "D:\\既存\\mtp.gguf" &&
         paths.options.mmproj == "D:\\mmproj.gguf" &&
         paths.options.cache_dir == "D:\\Cache" &&
         paths.options.root == "D:\\Bubo Data");
  const std::array<const char*, 3> resume_last{"model", "--no-resume",
                                               "--resume"};
  const std::array<const char*, 3> restart_last{"model", "--resume",
                                                "--no-resume"};
  assert(Parse(resume_last, "C:\\Local").options.resume);
  assert(!Parse(restart_last, "C:\\Local").options.resume);
  const std::array<const char*, 5> runtime_paths{
      "runtime", "--runtime-dir", "D:\\ROCm", "--cache-dir", "D:\\Cache"};
  const auto runtime = Parse(runtime_paths, "C:\\Local");
  assert(runtime && runtime.options.runtime_dir == "D:\\ROCm" &&
         runtime.options.cache_dir == "D:\\Cache");
  const std::array<const char*, 3> custom_output{"runtime", "--output",
                                                 "D:\\Bubo Data"};
  const auto custom_runtime = Parse(custom_output, "C:\\Local");
  assert(custom_runtime && custom_runtime.options.root == "D:\\Bubo Data" &&
         custom_runtime.options.local_app_data == "C:\\Local" &&
         custom_runtime.options.runtime_dir.empty());
  std::cout << "PASS: prepare_output_preserves_process_runtime_location "
               "(1 case)\n";
  std::cout << "PASS: prepare argument matrix (" << cases.size()
            << " cases), UTF-8 paths, defaults, production OPEN gate and exit "
               "codes\n";
}
