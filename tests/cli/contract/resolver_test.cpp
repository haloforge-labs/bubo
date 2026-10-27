#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/core/prepare/metadata_json.hpp"

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  using namespace bubo::cli;
  try {
    if (argc != 2)
      throw std::runtime_error("test needs an owned root");
    const auto root =
        fs::absolute(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    fs::create_directories(root);
    const auto local = (root / "local").string();
    const auto custom = root / "custom-output/models/custom/bubo-package.json";
    fs::create_directories(custom.parent_path());
    std::ofstream(custom) << "{}\n";
    unsigned passed = 0;
    const auto check = [&](bool condition) {
      if (!condition)
        throw std::runtime_error("resolver precedence failed");
      ++passed;
    };
    const auto standard = ResolvePreparedDefault({}, {}, {}, local);
    check(standard.source == "standard_location" &&
          standard.manifest.string().find("Bubo") != std::string::npos);
    std::string error;
    check(RecordPreparedDefault(local, custom, error));
    const auto saved = ResolvePreparedDefault({}, {}, {}, local);
    check(saved.source == "prepared_default" && saved.manifest == custom);
    const auto from_env = ResolvePreparedDefault(
        {}, {}, (root / "env/bubo-package.json").string(), local);
    check(from_env.source == "environment" && from_env.manifest != custom);
    const auto explicit_manifest = ResolvePreparedDefault(
        {}, root / "explicit/bubo-package.json", "env", local);
    check(explicit_manifest.source == "explicit_manifest");
    const auto explicit_model =
        ResolvePreparedDefault(root / "explicit/model.gguf", {}, "env", local);
    check(explicit_model.source == "explicit_model" &&
          explicit_model.manifest.empty());
    check(
        ResolvePreparedDefault(root / "model", custom, "env", local).manifest ==
        custom);
    const char* flags[] = {"model", "--output", "custom-output", "--json"};
    const auto parsed = bubo::prepare::Parse(flags, local);
    check(parsed && parsed.options.root == "custom-output" &&
          parsed.options.machine);
    // Replace the publication pointer through the same atomic writer.
    const auto next = root / "second/bubo-package.json";
    fs::create_directories(next.parent_path());
    std::ofstream(next) << "{}\n";
    check(RecordPreparedDefault(local, next, error));
    check(ResolvePreparedDefault({}, {}, {}, local).manifest == next);
    std::ofstream(saved.pointer, std::ios::trunc) << "broken";
    bool rejected = false;
    try {
      (void)ResolvePreparedDefault({}, {}, {}, local);
    } catch (const std::exception&) {
      rejected = true;
    }
    check(rejected);
    check(ResolvePreparedDefault({}, custom, {}, local).manifest == custom);
    std::cout << "Resolver precedence and custom output: " << passed
              << " passed, 0 failed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
