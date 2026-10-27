// --no-vision must win over the "mmproj-BF16.gguf beside the model" discovery of qwen::vision::Encoder::Open (Qwen3.8 27B, Qwen3.6 27B,
// Cyber-Tiel / A3B and Flash-Next all load their encoder through OpenUnlessDisabled). The sidecar here is a garbage file: Open must try it
// (and throw) when image input is on, and must not look at it at all when it is off.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "src/models/qwen/vision/encoder.hpp"

namespace fs = std::filesystem;
using gufo::models::qwen::vision::Encoder;

int main() {
  const auto root = fs::temp_directory_path() / ("bubo-vision-disabled-test-" + std::to_string(std::rand()));
  fs::create_directories(root);
  const auto model = root / "model.gguf";
  std::ofstream(model) << "not a model";
  std::ofstream(root / "mmproj-BF16.gguf") << "not a projector";

  int failures = 0;
  const auto expect = [&](bool ok, const char* what) {
    if (!ok) {
      std::fprintf(stderr, "FAIL: %s\n", what);
      ++failures;
    }
  };

  bool disabled_threw = false;
  std::shared_ptr<Encoder> disabled;
  try {
    disabled = Encoder::OpenUnlessDisabled(true, model, {}, 5120);
  } catch (...) {
    disabled_threw = true;
  }
  expect(!disabled_threw && disabled == nullptr, "disabled: nothing is opened, no encoder, no error, although a sidecar sits beside the model");

  // An explicit off wins over an explicit sidecar path as well.
  bool explicit_threw = false;
  std::shared_ptr<Encoder> explicit_off;
  try {
    explicit_off = Encoder::OpenUnlessDisabled(true, model, root / "mmproj-BF16.gguf", 5120);
  } catch (...) {
    explicit_threw = true;
  }
  expect(!explicit_threw && explicit_off == nullptr, "disabled with an explicit sidecar path: still nothing");

  // The control: with image input on, the discovery does reach the file beside the model.
  bool attempted = false;
  try {
    const auto encoder = Encoder::OpenUnlessDisabled(false, model, {}, 5120);
    attempted = encoder != nullptr;
  } catch (...) {
    attempted = true;
  }
  expect(attempted, "enabled: the sidecar beside the model is discovered and opened");

  std::error_code ignored;
  fs::remove_all(root, ignored);
  if (failures == 0)
    std::puts("qwen_vision_disabled_test: OK");
  return failures == 0 ? 0 : 1;
}
