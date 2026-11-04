#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "src/core/platform/env.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/publish.hpp"
#include "src/models/deepseek_v4_flash/runtime/model.h"
#include "tests/support/temporary_directory.hpp"

namespace {
void Integer(std::ostream& out, std::uint64_t value, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i)
    out.put(static_cast<char>((value >> (i * 8)) & 255));
}

void String(std::ostream& out, std::string_view text) {
  Integer(out, text.size(), 8);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

}  // namespace

int main() {
  TemporaryDirectory directory("gufo-ds4-load");
  const auto& root = directory.path();
  const auto path = root / "invalid.gguf";
  const auto lock_path = root / "instance.lock";
  assert(gufo::platform::SetEnv("DS4_LOCK_FILE", lock_path.string()));
  auto descriptors = gufo::platform::OpenHandleCount();
  for (int fixture = 0; fixture < 3; ++fixture) {
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (fixture == 0) {
        out << "truncated";
      } else {
        Integer(out, 0x46554747, 4);
        Integer(out, 3, 4);
        Integer(out, 0, 8);  // No tensors.
        Integer(out, fixture == 1 ? 0 : 2, 8);
        if (fixture == 2) {
          String(out, "tokenizer.ggml.tokens");
          Integer(out, 9, 4);  // Array.
          Integer(out, 8, 4);  // Strings.
          Integer(out, 1, 8);
          String(out, "hello");  // Required special tokens are missing.
          String(out, "tokenizer.ggml.merges");
          Integer(out, 9, 4);
          Integer(out, 8, 4);
          Integer(out, 0, 8);
        }
        while (out.tellp() < 32 || out.tellp() % 32)
          out.put(0);
      }
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
      ds4_engine* engine = nullptr;
      const std::string path_text = path.string();
      const ds4_engine_options options{path_text.c_str(), nullptr};
      assert(ds4_engine_open(&engine, &options) != 0);
      assert(engine == nullptr);
      // Failure must release both mapped files and the process lock.
      bool contended = false;
      auto lock = gufo::platform::FileLock::TryAcquireProcessLock(
          lock_path, &contended);
      assert(lock.valid() && !contended);
      // A separately held lock must cause another ordinary load failure.
      assert(ds4_engine_open(&engine, &options) != 0 && engine == nullptr);
      lock.Release();
      // Process handles include runtime state the first load creates once
      // (threads, events); measure growth from the first attempt instead.
      if (fixture == 0 && repeat == 0)
        descriptors = gufo::platform::OpenHandleCount();
      assert(gufo::platform::OpenHandleCount() == descriptors);
    }
  }
}
