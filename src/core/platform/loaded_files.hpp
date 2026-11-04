#ifndef GUFO_CORE_PLATFORM_LOADED_FILES_HPP_
#define GUFO_CORE_PLATFORM_LOADED_FILES_HPP_

#include <cerrno>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::platform {

struct LoadedFile {
  std::string path;
  std::size_t bytes{0};
  std::string purpose;
};

[[nodiscard]] inline std::string LoadedExecutablePath() {
  const auto text = ExecutablePath().u8string();
  return {text.begin(), text.end()};
}

// Resolve the files already open, including split shards and discovered vision
// sidecars. Re-running discovery could report a file replaced after model load.
inline void AppendLoadedFiles(std::span<const core::GgufMappedRegion> regions,
                              std::string_view purpose,
                              std::vector<LoadedFile>* files) {
  for (const auto& region : regions) {
    if (!FileValid(region.file_descriptor))
      throw std::runtime_error("loaded model file has no inspectable descriptor");
    const auto text = FilePath(region.file_descriptor).u8string();
    files->push_back({std::string(text.begin(), text.end()), region.size,
                      std::string(purpose)});
  }
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_LOADED_FILES_HPP_
