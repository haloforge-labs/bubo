#ifndef GUFO_CORE_PLATFORM_LOADED_FILES_HPP_
#define GUFO_CORE_PLATFORM_LOADED_FILES_HPP_

#include <cerrno>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "compat/win32/descriptor_path.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::platform {

struct LoadedFile {
  std::string path;
  std::size_t bytes{0};
  std::string purpose;
};

[[nodiscard]] inline std::string LoadedExecutablePath() {
  std::vector<char> buffer(1024);
  while (true) {
    const auto count = gufo_executable_path(buffer.data(), buffer.size());
    if (count >= 0 && static_cast<std::size_t>(count) < buffer.size())
      return std::string(buffer.data(), static_cast<std::size_t>(count));
    if (count < 0 && errno != ERANGE)
      throw std::runtime_error("cannot resolve the running executable path");
    if (buffer.size() >= 1024 * 1024)
      throw std::runtime_error("running executable path exceeds diagnostic limit");
    buffer.resize(buffer.size() * 2);
  }
}

// Resolve the files already open, including split shards and discovered vision
// sidecars. Re-running discovery could report a file replaced after model load.
inline void AppendLoadedFiles(std::span<const core::GgufMappedRegion> regions,
                              std::string_view purpose,
                              std::vector<LoadedFile>* files) {
  for (const auto& region : regions) {
    if (region.file_descriptor < 0)
      throw std::runtime_error("loaded model file has no inspectable descriptor");
    std::vector<char> buffer(1024);
    while (true) {
      const auto count = gufo_descriptor_path(region.file_descriptor,
                                             buffer.data(), buffer.size());
      if (count >= 0 && static_cast<std::size_t>(count) < buffer.size()) {
        files->push_back({std::string(buffer.data(), static_cast<std::size_t>(count)),
                          region.size, std::string(purpose)});
        break;
      }
      if (count < 0 && errno != ERANGE)
        throw std::runtime_error("cannot resolve loaded model descriptor path");
      if (buffer.size() >= 1024 * 1024)
        throw std::runtime_error("loaded model descriptor path exceeds diagnostic limit");
      buffer.resize(buffer.size() * 2);
    }
  }
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_LOADED_FILES_HPP_
