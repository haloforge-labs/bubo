#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DECODE_IQ_ROUTING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DECODE_IQ_ROUTING_HPP_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <utility>

namespace gufo::models::qwen38_flash_next::rocm {

// Diagnostic only: each line is one real trunk router invocation. Keeping the
// original slot order, duplicates, and negative IDs makes the file suitable
// for replaying the actual verification routing mix.
class DecodeIqRoutingCapture {
public:
  static constexpr std::uint32_t kMaxSamplesPerLayerWidth = 64;

  explicit DecodeIqRoutingCapture(std::string path) : path_(std::move(path)) {}

  static DecodeIqRoutingCapture& Instance() {
    static DecodeIqRoutingCapture capture([] {
      const char* path = std::getenv("GUFO_DECODE_IQ_ROUTING_PATH");
      return path == nullptr ? std::string{} : std::string(path);
    }());
    return capture;
  }

  [[nodiscard]] bool enabled() const noexcept { return !path_.empty(); }

  [[nodiscard]] bool Wants(std::uint32_t layer, std::uint32_t width) const {
    if (!enabled() || width < 1 || width > 8)
      return false;
    std::lock_guard lock(mutex_);
    const auto it = samples_.find({layer, width});
    return it == samples_.end() || it->second < kMaxSamplesPerLayerWidth;
  }

  bool Append(std::uint32_t layer, std::uint32_t width,
              std::span<const std::int32_t> ids, std::string* error) {
    if (path_.empty() || width < 1 || width > 8 || ids.size() != width * 10) {
      if (error)
        *error = "decode IQ routing: invalid path or routing shape";
      return false;
    }
    std::lock_guard lock(mutex_);
    auto& samples = samples_[{layer, width}];
    if (samples >= kMaxSamplesPerLayerWidth)
      return true;

    std::string line = std::to_string(layer) + " " + std::to_string(width);
    for (const auto id : ids)
      line += " " + std::to_string(id);
    line += '\n';

    // The first sample starts a fresh, process-bounded capture file.
    std::FILE* file = std::fopen(path_.c_str(), started_ ? "ab" : "wb");
    if (file == nullptr) {
      if (error)
        *error = "decode IQ routing: cannot open " + path_;
      return false;
    }
    const bool written =
        std::fwrite(line.data(), 1, line.size(), file) == line.size();
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed) {
      if (error)
        *error = "decode IQ routing: cannot write " + path_;
      return false;
    }
    started_ = true;
    ++samples;
    return true;
  }

private:
  const std::string path_;
  mutable std::mutex mutex_;
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> samples_;
  bool started_{false};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DECODE_IQ_ROUTING_HPP_
