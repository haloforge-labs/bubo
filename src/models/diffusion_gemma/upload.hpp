#ifndef GUFO_MODELS_DIFFUSION_GEMMA_UPLOAD_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_UPLOAD_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <vector>

namespace gufo::models::diffusion_gemma {

/// A byte range of a checkpoint file.
struct FileRange {
  std::filesystem::path file;
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
};

/// Copies each range to its device destination through Gufo's weight upload
/// pipeline (parallel direct reads into pinned staging) and waits for all
/// of them. Throws std::runtime_error on failure.
void UploadFileRanges(std::span<const FileRange> ranges,
                      std::span<void* const> destinations);

/// Converts checkpoint tensors on the device. Each Add stages its ranges in
/// a device buffer; ranges are uploaded in batches that fill the buffer, and
/// after a batch is on the device each of its `convert` callbacks runs on
/// `stream` with the device copies of its ranges. Call Flush to finish
/// everything queued.
class StagedConversion {
public:
  using Convert = std::function<void(std::span<const std::uint8_t* const>)>;

  /// `capacity` is raised to the largest single Add.
  StagedConversion(std::size_t capacity, hipStream_t stream);
  ~StagedConversion();
  StagedConversion(const StagedConversion&) = delete;
  StagedConversion& operator=(const StagedConversion&) = delete;

  void Add(std::vector<FileRange> ranges, Convert convert);
  void Flush();

private:
  struct Pending {
    std::vector<std::size_t> offsets;
    Convert convert;
  };

  std::size_t capacity_;
  hipStream_t stream_;
  std::uint8_t* buffer_{nullptr};
  std::size_t used_{0};
  std::vector<FileRange> ranges_;
  std::vector<void*> destinations_;
  std::vector<Pending> pending_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_UPLOAD_HPP_
