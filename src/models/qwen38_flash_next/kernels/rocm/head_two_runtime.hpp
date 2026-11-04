#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "device_model.hpp"
#include "head_two_kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

class HeadTwoRuntime {
public:
  ~HeadTwoRuntime();
  static bool Create(const DeviceTensor& output,
                     const std::filesystem::path& cheap_path,
                     std::unique_ptr<HeadTwoRuntime>& result,
                     std::string* error, int bench_shortlist = 0);
  bool enabled() const { return cheap_ != nullptr; }
  HeadTwoReason unavailable_reason() const { return unavailable_; }
  bool PrepareExtras(std::span<const std::uint32_t> ids, hipStream_t stream,
                     std::string* error) const;
  bool Enqueue(const void* q8, float* full, int width, hipStream_t stream,
               std::string* error, int row_rotation = 0) const;
  bool SaveFrontier(void* destination, int row, hipStream_t stream) const;
  const float* RebuildFrontier(const void* saved, int width, int row,
                               hipStream_t stream, std::string* error) const;
  void Row(int row, float* logits) const;
  MtpCandidateLogits Candidates(int row) const;
  void Record(int width, HeadTwoReason reason) const;
  std::size_t q8_row_bytes() const { return q8_row_bytes_; }
  int shortlist() const { return shortlist_; }
  std::uint64_t weight_bytes() const { return weight_bytes_; }
  std::size_t device_bytes() const { return device_bytes_; }
  std::size_t host_bytes() const {
    return host_ ? kHeadMaxWidth * sizeof(HeadTwoResult) +
                       kHeadExtras * sizeof(std::int32_t)
                 : 0;
  }

private:
  HeadTwoRuntime() = default;
  bool Allocate(void** p, std::size_t bytes, std::string* error);
  int shortlist_{128}, rows_{0}, cols_{0};
  int rescore_{kHeadCapacity};
  HeadTwoReason unavailable_{HeadTwoReason::kUnavailable};
  bool check_{false}, report_{true};
  const void* exact_{nullptr};
  void* cheap_{nullptr};
  void* q8_{nullptr};
  void* gathered_{nullptr};
  float* cross_{nullptr};
  std::size_t q8_row_bytes_{0};
  std::uint64_t weight_bytes_{0};
  std::size_t device_bytes_{0};
  HeadTwoBuffers buffers_{};
  HeadTwoResult* host_{nullptr};
  std::int32_t* extras_host_{nullptr};
  std::vector<void*> allocations_;
  mutable std::array<std::uint64_t, static_cast<int>(HeadTwoReason::kCount)>
      counts_{};
  mutable std::uint64_t checked_{0}, mismatches_{0}, missed_{0};
};
}  // namespace gufo::models::qwen38_flash_next::rocm
