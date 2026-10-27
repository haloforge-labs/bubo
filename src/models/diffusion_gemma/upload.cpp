#include "src/models/diffusion_gemma/upload.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

std::size_t Align(std::size_t x) {
  return (x + 255) & ~std::size_t{255};
}

/// Open descriptors of the files a set of ranges reads, closed on exit.
class OpenFiles {
public:
  explicit OpenFiles(std::span<const FileRange> ranges) {
    for (const auto& range : ranges) {
      if (index_.contains(range.file))
        continue;
      const int fd = ::open(range.file.c_str(), O_RDONLY | O_CLOEXEC);
      struct stat info{};
      if (fd < 0 || ::fstat(fd, &info) != 0) {
        if (fd >= 0)
          ::close(fd);
        throw std::runtime_error("cannot open " + range.file.string());
      }
      index_.emplace(range.file, static_cast<std::uint32_t>(regions_.size()));
      regions_.push_back({.data = nullptr,
                          .size = static_cast<std::size_t>(info.st_size),
                          .file_descriptor = fd});
    }
  }
  ~OpenFiles() {
    for (const auto& region : regions_)
      ::close(region.file_descriptor);
  }
  OpenFiles(const OpenFiles&) = delete;
  OpenFiles& operator=(const OpenFiles&) = delete;

  [[nodiscard]] std::span<const core::GgufMappedRegion> regions() const {
    return regions_;
  }
  [[nodiscard]] std::uint32_t index(const std::filesystem::path& file) const {
    return index_.at(file);
  }

private:
  std::map<std::filesystem::path, std::uint32_t> index_;
  std::vector<core::GgufMappedRegion> regions_;
};

}  // namespace

void UploadFileRanges(std::span<const FileRange> ranges,
                      std::span<void* const> destinations) {
  if (ranges.size() != destinations.size())
    throw std::invalid_argument("UploadFileRanges: one destination per range");
  if (ranges.empty())
    return;
  const OpenFiles files(ranges);
  std::string error;
  auto upload = hip::WeightUpload::Create(files.regions(), &error);
  if (!upload)
    throw std::runtime_error("DiffusionGemma weight upload: " + error);
  // Queue in file order, so that the readers move through each file.
  std::vector<std::size_t> order(ranges.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
    return std::tie(ranges[a].file, ranges[a].offset) <
           std::tie(ranges[b].file, ranges[b].offset);
  });
  for (const std::size_t i : order) {
    const auto& range = ranges[i];
    if (!upload->Copy(files.index(range.file), range.offset, range.bytes,
                      destinations[i], &error))
      break;
  }
  if (!upload->Finish(&error))
    throw std::runtime_error("DiffusionGemma weight upload: " + error);
}

StagedConversion::StagedConversion(std::size_t capacity, hipStream_t stream)
    : capacity_(capacity), stream_(stream) {}

StagedConversion::~StagedConversion() {
  if (buffer_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    (void)hipFree(buffer_);
  }
}

void StagedConversion::Add(std::vector<FileRange> ranges, Convert convert) {
  std::size_t bytes = 0;
  for (const auto& range : ranges)
    bytes = Align(bytes) + range.bytes;
  if (used_ > 0 && Align(used_) + bytes > capacity_)
    Flush();
  if (buffer_ == nullptr || bytes > capacity_) {
    if (buffer_ != nullptr)
      HipCheck(hipFree(buffer_), "conversion staging");
    capacity_ = std::max(capacity_, bytes);
    HipCheck(hipMalloc(&buffer_, capacity_), "conversion staging");
  }
  Pending pending{.offsets = {}, .convert = std::move(convert)};
  for (auto& range : ranges) {
    used_ = Align(used_);
    pending.offsets.push_back(used_);
    destinations_.push_back(buffer_ + used_);
    used_ += range.bytes;
    ranges_.push_back(std::move(range));
  }
  pending_.push_back(std::move(pending));
}

void StagedConversion::Flush() {
  if (pending_.empty())
    return;
  UploadFileRanges(ranges_, destinations_);
  for (const auto& pending : pending_) {
    std::vector<const std::uint8_t*> staged;
    for (const auto offset : pending.offsets)
      staged.push_back(buffer_ + offset);
    pending.convert(staged);
  }
  // The next batch overwrites the staging buffer.
  HipCheck(hipStreamSynchronize(stream_), "weight conversion");
  ranges_.clear();
  destinations_.clear();
  pending_.clear();
  used_ = 0;
}

}  // namespace gufo::models::diffusion_gemma
