#if defined(ENGINE_ENABLE_HIP)
#include <sys/mman.h>
#ifdef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      (void)munmap(region.host_copy, region.size);
    } else if (region.device_data != nullptr) {
      // Uploaded to device memory (UploadDeviceRegions).
      (void)hipFree(region.device_data);
    }
    region = {};
  }
}

void CopyMappedWeights(const core::GgufMappedRegion& source, void* copy) {
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto copy_chunk = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      const auto bytes = std::min(kChunkBytes, source.size - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
      if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      std::memcpy(static_cast<std::uint8_t*>(copy) + offset, input, bytes);
      // The immutable copy now owns these resident bytes. Keep the original
      // mapping valid for the reader without retaining its populated PTEs.
      (void)madvise(input, bytes, MADV_DONTNEED);
    }
  };
  {
    std::vector<std::jthread> workers;
    for (std::size_t i = 1; i < std::min<std::size_t>(16, chunks); ++i)
      workers.emplace_back(copy_chunk);
    copy_chunk();
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read mapped Qwen weights");
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* host_copy = mmap(nullptr, source.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED)
    return hipErrorOutOfMemory;
  const auto unmap = [&](void* pointer) { (void)munmap(pointer, source.size); };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  (void)madvise(host_copy, source.size, MADV_HUGEPAGE);
  CopyMappedWeights(source, host_copy);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

#ifdef _WIN32
// Windows: copy each shard into device memory, which is the dedicated VRAM
// carve-out. A registered host copy lands in shared GPU memory instead, which
// WDDM caps at half of system RAM (~16 GiB beside a 96 GiB carve-out): the
// 16.7 GiB UD-Q4_K_XL target does not fit, and the load stalls once the GPU
// touches it. Kernels read the same bytes either way.
//
// The shards are read here rather than through core's WeightUpload, which
// opens O_DIRECT: NTFS serializes O_DIRECT reads of a file that is also
// mapped (16 readers managed 0.03 GiB/s each). A cached overlapped handle
// (O_CONCURRENT_RANDOM, compat/win32/posix.cpp) keeps full speed.
[[nodiscard]] bool UploadDeviceRegions(
    std::span<const core::GgufMappedRegion> source_regions,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  constexpr std::size_t kTailMargin = 4096;
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  constexpr std::size_t kReaders = 16;
  const auto started = std::chrono::steady_clock::now();
  std::mutex error_mutex;
  std::string error;
  std::atomic<bool> failed{false};
  const auto fail = [&](std::string message) {
    std::lock_guard lock(error_mutex);
    if (!failed.exchange(true))
      error = std::move(message);
  };

  weight_regions.resize(source_regions.size());
  std::vector<int> fds(source_regions.size(), -1);
  struct Chunk {
    std::size_t shard;
    std::size_t offset;
  };
  std::vector<Chunk> chunks;
  std::size_t total_bytes = 0;
  for (std::size_t i = 0; i < source_regions.size() && !failed; ++i) {
    const auto& source = source_regions[i];
    void* device = nullptr;
    if (hipMalloc(&device, source.size + kTailMargin) != hipSuccess) {
      fail("hipMalloc failed for GGUF shard " + std::to_string(i) + " (" +
           std::to_string(source.size) + " bytes)");
      break;
    }
    weight_regions[i] = {.host_data = source.data,
                         .device_data = device,
                         .host_copy = nullptr,
                         .size = source.size};
    (void)hipMemset(static_cast<std::uint8_t*>(device) + source.size, 0,
                    kTailMargin);
    const auto path = "/proc/self/fd/" + std::to_string(source.file_descriptor);
    fds[i] = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_CONCURRENT_RANDOM);
    if (fds[i] < 0) {
      fail("cannot reopen GGUF shard " + std::to_string(i) + ": " +
           std::strerror(errno));
      break;
    }
    for (std::size_t offset = 0; offset < source.size; offset += kChunkBytes)
      chunks.push_back({i, offset});
    total_bytes += source.size;
  }

  std::atomic<std::size_t> next{0};
  const auto reader = [&] {
    void* staging = nullptr;
    hipStream_t stream = nullptr;
    if (hipHostMalloc(&staging, kChunkBytes) != hipSuccess ||
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
      fail("weight upload staging allocation failed");
    }
    while (!failed) {
      const auto index = next.fetch_add(1);
      if (index >= chunks.size())
        break;
      const auto [shard, offset] = chunks[index];
      const auto bytes =
          std::min(kChunkBytes, source_regions[shard].size - offset);
      for (std::size_t got = 0; got < bytes && !failed;) {
        const auto n = ::pread(fds[shard], static_cast<char*>(staging) + got,
                               bytes - got, static_cast<off_t>(offset + got));
        if (n <= 0) {
          fail("GGUF shard read failed: " +
               std::string(n == 0 ? "unexpected end of file"
                                  : std::strerror(errno)));
          break;
        }
        got += static_cast<std::size_t>(n);
      }
      if (failed)
        break;
      auto* destination =
          static_cast<std::uint8_t*>(weight_regions[shard].device_data) +
          offset;
      auto status = hipMemcpyAsync(destination, staging, bytes,
                                   hipMemcpyHostToDevice, stream);
      if (status == hipSuccess)
        status = hipStreamSynchronize(stream);
      if (status != hipSuccess)
        fail("weight upload failed: " + std::string(hipGetErrorString(status)));
    }
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
    if (staging != nullptr)
      (void)hipHostFree(staging);
  };
  if (!failed) {
    std::vector<std::jthread> readers;
    for (std::size_t i = 1; i < kReaders; ++i)
      readers.emplace_back(reader);
    reader();
  }
  for (const int fd : fds) {
    if (fd >= 0)
      (void)::close(fd);
  }
  if (failed) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr)
      *error_msg = "Qwen weight upload failed: " + error;
    return false;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const double gib = static_cast<double>(total_bytes) / (1ULL << 30);
  std::fprintf(stderr, "qwen weight upload: %.1f GiB in %.1f s (%.2f GiB/s)\n",
               gib, seconds, seconds > 0 ? gib / seconds : 0.0);
  return true;
}
#endif

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }
#ifdef _WIN32
  if (std::ranges::all_of(source_regions, [](const auto& region) {
        return region.file_descriptor >= 0;
      })) {
    return UploadDeviceRegions(source_regions, weight_regions, error_msg);
  }
#endif

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      map_error = MapRegisteredRegion(source, destination);
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible through mapped registration: " +
                     hipGetErrorString(map_error);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer)
    return nullptr;

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  return std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision));
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
