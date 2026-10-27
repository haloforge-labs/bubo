// SPDX-License-Identifier: MIT
// Engine/BO sequence adapted from the MIT fork mikealanni/gufo.
// Source contract and Apache-2.0 WITH LLVM-exception AIE attribution:
// NOTICES.md.
#include "private_bo_adapter.hpp"

#include <windows.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "bo_address_table.hpp"
#include "src/core/crypto/sha256.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt_runtime.hpp"

namespace gufo::npu {
namespace {
using Bytes = std::vector<std::byte>;
constexpr char kBinName[] = "final_4096x2560x1024_128x64x128.xclbin";
constexpr char kInstName[] = "insts_4096x2560x1024_128x64x128.txt";
constexpr char kBinHash[] =
    "d31ceb12b0b62dcbb2e32d1ae73f2125b3fe15871b8570a1c89372f3c3fb410e";
constexpr char kInstHash[] =
    "bfec14665c6bc0bda9c8abd2536172b9459a3e234c7890faa96b6c9ecbda8a90";

// Read once per adapter (process environment only), not per dispatch.
bool WeightGuardFromEnvironment() {
  const char* text = std::getenv("GUFO_NPU_WEIGHT_GUARD");
  return text != nullptr && std::string_view(text) == "1";
}
// FNV-1a over the first and last 4 KiB: cheap, catches reuse of an address.
std::uint64_t Fingerprint(std::span<const std::byte> b) {
  constexpr std::size_t kEdge = 4096;
  std::uint64_t hash = 14695981039346656037ULL;
  const auto mix = [&](std::span<const std::byte> part) {
    for (const auto byte : part) {
      hash ^= static_cast<std::uint8_t>(byte);
      hash *= 1099511628211ULL;
    }
  };
  mix(b.first(std::min(kEdge, b.size())));
  mix(b.last(std::min(kEdge, b.size())));
  return hash;
}

void Require(bool valid, const char* message,
             ErrorKind kind = ErrorKind::Validation) {
  if (!valid)
    throw AdapterError(kind, message);
}

std::string Hash(const Bytes& bytes) {
  return crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

// Keep both handles open through parse/register: no path replacement or write
// between byte verification and XRT's file parse. Size is checked before alloc.
struct Artifact {
  HANDLE handle = INVALID_HANDLE_VALUE;
  Bytes bytes;
  Artifact(const std::filesystem::path& path, DWORD expected,
           const char* hash) {
    handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    Require(handle != INVALID_HANDLE_VALUE, "artifact open failed");
    try {
      LARGE_INTEGER size{};
      Require(GetFileSizeEx(handle, &size) && size.QuadPart == expected,
              "artifact byte length differs from pinned contract");
      bytes.resize(expected);
      DWORD count = 0;
      Require(ReadFile(handle, bytes.data(), expected, &count, nullptr) &&
                  count == expected,
              "artifact short read");
      Require(Hash(bytes) == hash,
              "artifact SHA256 differs from pinned contract");
    } catch (...) {
      CloseHandle(handle);
      handle = INVALID_HANDLE_VALUE;
      throw;
    }
  }
  ~Artifact() { CloseHandle(handle); }
};
}  // namespace

AdapterError::AdapterError(ErrorKind kind, std::string text, bool dma_possible,
                           std::optional<int> ert_state,
                           std::optional<int> native_code)
    : std::runtime_error(std::move(text)),
      kind(kind),
      dma_possible(dma_possible),
      ert_state(ert_state),
      native_code(native_code) {}

std::uint32_t InstructionWordCount(std::uint64_t bytes) {
  Require(bytes != 0 && bytes % sizeof(std::uint32_t) == 0,
          "instructions must contain nonzero whole binary uint32 words");
  Require(bytes / sizeof(std::uint32_t) <=
              std::numeric_limits<std::uint32_t>::max(),
          "instruction word count exceeds uint32 ABI");
  return static_cast<std::uint32_t>(bytes / sizeof(std::uint32_t));
}

struct PrivateBoAdapter::Impl {
  std::chrono::milliseconds timeout;
  bool dma_possible = false, failed = false, output_ready = false;
  std::unique_ptr<xrt::device> device;
  std::unique_ptr<xrt::xclbin> bin;
  std::unique_ptr<xrt::hw_context> context;
  std::unique_ptr<xrt::kernel> kernel;
  std::array<std::unique_ptr<xrt::bo>, 6> bos;
  std::array<std::byte*, 6> maps{};
  std::array<std::size_t, 6> sizes{};
  std::unique_ptr<xrt::run> run, heartbeat_run;
  std::unique_ptr<xrt::bo> heartbeat_a, heartbeat_b;
  int weight_group = 0;
  std::array<std::unique_ptr<xrt::bo>, 4> outputs;
  std::array<std::byte*, 4> output_maps{};
  std::array<std::unique_ptr<xrt::run>, 4> runs;

  struct Weight {
    std::unique_ptr<xrt::bo> bo;
    std::uint64_t fingerprint = 0;  // GUFO_NPU_WEIGHT_GUARD only
  };
  // Keyed by the caller's vector data pointer. The key does not say whether
  // the bytes behind it changed: a different weight vector that reuses the
  // address of a freed one is NOT re-sent by ExecuteBatch/SetResidentInputs
  // (PrepareWeight on a known address does re-copy). The guard below is the
  // optional check; default behaviour is unchanged.
  std::map<const std::byte*, Weight> weights;
  // Device address of every BO (log-only, fail-open): published as an
  // immutable snapshot so the HAL monitor thread can resolve a fault address.
  std::vector<BoEntry> table;
  unsigned weight_index = 0;
  void Record(const char* kind, unsigned index, xrt::bo* bo,
              std::uint64_t size) noexcept {
    BoEntry entry;
    entry.kind = kind;
    entry.index = index;
    entry.size = size;
    try {
      if (bo) {
        const std::uint64_t address = bo->address();
        entry.address = address;
        entry.known = address != 0;  // 0 means "no device address": na
      }
    } catch (...) {
    }
    try {
      table.push_back(entry);
    } catch (...) {
    }
  }
  void Publish() noexcept {
    try {
      PublishBoTable(std::make_shared<const BoTable>(BoTable{table}));
    } catch (...) {
    }
  }
  std::uint64_t stale_repairs = 0;
  const bool weight_guard = WeightGuardFromEnvironment();
  explicit Impl(std::chrono::milliseconds timeout) : timeout(timeout) {}
  template<class F>
  auto Call(const std::string&, F&& f) {
    return f();
  }
  void Ready() const {
    Require(!failed && !dma_possible, "adapter failed or dispatch outstanding",
            ErrorKind::State);
  }
  void Sync(std::size_t i, xclBOSyncDirection dir) {
    bos[i]->sync(dir, sizes[i], 0);
  }
  void Fill(std::size_t i, std::span<const std::byte> data) {
    std::memcpy(maps[i], data.data(), data.size());
    Sync(i, XCL_BO_SYNC_BO_TO_DEVICE);
  }
  void Initialize(const std::filesystem::path& dir,
                  std::span<const std::byte> a, std::span<const std::byte> b) {
    static_assert(std::endian::native == std::endian::little);
    Require(a.size() == kABytes && b.size() == kBBytes,
            "A/B byte length mismatch");
    Require(timeout.count() > 0 && timeout.count() <= 8000,
            "completion timeout must be in 1..8000 ms");
    Artifact instructions(dir / kInstName, 22032, kInstHash);
    Artifact artifact(dir / kBinName, 878544, kBinHash);
    const auto words = InstructionWordCount(instructions.bytes.size());
    Call("runtime.verify", [] { EnsureXrtRuntime(); });
    device =
        Call("device.create", [] { return std::make_unique<xrt::device>(0u); });
    bin = Call("xclbin.parse", [&] {
      return std::make_unique<xrt::xclbin>((dir / kBinName).string());
    });
    auto uuid = Call("device.register_xclbin",
                     [&] { return device->register_xclbin(*bin); });
    context = Call("hw_context.create", [&] {
      return std::make_unique<xrt::hw_context>(*device, uuid);
    });
    kernel = Call("kernel.create", [&] {
      return std::make_unique<xrt::kernel>(*context, "MLIR_AIE");
    });
    // Allocate each buffer in the group reported for its kernel slot.
    constexpr std::array<int, 6> slots{1, 3, 4, 5, 6, 7};
    std::array<int, 6> groups{};
    for (std::size_t i = 0; i < slots.size(); ++i) {
      groups[i] = Call("kernel.group_id." + std::to_string(slots[i]),
                       [&] { return kernel->group_id(slots[i]); });
    }
    weight_group = groups[2];
    sizes = {
        instructions.bytes.size(), kABytes, kBBytes, kCBytes, kABytes, kABytes};
    for (std::size_t i = 0; i < bos.size(); ++i) {
      bos[i] = Call("bo.create." + std::to_string(i), [&] {
        return std::make_unique<xrt::bo>(
            *device, sizes[i],
            i == 0 ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only,
            groups[i]);
      });
      const auto size =
          Call("bo.size." + std::to_string(i), [&] { return bos[i]->size(); });
      Require(size == sizes[i], "allocated BO size differs from contract");
      maps[i] = static_cast<std::byte*>(
          Call("bo.map." + std::to_string(i), [&] { return bos[i]->map(); }));
      Require(maps[i] != nullptr, "BO map returned null");
      std::memset(maps[i], 0, sizes[i]);
    }
    Fill(0, instructions.bytes);
    Fill(1, a);
    Fill(2, b);

    for (std::size_t i : {4u, 5u}) {
      std::memset(maps[i], 0, kABytes);
      Sync(i, XCL_BO_SYNC_BO_TO_DEVICE);
    }
    run =
        Call("run.create", [&] { return std::make_unique<xrt::run>(*kernel); });
    Call("run.set_arg.0", [&] { run->set_arg(0, std::uint64_t{3}); });
    Call("run.set_arg.2", [&] { run->set_arg(2, words); });
    for (std::size_t i = 0; i < slots.size(); ++i)
      Call("run.set_arg." + std::to_string(slots[i]),
           [&] { run->set_arg(slots[i], *bos[i]); });
    heartbeat_a = std::make_unique<xrt::bo>(
        *device, kABytes, xrt::bo::flags::host_only, groups[1]);
    heartbeat_b = std::make_unique<xrt::bo>(
        *device, kBBytes, xrt::bo::flags::host_only, groups[2]);
    Require(heartbeat_a->size() == kABytes && heartbeat_b->size() == kBBytes,
            "heartbeat BO size mismatch");
    auto* ha = heartbeat_a->map();
    auto* hb = heartbeat_b->map();
    Require(ha && hb, "heartbeat map failed");
    std::memset(ha, 0, kABytes);
    std::memset(hb, 0, kBBytes);
    heartbeat_a->sync(XCL_BO_SYNC_BO_TO_DEVICE, kABytes, 0);
    heartbeat_b->sync(XCL_BO_SYNC_BO_TO_DEVICE, kBBytes, 0);
    heartbeat_run = std::make_unique<xrt::run>(*kernel);
    heartbeat_run->set_arg(0, std::uint64_t{3});
    heartbeat_run->set_arg(2, words);
    for (std::size_t j = 0; j < slots.size(); ++j)
      heartbeat_run->set_arg(slots[j], *bos[j]);
    heartbeat_run->set_arg(3, *heartbeat_a);
    heartbeat_run->set_arg(4, *heartbeat_b);
    for (std::size_t i = 0; i < runs.size(); ++i) {
      outputs[i] = std::make_unique<xrt::bo>(
          *device, kCBytes, xrt::bo::flags::host_only, groups[3]);
      Require(outputs[i]->size() == kCBytes, "C BO size mismatch");
      output_maps[i] = static_cast<std::byte*>(outputs[i]->map());
      Require(output_maps[i] != nullptr, "C BO map failed");
      runs[i] = std::make_unique<xrt::run>(*kernel);
      runs[i]->set_arg(0, std::uint64_t{3});
      runs[i]->set_arg(2, words);
      for (std::size_t j = 0; j < slots.size(); ++j)
        runs[i]->set_arg(slots[j], *bos[j]);
      runs[i]->set_arg(5, *outputs[i]);
    }
    // BO address table: every fixed BO, once.
    static constexpr const char* kinds[6] = {"instr", "a",       "b",
                                             "c",     "scratch", "scratch"};
    static constexpr unsigned indexes[6] = {0, 0, 0, 0, 0, 1};
    for (std::size_t i = 0; i < bos.size(); ++i)
      Record(kinds[i], indexes[i], bos[i].get(), sizes[i]);
    Record("heartbeat_a", 0, heartbeat_a.get(), kABytes);
    Record("heartbeat_b", 0, heartbeat_b.get(), kBBytes);
    for (std::size_t i = 0; i < outputs.size(); ++i)
      Record("output", static_cast<unsigned>(i), outputs[i].get(), kCBytes);
    Publish();
  }
};

PrivateBoAdapter::PrivateBoAdapter() = default;

std::unique_ptr<PrivateBoAdapter> PrivateBoAdapter::Create(
    const core::ExperimentalAuthorization& authorization,
    const std::filesystem::path& directory, std::span<const std::byte> a,
    std::span<const std::byte> b, std::chrono::milliseconds timeout) {
  authorization.Require(core::ExperimentalFeature::Prefill);
  auto result = std::unique_ptr<PrivateBoAdapter>(new PrivateBoAdapter);
  result->impl_ = std::make_unique<Impl>(timeout);
  result->impl_->Initialize(directory, a, b);
  return result;
}
PrivateBoAdapter::~PrivateBoAdapter() {
  // Like the fork: retain all pages and run objects until process exit after
  // an uncertain submission. Never clear pending DMA ownership on an error.
  if (impl_ && impl_->dma_possible)
    (void)impl_.release();
}
void PrivateBoAdapter::Heartbeat() {
  impl_->Ready();
  impl_->dma_possible = true;
  impl_->heartbeat_run->start();
  const auto state = impl_->heartbeat_run->wait(impl_->timeout);
  if (state != ERT_CMD_STATE_COMPLETED) {
    impl_->failed = true;
    throw AdapterError(state == ERT_CMD_STATE_TIMEOUT ? ErrorKind::Timeout
                                                      : ErrorKind::Completion,
                       "XRT heartbeat did not complete", true,
                       static_cast<int>(state));
  }
  impl_->dma_possible = false;
}
void PrivateBoAdapter::PrepareWeight(std::span<const std::byte> b) {
  impl_->Ready();
  Require(b.size() == kBBytes, "B byte length mismatch");
  if (auto it = impl_->weights.find(b.data()); it != impl_->weights.end()) {
    std::memcpy(it->second.bo->map(), b.data(), b.size());
    it->second.bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, kBBytes, 0);
    if (impl_->weight_guard)
      it->second.fingerprint = Fingerprint(b);
    return;
  }
  auto bo = std::make_unique<xrt::bo>(
      *impl_->device, kBBytes, xrt::bo::flags::host_only, impl_->weight_group);
  Require(bo->size() == kBBytes, "weight BO size mismatch");
  auto* host = bo->map();
  Require(host != nullptr, "weight BO map failed");
  std::memcpy(host, b.data(), b.size());
  bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, kBBytes, 0);
  impl_->Record("weight", impl_->weight_index++, bo.get(), kBBytes);
  impl_->Publish();  // also covers a weight BO created lazily after load
  impl_->weights.emplace(
      b.data(),
      Impl::Weight{std::move(bo), impl_->weight_guard ? Fingerprint(b) : 0});
}
// Guard: a known address whose bytes changed is re-sent (PrepareWeight copies
// into the existing buffer). Off by default.
void PrivateBoAdapter::RefreshIfStale(std::span<const std::byte> b) {
  if (!impl_->weight_guard)
    return;
  const auto it = impl_->weights.find(b.data());
  if (it == impl_->weights.end() || it->second.fingerprint == Fingerprint(b))
    return;
  ++impl_->stale_repairs;
  PrepareWeight(b);
}
std::uint64_t PrivateBoAdapter::StaleWeightRepairs() const noexcept {
  return impl_->stale_repairs;
}
void PrivateBoAdapter::ExecuteBatch(
    std::span<const std::byte> a,
    const std::vector<std::vector<std::uint8_t>>& weights,
    const std::function<void(std::size_t, std::span<const std::byte>)>&
        consume) {
  impl_->Ready();
  Require(a.size() == kABytes && !weights.empty() && weights.size() <= 12 &&
              bool(consume),
          "batch input geometry mismatch");
  for (const auto& b : weights) {
    Require(b.size() == kBBytes, "B byte length mismatch");
    if (!impl_->weights.contains(reinterpret_cast<const std::byte*>(b.data())))
      PrepareWeight(std::as_bytes(std::span(b)));
    else
      RefreshIfStale(std::as_bytes(std::span(b)));
  }
  impl_->Fill(1, a);  // once for all slices, exactly as the fork's Process()
  std::size_t launched = 0, reaped = 0;
  const auto reap = [&] {
    const std::size_t slice = reaped;  // own the id before advancing the queue
    const auto slot = slice % 4;
    const auto state = impl_->runs[slot]->wait(impl_->timeout);
    if (state != ERT_CMD_STATE_COMPLETED) {
      impl_->failed = true;
      throw AdapterError(state == ERT_CMD_STATE_TIMEOUT ? ErrorKind::Timeout
                                                        : ErrorKind::Completion,
                         "XRT batch did not complete", true,
                         static_cast<int>(state));
    }
    ++reaped;
    impl_->dma_possible = reaped != launched;
    impl_->outputs[slot]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, kCBytes, 0);
    consume(slice, {impl_->output_maps[slot], kCBytes});
  };
  for (std::size_t s = 0; s < weights.size(); ++s) {
    if (launched - reaped == 4)
      reap();
    auto& run = *impl_->runs[s % 4];
    run.set_arg(4,
                *impl_->weights
                     .at(reinterpret_cast<const std::byte*>(weights[s].data()))
                     .bo);
    impl_->dma_possible =
        true;  // an exception from start can follow submission
    run.start();
    ++launched;
  }
  while (reaped != launched)
    reap();
}
void PrivateBoAdapter::SetInputs(std::span<const std::byte> a,
                                 std::span<const std::byte> b) {
  impl_->Ready();
  Require(a.size() == kABytes && b.size() == kBBytes,
          "A/B byte length mismatch");
  impl_->Fill(1, a);
  impl_->Fill(2, b);
  impl_->run->set_arg(4, *impl_->bos[2]);
}
void PrivateBoAdapter::SetResidentInputs(std::span<const std::byte> a,
                                         std::span<const std::byte> b,
                                         bool copy_a) {
  impl_->Ready();
  Require(a.size() == kABytes && b.size() == kBBytes,
          "A/B byte length mismatch");
  if (!impl_->weights.contains(b.data()))
    PrepareWeight(b);
  else
    RefreshIfStale(b);
  if (copy_a)
    impl_->Fill(1, a);
  impl_->run->set_arg(4, *impl_->weights.at(b.data()).bo);
}
void PrivateBoAdapter::Submit() {
  impl_->Ready();
  impl_->output_ready = false;
  impl_->dma_possible = true;
  impl_->run->start();
}
void PrivateBoAdapter::Finish() {
  Require(impl_->dma_possible && !impl_->failed,
          "no healthy outstanding dispatch", ErrorKind::State);
  const auto state = impl_->run->wait(impl_->timeout);
  if (state != ERT_CMD_STATE_COMPLETED) {
    impl_->failed = true;
    throw AdapterError(state == ERT_CMD_STATE_TIMEOUT ? ErrorKind::Timeout
                                                      : ErrorKind::Completion,
                       "XRT did not confirm successful completion", true,
                       static_cast<int>(state));
  }
  impl_->dma_possible = false;
  impl_->Sync(3, XCL_BO_SYNC_BO_FROM_DEVICE);
  impl_->output_ready = true;
}
void PrivateBoAdapter::CopyOutput(std::span<std::byte> output) {
  impl_->Ready();
  Require(impl_->output_ready && output.size() == kCBytes,
          "output not completed or invalid size", ErrorKind::State);
  std::memcpy(output.data(), impl_->maps[3], kCBytes);
}
bool PrivateBoAdapter::DmaPossible() const noexcept {
  return impl_->dma_possible;
}
bool PrivateBoAdapter::Failed() const noexcept {
  return impl_->failed;
}
std::vector<const void*> PrivateBoAdapter::SamplePages() {
  constexpr std::size_t kPage = 4096;
  std::vector<const void*> pages;
  if (impl_->weights.empty())
    return pages;  // weights are prepared after start-up; try again later
  const auto add = [&](const std::byte* base, std::size_t size,
                       std::size_t where) {
    if (base && size > kPage)
      pages.push_back(base + ((where % size) & ~(kPage - 1)));
  };
  add(impl_->maps[1], kABytes, 0);
  add(impl_->maps[1], kABytes, kABytes / 2);
  add(impl_->output_maps[0], kCBytes, 0);
  add(impl_->output_maps[0], kCBytes, kCBytes / 2);
  const std::size_t count = impl_->weights.size();
  const std::size_t picks = std::min<std::size_t>(4, count);
  std::size_t index = 0, next = 0;
  for (auto& entry : impl_->weights) {
    const std::size_t wanted = picks > 1 ? next * (count - 1) / (picks - 1) : 0;
    if (index == wanted && next < picks) {
      add(static_cast<const std::byte*>(entry.second.bo->map()), kBBytes,
          kBBytes / 2);
      ++next;
    }
    ++index;
  }
  return pages;
}
std::string PrivateBoAdapter::PowerMode() {
  return impl_->device->get_info<xrt::info::device::platform>();
}
}  // namespace gufo::npu
