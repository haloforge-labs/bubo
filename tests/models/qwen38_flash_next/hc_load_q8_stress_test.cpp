// Stress for the host side of GUFO_HC_LOAD_Q8, written after an intermittent
// draft collapse seen on the GPU in one ON launch of three (decode-hcq8
// review). It asks the questions a GPU-free machine can answer:
//   1. is the image uploaded for a converted tensor (blocks AND the 4096-byte
//      zero tail) bit-identical for every thread count, every repetition, and
//      with a buffer that holds poison from the previous use?
//   2. does that stay true while 16 reader threads read the same file with
//      unbuffered I/O (what the weight stager does) and the converter reads the
//      memory-mapped view of it?
//   3. do the unbuffered reads (the stager's path, used for every pass-through
//      tensor such as the MTP sidecar's Q8_0 mixers) return exactly the bytes of
//      the mapped view, every time?
// `--real <gguf>...` uses full HC tensors of native files, `--mtp <gguf>` the
// tensors of the MTP sidecar, `--reps N` repetitions per configuration
// (default 6), `--readers N` hammer threads (default 16).
#include "src/models/qwen38_flash_next/hc_load_q8_policy.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/bf16_q8_0.hpp"

namespace p = gufo::models::qwen38_flash_next::hc_load_q8;
using gufo::core::GgmlType;

namespace {

constexpr std::size_t kTail = 4096;
constexpr std::uint8_t kPoison = 0xA5;

void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}

std::uint16_t Encode(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}

const unsigned kThreadCounts[] = {1, 2, 3, 4, 5, 7, 8, 16, 64};

/// One converted-tensor case: the reference image (serial) and every thread
/// count x repetition against it, with the image buffer poisoned before each use.
void StressTensor(const std::string& label, const void* bf16, std::size_t rows,
                  std::size_t cols, bool inject, int reps) {
  std::vector<std::uint8_t> reference;
  std::string error;
  Require(inject ? gufo::quant::BuildBf16F32Image(bf16, rows * cols, kTail, reference, &error)
                 : gufo::quant::BuildBf16Q8_0Image(bf16, rows, cols, kTail, 1, reference, &error),
          label + ": " + error);
  const std::size_t payload = reference.size() - kTail;
  for (std::size_t i = payload; i < reference.size(); ++i)
    Require(reference[i] == 0, label + ": reference tail is not zero");
  std::vector<std::uint8_t> image;
  std::size_t runs = 0;
  for (const unsigned threads : kThreadCounts) {
    for (int rep = 0; rep < reps; ++rep) {
      image.assign(reference.size(), kPoison);  // every byte must be rewritten
      Require(inject ? gufo::quant::BuildBf16F32Image(bf16, rows * cols, kTail, image, &error)
                     : gufo::quant::BuildBf16Q8_0Image(bf16, rows, cols, kTail, threads, image, &error),
              label + ": " + error);
      Require(image.size() == reference.size(), label + ": image size changed");
      if (std::memcmp(image.data(), reference.data(), reference.size()) != 0) {
        std::size_t at = 0;
        while (image[at] == reference[at])
          ++at;
        throw std::runtime_error(label + ": image differs from the serial reference at byte " +
                                 std::to_string(at) + " of " + std::to_string(reference.size()) +
                                 " (threads=" + std::to_string(threads) + ", rep=" +
                                 std::to_string(rep) + (at >= payload ? ", in the tail" : "") + ")");
      }
      ++runs;
    }
  }
  (void)runs;
}

void SyntheticStress(int reps) {
  std::mt19937_64 rng(0xBADC0DE);
  std::normal_distribution<float> normal(0.0F, 0.2F);
  const struct {
    std::size_t rows, cols;
    bool inject;
  } shapes[] = {{320, 10240, false}, {10240, 320, false}, {4, 10240, true}};
  for (const auto& s : shapes) {
    std::vector<std::uint16_t> bf16(s.rows * s.cols);
    for (std::size_t b = 0; b < bf16.size() / 32; ++b) {
      const float scale = std::pow(10.0F, float(rng() % 500) / 100.0F - 3.0F);
      const bool zero = rng() % 61 == 0;
      for (std::size_t i = 0; i < 32; ++i)
        bf16[b * 32 + i] = zero ? 0 : Encode(normal(rng) * scale);
    }
    StressTensor("synthetic " + std::to_string(s.cols) + "x" + std::to_string(s.rows), bf16.data(),
                 s.rows, s.cols, s.inject, reps);
  }
  std::cout << "  synthetic: 3 shapes x " << std::size(kThreadCounts) << " thread counts x " << reps
            << " repetitions identical to the serial image, tail zero, poisoned buffers rewritten\n";
}

/// An unbuffered reader of one file (what the stager does per reader slot).
struct Direct {
  gufo::platform::PlatformFile fd{gufo::platform::kInvalidFile};
  ~Direct() { gufo::platform::CloseFile(fd); }
};

constexpr std::size_t kAlign = 4096;

/// Reads [offset, offset+size) through the unbuffered handle exactly like
/// WeightUpload::Transfer (aligned window, then the payload inside it).
bool DirectRead(gufo::platform::PlatformFile fd, std::uint64_t offset, std::size_t size,
                std::vector<std::uint8_t>& scratch, const std::uint8_t*& payload) {
  const std::uint64_t begin = offset & ~std::uint64_t{kAlign - 1};
  const std::size_t skip = static_cast<std::size_t>(offset - begin);
  const std::size_t length = (skip + size + kAlign - 1) & ~(kAlign - 1);
  scratch.resize(length + kAlign);
  auto* aligned = reinterpret_cast<std::uint8_t*>(
      (reinterpret_cast<std::uintptr_t>(scratch.data()) + kAlign - 1) & ~std::uintptr_t{kAlign - 1});
  std::size_t got = 0;
  while (got < skip + size) {
    std::error_code ec;
    const auto n = gufo::platform::ReadAt(fd, begin + got, aligned + got, length - got, ec);
    if (n == 0)
      return false;
    got += n;
  }
  payload = aligned + skip;
  return true;
}

struct FileCase {
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string path;
};

FileCase Open(const std::string& path) {
  FileCase c;
  std::string error;
  c.reader = gufo::core::GgufReader::OpenHeaderFile(path, &error);
  Require(c.reader != nullptr, path + ": " + error);
  c.path = path;
  return c;
}

bool IsHcTensor(const std::string& name) {
  return (name.find("hc_") != std::string::npos || name.rfind("output_hc", 0) == 0) &&
         (name.find("_down.weight") != std::string::npos || name.find("_up.weight") != std::string::npos ||
          name.find("_inject.weight") != std::string::npos);
}

/// Hammer: `count` threads reading random 1 MiB windows of the file with
/// unbuffered I/O, comparing every window with the mapped view.
class Hammer {
public:
  Hammer(const FileCase& file, int count) : file_(file) {
    const auto regions = file.reader->GetMappedRegions();
    base_ = static_cast<const std::uint8_t*>(regions[0].data);
    size_ = regions[0].size;
    for (int i = 0; i < count; ++i) {
      auto d = std::make_unique<Direct>();
      auto dup = gufo::platform::DuplicateFile(regions[0].file_descriptor);
      d->fd = gufo::platform::ReopenUnbuffered(dup);
      gufo::platform::CloseFile(dup);
      if (!gufo::platform::FileValid(d->fd)) {
        unbuffered_ = false;
        break;
      }
      handles_.push_back(std::move(d));
    }
    for (auto& h : handles_)
      threads_.emplace_back([this, &h, seed = reinterpret_cast<std::uintptr_t>(h.get())] { Run(*h, seed); });
  }
  ~Hammer() { Stop(); }
  void Stop() {
    stop_ = true;
    for (auto& t : threads_)
      if (t.joinable())
        t.join();
    threads_.clear();
  }
  bool unbuffered() const { return unbuffered_ && !handles_.empty(); }
  std::uint64_t windows() const { return windows_; }
  std::uint64_t mismatches() const { return mismatches_; }

private:
  void Run(Direct& d, std::uintptr_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint8_t> scratch;
    constexpr std::size_t kWindow = 1u << 20;
    while (!stop_) {
      const std::uint64_t offset = (rng() % (size_ - kWindow)) & ~std::uint64_t{15};
      const std::uint8_t* payload = nullptr;
      if (!DirectRead(d.fd, offset, kWindow, scratch, payload)) {
        ++mismatches_;
        continue;
      }
      if (std::memcmp(payload, base_ + offset, kWindow) != 0)
        ++mismatches_;
      ++windows_;
    }
  }
  const FileCase& file_;
  const std::uint8_t* base_{nullptr};
  std::size_t size_{0};
  bool unbuffered_{true};
  std::atomic<bool> stop_{false};
  std::atomic<std::uint64_t> windows_{0}, mismatches_{0};
  std::vector<std::unique_ptr<Direct>> handles_;
  std::vector<std::thread> threads_;
};

void RealStress(const std::vector<std::string>& files, int reps, int readers) {
  std::size_t tensors = 0;
  for (const auto& path : files) {
    FileCase file = Open(path);
    const auto regions = file.reader->GetMappedRegions();
    Require(!regions.empty(), path + ": no mapped region");
    // A spread of HC tensors: first/middle/last layer and the output HC.
    std::vector<const gufo::core::GgufTensorInfo*> picked;
    for (const auto& info : file.reader->GetTensors()) {
      const std::string name(info.name);
      if (info.type != GgmlType::kBF16 || !IsHcTensor(name))
        continue;
      if (name.rfind("blk.0.", 0) == 0 || name.rfind("blk.1.", 0) == 0 || name.rfind("blk.24.", 0) == 0 ||
          name.rfind("blk.47.", 0) == 0 || name.rfind("output_hc", 0) == 0)
        picked.push_back(&info);
    }
    if (picked.empty())
      continue;
    Hammer hammer(file, readers);
    for (const auto* info : picked) {
      const std::size_t cols = info->dimensions[0];
      const std::size_t rows = info->dimensions.size() > 1 ? info->dimensions[1] : 1;
      const bool inject = std::string(info->name).find("_inject") != std::string::npos;
      StressTensor(std::string(info->name), info->data, rows, cols, inject, reps);
      ++tensors;
    }
    hammer.Stop();
    Require(hammer.mismatches() == 0,
            path + ": an unbuffered read returned bytes different from the mapped view (" +
                std::to_string(hammer.mismatches()) + " of " + std::to_string(hammer.windows()) +
                " windows)");
    std::cout << "  " << path.substr(path.find_last_of("/\\") + 1) << ": " << picked.size()
              << " HC tensors x " << std::size(kThreadCounts) << " thread counts x " << reps
              << " reps identical while " << readers << " unbuffered readers ran ("
              << (hammer.unbuffered() ? "unbuffered" : "buffered fallback") << ", " << hammer.windows()
              << " one-MiB windows equal to the mapped view)\n";
  }
  Require(tensors > 0, "no BF16 HC tensor found in the given files");
}

void PassThroughStress(const std::string& path, int reps, int readers) {
  // The MTP sidecar's tensors are copied by the stager: an unbuffered aligned
  // read, then a device copy. Check, for every tensor, that exactly those bytes
  // equal the mapped view, `reps` times, while the hammer runs.
  FileCase file = Open(path);
  Hammer hammer(file, readers);
  gufo::platform::PlatformFile dup = gufo::platform::DuplicateFile(file.reader->GetMappedRegions()[0].file_descriptor);
  Direct direct;
  direct.fd = gufo::platform::ReopenUnbuffered(dup);
  gufo::platform::CloseFile(dup);
  std::vector<std::uint8_t> scratch;
  std::size_t tensors = 0, bytes = 0;
  for (int rep = 0; rep < reps; ++rep) {
    for (const auto& info : file.reader->GetTensors()) {
      if (info.size_bytes == 0)
        continue;
      const auto* mapped = static_cast<const std::uint8_t*>(info.data);
      const auto* base = static_cast<const std::uint8_t*>(file.reader->GetMappedRegions()[0].data);
      const std::uint64_t offset = static_cast<std::uint64_t>(mapped - base);
      // 16 MiB chunks like the stager
      for (std::size_t done = 0; done < info.size_bytes;) {
        const std::size_t n = std::min<std::size_t>(16u << 20, info.size_bytes - done);
        const std::uint8_t* payload = nullptr;
        if (gufo::platform::FileValid(direct.fd)) {
          Require(DirectRead(direct.fd, offset + done, n, scratch, payload),
                  std::string(info.name) + ": unbuffered read failed");
        } else {
          payload = mapped + done;  // buffered fallback: nothing to compare
        }
        Require(std::memcmp(payload, mapped + done, n) == 0,
                std::string(info.name) + ": unbuffered bytes differ from the mapped view at +" +
                    std::to_string(done));
        done += n;
        bytes += n;
      }
      ++tensors;
    }
  }
  hammer.Stop();
  Require(hammer.mismatches() == 0, path + ": hammer saw an unbuffered/mapped difference");
  std::cout << "  " << path.substr(path.find_last_of("/\\") + 1) << ": " << tensors
            << " tensor reads (" << bytes / (1u << 20) << " MiB) via the stager's unbuffered path equal to the mapped view, "
            << reps << " pass(es), " << readers << " concurrent readers\n";
}

}  // namespace

int main(int argc, char** argv) try {
  std::vector<std::string> real;
  std::string mtp;
  int reps = 6, readers = 16;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--reps" && i + 1 < argc)
      reps = std::stoi(argv[++i]);
    else if (arg == "--readers" && i + 1 < argc)
      readers = std::stoi(argv[++i]);
    else if (arg == "--mtp" && i + 1 < argc)
      mtp = argv[++i];
    else if (arg == "--real") {
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        real.push_back(argv[++i]);
    } else
      throw std::runtime_error("unknown argument " + arg);
  }
  const auto begin = std::chrono::steady_clock::now();
  SyntheticStress(reps);
  if (!real.empty())
    RealStress(real, reps, readers);
  if (!mtp.empty())
    PassThroughStress(mtp, std::max(1, reps / 3), readers);
  std::cout << "  elapsed " << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
            << " s\nHC LOAD Q8 STRESS PASS\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
