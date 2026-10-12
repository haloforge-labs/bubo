#include "src/core/prepare/hash_file.hpp"

#include <array>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "src/core/prepare/file_ops.hpp"

namespace bubo::prepare {
std::uint64_t HashInto(const std::filesystem::path& path,
                       gufo::crypto::Sha256Hasher& hasher, std::uint64_t limit,
                       const HashProgress& progress,
                       std::intptr_t protected_handle) {
  std::ifstream input;
#ifdef _WIN32
  // Share read/write/delete with our own already protected transfer handle.
  // The caller's lease enforces immutability; this reader never weakens it.
  std::unique_ptr<void, decltype(&CloseHandle)> owned(nullptr, CloseHandle);
  if (protected_handle == -1) {
    const auto handle =
        CreateFileW(NativePath(path).c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
      throw std::runtime_error("cannot open protected SHA-256 reader");
    owned.reset(handle);
    protected_handle = reinterpret_cast<std::intptr_t>(handle);
  }
#else
  if (protected_handle == -1)
    input.open(path, std::ios::binary);
#endif
  if ((protected_handle == -1 && !input) ||
      limit > std::filesystem::file_size(path))
    throw std::runtime_error("cannot read SHA-256 prefix: " + path.string());
#ifdef _WIN32
  if (protected_handle != -1 &&
      !SetFilePointerEx(reinterpret_cast<HANDLE>(protected_handle), {}, nullptr,
                        FILE_BEGIN))
    throw std::runtime_error("cannot seek protected SHA-256 prefix");
#endif
  struct Slot {
    std::vector<std::uint8_t> bytes =
        std::vector<std::uint8_t>(4 * 1024 * 1024);
    std::size_t size = 0;
    bool ready = false;
  };
  std::array<Slot, 2> slots;
  std::mutex mutex;
  std::condition_variable changed;
  bool stop = false;
  std::exception_ptr failure;
  std::thread reader([&] {
    try {
      std::uint64_t read = 0;
      std::size_t index = 0;
      while (read < limit) {
        auto& slot = slots[index];
        {
          std::unique_lock lock(mutex);
          changed.wait(lock, [&] { return stop || !slot.ready; });
          if (stop)
            return;
        }
        const auto size = static_cast<std::size_t>(
            std::min<std::uint64_t>(slot.bytes.size(), limit - read));
#ifdef _WIN32
        if (protected_handle != -1) {
          DWORD read_count = 0;
          if (!ReadFile(reinterpret_cast<HANDLE>(protected_handle),
                        slot.bytes.data(), static_cast<DWORD>(size),
                        &read_count, nullptr) ||
              read_count != size)
            throw std::runtime_error("short protected read during SHA-256");
        } else
#endif
        {
          input.read(reinterpret_cast<char*>(slot.bytes.data()), size);
          if (input.gcount() != static_cast<std::streamsize>(size))
            throw std::runtime_error("short read during SHA-256");
        }
        {
          std::lock_guard lock(mutex);
          slot.size = size;
          slot.ready = true;
        }
        changed.notify_all();
        read += size;
        index ^= 1;
      }
    } catch (...) {
      std::lock_guard lock(mutex);
      failure = std::current_exception();
      changed.notify_all();
    }
  });
  std::uint64_t hashed = 0;
  try {
    std::size_t index = 0;
    while (hashed < limit) {
      auto& slot = slots[index];
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return slot.ready || failure; });
        if (failure)
          std::rethrow_exception(failure);
      }
      hasher.Update(std::span(slot.bytes.data(), slot.size));
      hashed += slot.size;
      if (progress)
        progress(hashed, limit);
      {
        std::lock_guard lock(mutex);
        slot.ready = false;
      }
      changed.notify_all();
      index ^= 1;
    }
  } catch (...) {
    {
      std::lock_guard lock(mutex);
      stop = true;
    }
    changed.notify_all();
    reader.join();
    throw;
  }
  reader.join();
  return hashed;
}
std::string HashFile(const std::filesystem::path& path,
                     const HashProgress& progress) {
  gufo::crypto::Sha256Hasher hasher;
  HashInto(path, hasher, std::filesystem::file_size(path), progress);
  return hasher.FinishHex();
}
}  // namespace bubo::prepare
