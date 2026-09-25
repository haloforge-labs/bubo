#include "src/core/gguf_identity.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace gufo::core {
namespace {

class FileDescriptor {
public:
  explicit FileDescriptor(int fd) : fd_(fd) {}
  ~FileDescriptor() {
    if (fd_ >= 0)
      close(fd_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] int get() const { return fd_; }

private:
  int fd_;
};

void HashString(crypto::Sha256Hasher& hash, std::string_view value) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}

std::string FileStamp(int fd, std::size_t size) {
  struct stat st{};
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
      static_cast<std::size_t>(st.st_size) != size) {
    throw std::runtime_error("GGUF file changed or cannot be inspected");
  }
  return std::to_string(st.st_dev) + ':' + std::to_string(st.st_ino) + ':' +
         std::to_string(st.st_size) + ':' + std::to_string(st.st_mtim.tv_sec) +
         ':' + std::to_string(st.st_mtim.tv_nsec) + ':' +
         std::to_string(st.st_ctim.tv_sec) + ':' +
         std::to_string(st.st_ctim.tv_nsec);
}

int OpenCacheDirectory() {
  const char* base = std::getenv("XDG_CACHE_HOME");
  std::filesystem::path path;
#ifdef _WIN32
  // %LOCALAPPDATA% is the per-user cache root on Windows; HOME is usually
  // unset outside POSIX shells.
  if (!base || !*base)
    base = std::getenv("LOCALAPPDATA");
#endif
  if (base && *base)
    path = base;
  else {
    const char* home = std::getenv("HOME");
    if (!home || !*home)
      return -1;
    path = std::filesystem::path(home) / ".cache";
  }
  path /= "gufo";
  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error)
    return -1;
  path /= "gguf-sha256-v1";
  (void)mkdir(path.c_str(), 0700);
  int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0)
    return -1;
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_uid != geteuid() ||
      (st.st_mode & 0077) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

std::string ReadDigest(int directory, const std::string& key) {
  if (directory < 0)
    return {};
  FileDescriptor file(
      openat(directory, key.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  struct stat st{};
  if (file.get() < 0 || fstat(file.get(), &st) != 0 || !S_ISREG(st.st_mode) ||
      st.st_uid != geteuid() || (st.st_mode & 0022) != 0 || st.st_size != 65)
    return {};
  std::array<char, 65> bytes{};
  if (read(file.get(), bytes.data(), bytes.size()) !=
          static_cast<ssize_t>(bytes.size()) ||
      bytes.back() != '\n' ||
      !std::all_of(bytes.begin(), bytes.end() - 1, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return {};
  return {bytes.data(), bytes.size() - 1};
}

void StoreDigest(int directory, const std::string& key,
                 const std::string& digest) {
  if (directory < 0)
    return;
  static std::atomic<unsigned long> serial{0};
  const std::string temp = "." + key + "." + std::to_string(getpid()) + "." +
                           std::to_string(serial.fetch_add(1));
  FileDescriptor file(
      openat(directory, temp.c_str(),
             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
  if (file.get() < 0)
    return;
  const std::string contents = digest + '\n';
  if (write(file.get(), contents.data(), contents.size()) ==
      static_cast<ssize_t>(contents.size())) {
    (void)renameat(directory, temp.c_str(), directory, key.c_str());
  }
  (void)unlinkat(directory, temp.c_str(), 0);
}

void HashFile(crypto::Sha256Hasher& hash, int fd, std::size_t size) {
  constexpr std::size_t kChunk = 8 * 1024 * 1024;
  constexpr std::size_t kAlignment = 4096;
  constexpr std::size_t kReaders = 4;
  // Preserve the open inode and avoid filling the remaining UMA with a second
  // copy of weights after GPU loading. Hashing stays in exact file order.
  FileDescriptor direct(
      size >= kChunk ? open(("/proc/self/fd/" + std::to_string(fd)).c_str(),
                            O_RDONLY | O_CLOEXEC | O_DIRECT)
                     : -1);
  // Aligned operator new rather than std::aligned_alloc, which the MSVC
  // runtime does not provide.
  struct AlignedDelete {
    void operator()(std::uint8_t* pointer) const noexcept {
      ::operator delete(pointer, std::align_val_t{kAlignment});
    }
  };
  struct Slot {
    std::unique_ptr<std::uint8_t, AlignedDelete> buffer;
    std::future<void> read;
    std::size_t size{0};
  };
  std::array<Slot, kReaders> slots;
  std::size_t next = 0;
  const auto queue = [&](Slot& slot) {
    if (next == size)
      return;
    if (!slot.buffer) {
      slot.buffer.reset(static_cast<std::uint8_t*>(
          ::operator new(kChunk, std::align_val_t{kAlignment}, std::nothrow)));
      if (!slot.buffer)
        throw std::bad_alloc();
    }
    const auto offset = next;
    slot.size = std::min(kChunk, size - next);
    next += slot.size;
    slot.read = std::async(std::launch::async, [&, offset] {
      std::size_t done = 0;
      bool buffered = direct.get() < 0;
      while (done < slot.size) {
        const auto remaining = slot.size - done;
        const auto requested =
            buffered ? remaining
                     : (remaining + kAlignment - 1) / kAlignment * kAlignment;
        const auto count =
            pread(buffered ? fd : direct.get(), slot.buffer.get() + done,
                  requested, static_cast<off_t>(offset + done));
        if (count < 0 && errno == EINTR)
          continue;
        if (count < 0 && !buffered &&
            (errno == EINVAL || errno == EOPNOTSUPP)) {
          buffered = true;
          continue;
        }
        if (count <= 0)
          throw std::runtime_error("cannot read complete GGUF for identity");
        done += std::min(static_cast<std::size_t>(count), remaining);
        if (done % kAlignment != 0)
          buffered = true;
      }
    });
  };
  for (auto& slot : slots)
    queue(slot);
  for (std::size_t index = 0; slots[index].read.valid();
       index = (index + 1) % slots.size()) {
    auto& slot = slots[index];
    slot.read.get();
    hash.Update({slot.buffer.get(), slot.size});
    queue(slot);
  }
}

std::string RegionDigest(const GgufMappedRegion& region) {
  std::string stamp;
  if (region.file_descriptor >= 0)
    stamp = FileStamp(region.file_descriptor, region.size);
  FileDescriptor cache(stamp.empty() ? -1 : OpenCacheDirectory());
  crypto::Sha256Hasher key_hash;
  HashString(key_hash, stamp);
  const auto key = key_hash.FinishHex();
  std::string digest = ReadDigest(cache.get(), key);
  const bool cached = !digest.empty();
  if (!cached) {
    crypto::Sha256Hasher hash;
    constexpr std::size_t chunk = 8 * 1024 * 1024;
    if (region.file_descriptor >= 0) {
      HashFile(hash, region.file_descriptor, region.size);
    } else {
      const auto* bytes = static_cast<const std::uint8_t*>(region.data);
      for (std::size_t offset = 0; offset < region.size;) {
        const auto length = std::min(chunk, region.size - offset);
        hash.Update({bytes + offset, length});
        offset += length;
      }
    }
    digest = hash.FinishHex();
  }
  if (!stamp.empty() &&
      FileStamp(region.file_descriptor, region.size) != stamp) {
    throw std::runtime_error("GGUF file changed during identity lookup");
  }
  if (!cached)
    StoreDigest(cache.get(), key, digest);
  return digest;
}

}  // namespace

std::string GgufIdentityHex(const GgufReader& reader) {
  std::vector<std::string> stamps;
  for (const auto& region : reader.GetMappedRegions())
    stamps.push_back(region.file_descriptor < 0
                         ? std::string{}
                         : FileStamp(region.file_descriptor, region.size));
  crypto::Sha256Hasher hash;
  HashString(hash, kGgufIdentityScheme);
  for (const auto& region : reader.GetMappedRegions()) {
    HashString(hash, ":" + std::to_string(region.size) + ":");
    HashString(hash, RegionDigest(region));
  }
  for (std::size_t index = 0; index < stamps.size(); ++index) {
    const auto& region = reader.GetMappedRegions()[index];
    if (!stamps[index].empty() &&
        stamps[index] != FileStamp(region.file_descriptor, region.size))
      throw std::runtime_error("GGUF artifact changed during identity lookup");
  }
  return hash.FinishHex();
}

}  // namespace gufo::core
