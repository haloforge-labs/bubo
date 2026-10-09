#include "src/cli/serve/continuation_disk_store.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {

struct WriterSettings {
  bool evict_idle{false};
  bool slice{false};
  std::chrono::milliseconds idle{30000};
  std::size_t evict_backlog_bytes{2ULL * 1024 * 1024 * 1024};
  // GUFO_DISK_EVICT_SKIP_TOKENS: an evicted snapshot whose persisted strict prefix (same compatibility identity) is at most this many tokens shorter is not
  // written again (restore uses the prefix and prefills the tail). 0 = off (every eviction is persisted).
  std::size_t evict_skip_tokens{0};
};
const WriterSettings& DiskWriterSettings() {
  static const WriterSettings settings = [] {
    WriterSettings result;
    const auto* persist = std::getenv("GUFO_DISK_PERSIST");
    result.evict_idle = persist && std::strcmp(persist, "evict_idle") == 0;
    const auto* slice = std::getenv("GUFO_DISK_SLICE");
    result.slice = slice && std::strcmp(slice, "1") == 0;
    const auto* idle = std::getenv("GUFO_DISK_IDLE_MS");
    if (idle && *idle) {
      char* end = nullptr;
      const auto value = std::strtol(idle, &end, 10);
      if (end && !*end && value >= 0 && value <= 600000)
        result.idle = std::chrono::milliseconds(value);
    }
    const auto* backlog = std::getenv("GUFO_DISK_EVICT_BACKLOG_BYTES");
    if (backlog && *backlog && *backlog != '-') {
      char* end = nullptr;
      errno = 0;
      const auto value = std::strtoull(backlog, &end, 10);
      if (!errno && end && !*end && value <= SIZE_MAX)
        result.evict_backlog_bytes = static_cast<std::size_t>(value);
    }
    const auto* skip = std::getenv("GUFO_DISK_EVICT_SKIP_TOKENS");
    if (skip && *skip && *skip != '-') {
      char* end = nullptr;
      errno = 0;
      const auto value = std::strtoull(skip, &end, 10);
      if (!errno && end && !*end && value <= 1000000)
        result.evict_skip_tokens = static_cast<std::size_t>(value);
    }
    return result;
  }();
  return settings;
}
std::int64_t ActivityTime() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool DiskDeferEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_DISK_DEFER");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// GUFO_DISK_DEFER_MS overrides the idle-gate bound (default 10 s, accepted 0..600000 ms), read once: the A/B can test a bound longer than a typical turn.
std::chrono::milliseconds DiskDeferMaxWait(std::chrono::milliseconds fallback) {
  // Only the environment parse is cached; the caller's option value is not (a static capturing `fallback` would pin the first store's bound).
  static const std::optional<std::chrono::milliseconds> override_wait = []() -> std::optional<std::chrono::milliseconds> {
    const char* text = std::getenv("GUFO_DISK_DEFER_MS");
    if (!text || !*text)
      return std::nullopt;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end && !*end && parsed >= 0 && parsed <= 600000)
      return std::chrono::milliseconds(parsed);
    return std::nullopt;
  }();
  return override_wait ? *override_wait : fallback;
}

TextRunnerDescriptor DescriptorForInput(
    const TextModelRunner& runner,
    std::span<const std::uint8_t> input_identity) {
  auto descriptor = runner.Descriptor();
  if (input_identity.empty() || !descriptor.persistence)
    return descriptor;
  auto& identity = descriptor.persistence->compatibility_identity;
  const auto model_size = identity.size();
  identity.insert(identity.begin(), {'I', 'M', 'G', 1});
  for (unsigned shift = 0; shift < 64; shift += 8) {
    identity.push_back(
        static_cast<std::uint8_t>(std::uint64_t{model_size} >> shift));
  }
  identity.insert(identity.end(), input_identity.begin(), input_identity.end());
  return descriptor;
}

template<class Visit>
void VisitInputRanges(std::size_t token_count,
                      std::span<const std::uint8_t> identity,
                      std::span<const ContinuationInputPrefix> prefixes,
                      Visit&& visit) {
  std::size_t begin = 0;
  for (const auto& prefix : prefixes) {
    const auto end = std::min(prefix.token_count, token_count);
    if (end < begin)
      throw std::invalid_argument("input identity boundaries are invalid");
    visit(begin, end, std::span<const std::uint8_t>(prefix.identity));
    if (end == token_count)
      return;
    begin = end + 1;
  }
  visit(begin, token_count, identity);
}

constexpr std::array<std::uint8_t, 8> kMagic = {'G', 'U', 'F', 'O',
                                                'K', 'V', 'C', '1'};
constexpr std::uint32_t kFileVersion = 1;
constexpr std::size_t kChecksumBytes = 64;
constexpr std::size_t kMagicOffset = 0;
constexpr std::size_t kFileVersionOffset = 8;
constexpr std::size_t kPayloadVersionOffset = 12;
constexpr std::size_t kIdentityBytesOffset = 16;
constexpr std::size_t kTokenCountOffset = 20;
constexpr std::size_t kPayloadBytesOffset = 24;
constexpr std::size_t kChecksumOffset = 32;
constexpr std::size_t kHeaderBytes = kChecksumOffset + kChecksumBytes;
constexpr std::size_t kMaxIdentityBytes = std::size_t{64} * 1024U;
constexpr std::size_t kMaxTokenCount = std::size_t{16} * 1024U * 1024U;
constexpr std::string_view kFileSuffix = ".kvc";
constexpr std::string_view kTemporaryPrefix = ".tmp-";

class ScopedFileDescriptor {
public:
  ScopedFileDescriptor() = default;
  explicit ScopedFileDescriptor(int descriptor) noexcept
      : descriptor_(descriptor) {}
  ~ScopedFileDescriptor() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

  ScopedFileDescriptor(ScopedFileDescriptor&& other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  ScopedFileDescriptor& operator=(ScopedFileDescriptor&& other) noexcept {
    if (this != &other) {
      if (descriptor_ >= 0) {
        ::close(descriptor_);
      }
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] int release() noexcept {
    return std::exchange(descriptor_, -1);
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    return descriptor_ >= 0;
  }

private:
  int descriptor_{-1};
};

class ScopedOperationPermit {
public:
  explicit ScopedOperationPermit(std::binary_semaphore& gate, bool wait = true)
      : gate_(gate) {
    if (wait) {
      gate_.acquire();
      acquired_ = true;
    } else {
      acquired_ = gate_.try_acquire();
    }
  }
  ~ScopedOperationPermit() {
    if (acquired_)
      gate_.release();
  }
  explicit operator bool() const noexcept { return acquired_; }
  void Unlock() {
    gate_.release();
    acquired_ = false;
  }
  void Lock() {
    gate_.acquire();
    acquired_ = true;
  }

  ScopedOperationPermit(const ScopedOperationPermit&) = delete;
  ScopedOperationPermit& operator=(const ScopedOperationPermit&) = delete;
  ScopedOperationPermit(ScopedOperationPermit&&) = delete;
  ScopedOperationPermit& operator=(ScopedOperationPermit&&) = delete;

private:
  std::binary_semaphore& gate_;
  bool acquired_{false};
};

class ScopedFileLock {
public:
  ScopedFileLock(int descriptor, int operation) noexcept
      : descriptor_(descriptor) {
    int result;
    do {
      result = ::flock(descriptor_, operation);
    } while (result != 0 && errno == EINTR);
    acquired_ = result == 0;
  }

  ~ScopedFileLock() {
    if (acquired_) {
      while (::flock(descriptor_, LOCK_UN) != 0 && errno == EINTR) {
      }
    }
  }

  explicit operator bool() const noexcept { return acquired_; }

  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;

private:
  int descriptor_;
  bool acquired_{false};
};

template<typename Integer>
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  if (offset > destination.size() ||
      sizeof(Integer) > destination.size() - offset) {
    throw std::out_of_range("continuation disk header write overflow");
  }
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename Integer>
bool GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset,
                     Integer* value) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  if (value == nullptr || offset > source.size() ||
      sizeof(Integer) > source.size() - offset) {
    return false;
  }
  Integer result = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    result |= static_cast<Integer>(source[offset + byte]) << (byte * 8U);
  }
  *value = result;
  return true;
}

bool IsLowerHexDigest(std::string_view digest) noexcept {
  if (digest.size() != kChecksumBytes) {
    return false;
  }
  return std::ranges::all_of(digest, [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

std::string DefaultKeyHash(std::span<const std::uint8_t> bytes) {
  return crypto::Sha256Hex(bytes);
}

std::vector<std::uint8_t> MakeKeyBytes(
    const TextRunnerPersistenceDescriptor& persistence,
    std::span<const TextRunnerToken> tokens) {
  if (persistence.compatibility_identity.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("continuation disk key is too large");
  }
  const std::size_t identity_bytes = persistence.compatibility_identity.size();
  if (tokens.size() >
      (std::numeric_limits<std::size_t>::max() - 12U - identity_bytes) /
          sizeof(std::uint32_t)) {
    throw std::overflow_error("continuation disk key size overflows");
  }
  std::vector<std::uint8_t> key(12U + identity_bytes +
                                tokens.size() * sizeof(std::uint32_t));
  PutLittleEndian<std::uint32_t>(key, 0,
                                 static_cast<std::uint32_t>(identity_bytes));
  PutLittleEndian<std::uint32_t>(key, 4, persistence.payload_version);
  PutLittleEndian<std::uint32_t>(key, 8,
                                 static_cast<std::uint32_t>(tokens.size()));
  std::copy(persistence.compatibility_identity.begin(),
            persistence.compatibility_identity.end(), key.begin() + 12);
  std::size_t offset = 12U + identity_bytes;
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(key, offset, token);
    offset += sizeof(std::uint32_t);
  }
  return key;
}

std::size_t CheckedFileBytes(std::size_t identity_bytes,
                             std::size_t token_count,
                             std::size_t payload_bytes) {
  if (identity_bytes > kMaxIdentityBytes || token_count > kMaxTokenCount) {
    throw std::invalid_argument(
        "continuation disk metadata exceeds format limits");
  }
  if (token_count > (std::numeric_limits<std::size_t>::max() - kHeaderBytes -
                     identity_bytes) /
                        sizeof(std::uint32_t)) {
    throw std::overflow_error("continuation disk metadata size overflows");
  }
  const std::size_t metadata_bytes =
      kHeaderBytes + identity_bytes + token_count * sizeof(std::uint32_t);
  if (payload_bytes >
      std::numeric_limits<std::size_t>::max() - metadata_bytes) {
    throw std::overflow_error("continuation disk file size overflows");
  }
  return metadata_bytes + payload_bytes;
}

std::string ErrnoMessage(std::string_view operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool WriteAll(int descriptor, std::span<const std::uint8_t> bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t written =
        ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

bool ReadAll(int descriptor, std::span<std::uint8_t> bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (count == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

bool HasSuffix(std::string_view value, std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

enum class ParseFailure : std::uint8_t {
  kNone,
  kCorrupt,
  kChecksum,
};

struct ParsedImage {
  TextRunnerPersistenceDescriptor persistence;
  std::vector<TextRunnerToken> tokens;
  std::size_t payload_offset{0};
  std::size_t payload_bytes{0};
};

ParseFailure ParseAndVerifyImage(std::vector<std::uint8_t>* image,
                                 ParsedImage* parsed) {
  if (image == nullptr || parsed == nullptr || image->size() < kHeaderBytes ||
      !std::equal(kMagic.begin(), kMagic.end(),
                  image->begin() + kMagicOffset)) {
    return ParseFailure::kCorrupt;
  }

  std::uint32_t file_version = 0;
  std::uint32_t payload_version = 0;
  std::uint32_t identity_bytes_u32 = 0;
  std::uint32_t token_count_u32 = 0;
  std::uint64_t payload_bytes_u64 = 0;
  const std::span<const std::uint8_t> readonly(*image);
  if (!GetLittleEndian(readonly, kFileVersionOffset, &file_version) ||
      !GetLittleEndian(readonly, kPayloadVersionOffset, &payload_version) ||
      !GetLittleEndian(readonly, kIdentityBytesOffset, &identity_bytes_u32) ||
      !GetLittleEndian(readonly, kTokenCountOffset, &token_count_u32) ||
      !GetLittleEndian(readonly, kPayloadBytesOffset, &payload_bytes_u64) ||
      file_version != kFileVersion || payload_version == 0 ||
      identity_bytes_u32 == 0 || identity_bytes_u32 > kMaxIdentityBytes ||
      token_count_u32 == 0 || token_count_u32 > kMaxTokenCount ||
      payload_bytes_u64 == 0 ||
      payload_bytes_u64 >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ParseFailure::kCorrupt;
  }

  const std::size_t identity_bytes = identity_bytes_u32;
  const std::size_t token_count = token_count_u32;
  const std::size_t payload_bytes = static_cast<std::size_t>(payload_bytes_u64);
  std::size_t expected_bytes = 0;
  try {
    expected_bytes =
        CheckedFileBytes(identity_bytes, token_count, payload_bytes);
  } catch (...) {
    return ParseFailure::kCorrupt;
  }
  if (expected_bytes != image->size()) {
    return ParseFailure::kCorrupt;
  }

  const std::string stored_checksum(
      reinterpret_cast<const char*>(image->data() + kChecksumOffset),
      kChecksumBytes);
  if (!IsLowerHexDigest(stored_checksum)) {
    return ParseFailure::kCorrupt;
  }
  std::fill_n(image->begin() + kChecksumOffset, kChecksumBytes, 0);
  const std::string computed_checksum = crypto::Sha256Hex(*image);
  std::copy(stored_checksum.begin(), stored_checksum.end(),
            image->begin() + kChecksumOffset);
  if (computed_checksum != stored_checksum) {
    return ParseFailure::kChecksum;
  }

  parsed->persistence.payload_version = payload_version;
  parsed->persistence.compatibility_identity.resize(identity_bytes);
  std::memcpy(parsed->persistence.compatibility_identity.data(),
              image->data() + kHeaderBytes, identity_bytes);
  parsed->tokens.resize(token_count);
  std::size_t token_offset = kHeaderBytes + identity_bytes;
  for (TextRunnerToken& token : parsed->tokens) {
    if (!GetLittleEndian(readonly, token_offset, &token)) {
      return ParseFailure::kCorrupt;
    }
    token_offset += sizeof(std::uint32_t);
  }
  parsed->payload_offset = token_offset;
  parsed->payload_bytes = payload_bytes;
  return ParseFailure::kNone;
}

std::vector<std::uint8_t> BuildHeader(
    const TextRunnerPersistenceDescriptor& persistence,
    std::span<const TextRunnerToken> tokens, std::size_t payload_bytes) {
  const std::size_t file_bytes = CheckedFileBytes(
      persistence.compatibility_identity.size(), tokens.size(), payload_bytes);
  std::vector<std::uint8_t> image(file_bytes - payload_bytes);
  std::copy(kMagic.begin(), kMagic.end(), image.begin() + kMagicOffset);
  PutLittleEndian<std::uint32_t>(image, kFileVersionOffset, kFileVersion);
  PutLittleEndian<std::uint32_t>(image, kPayloadVersionOffset,
                                 persistence.payload_version);
  PutLittleEndian<std::uint32_t>(
      image, kIdentityBytesOffset,
      static_cast<std::uint32_t>(persistence.compatibility_identity.size()));
  PutLittleEndian<std::uint32_t>(image, kTokenCountOffset,
                                 static_cast<std::uint32_t>(tokens.size()));
  PutLittleEndian<std::uint64_t>(image, kPayloadBytesOffset,
                                 static_cast<std::uint64_t>(payload_bytes));
  std::copy(persistence.compatibility_identity.begin(),
            persistence.compatibility_identity.end(),
            image.begin() + kHeaderBytes);
  std::size_t token_offset =
      kHeaderBytes + persistence.compatibility_identity.size();
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(image, token_offset, token);
    token_offset += sizeof(std::uint32_t);
  }
  return image;
}

}  // namespace

struct ContinuationDiskStore::Impl {
  struct Entry {
    std::string filename;
    std::string key_hash;
    TextRunnerPersistenceDescriptor persistence;
    std::vector<TextRunnerToken> tokens;
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
    std::filesystem::file_time_type last_access;
  };

  using EntryIterator = std::list<Entry>::iterator;

  // Compressed token edges keep lookup proportional to the prompt length.
  // Full compatibility bytes and tokens remain the keys, never only a hash.
  struct PrefixNode {
    std::vector<TextRunnerToken> edge;
    std::optional<EntryIterator> entry;
    std::map<TextRunnerToken, std::unique_ptr<PrefixNode>> children;

    void Insert(std::span<const TextRunnerToken> tokens, EntryIterator value) {
      if (tokens.empty()) {
        entry = value;
        return;
      }
      auto& child = children[tokens.front()];
      if (!child) {
        child = std::make_unique<PrefixNode>();
        child->edge.assign(tokens.begin(), tokens.end());
        child->entry = value;
        return;
      }
      const auto mismatch = std::ranges::mismatch(child->edge, tokens);
      const auto shared =
          static_cast<std::size_t>(mismatch.in1 - child->edge.begin());
      if (shared < child->edge.size()) {
        auto branch = std::make_unique<PrefixNode>();
        branch->edge.assign(child->edge.begin(), child->edge.begin() + shared);
        child->edge.erase(child->edge.begin(), child->edge.begin() + shared);
        const auto key = child->edge.front();
        branch->children.emplace(key, std::move(child));
        child = std::move(branch);
      }
      child->Insert(tokens.subspan(shared), value);
    }

    void Erase(std::span<const TextRunnerToken> tokens) noexcept {
      if (tokens.empty()) {
        entry.reset();
        return;
      }
      auto found = children.find(tokens.front());
      if (found == children.end())
        return;
      auto& child = *found->second;
      if (tokens.size() < child.edge.size() ||
          !std::ranges::equal(child.edge, tokens.first(child.edge.size()))) {
        return;
      }
      child.Erase(tokens.subspan(child.edge.size()));
      if (!child.entry && child.children.empty())
        children.erase(found);
    }

    struct Match {
      std::optional<EntryIterator> longest;
      std::vector<std::size_t> shared_boundaries;
    };

    [[nodiscard]] Match Find(std::span<const TextRunnerToken> prompt,
                             bool boundaries = false) const {
      Match result;
      const PrefixNode* node = this;
      std::size_t offset = 0;
      while (true) {
        if (node->entry)
          result.longest = node->entry;
        if (offset == prompt.size())
          break;
        const auto next = node->children.find(prompt[offset]);
        if (boundaries && !node->entry && offset != 0 &&
            (node->children.size() > 1 ||
             (!node->children.empty() && next == node->children.end()))) {
          result.shared_boundaries.push_back(offset);
        }
        if (next == node->children.end())
          break;
        const auto& child = *next->second;
        const auto mismatch =
            std::ranges::mismatch(child.edge, prompt.subspan(offset));
        const auto shared =
            static_cast<std::size_t>(mismatch.in1 - child.edge.begin());
        offset += shared;
        if (shared != child.edge.size()) {
          if (boundaries && offset != 0 && offset < prompt.size()) {
            result.shared_boundaries.push_back(offset);
          }
          break;
        }
        node = &child;
      }
      return result;
    }
  };

  using PersistenceKey = std::pair<std::uint32_t, std::vector<std::uint8_t>>;
  static PersistenceKey PrefixKey(
      const TextRunnerPersistenceDescriptor& descriptor) {
    return {descriptor.payload_version, descriptor.compatibility_identity};
  }

  struct PendingSave {
    std::shared_ptr<const TextModelRunner> runner;
    std::vector<TextRunnerToken> tokens;
    std::shared_ptr<const TextRunnerSnapshot> snapshot;
    std::vector<std::uint8_t> identity;
    std::size_t retained_bytes;
    bool shared_prefix{false};
    std::chrono::steady_clock::time_point enqueued;
    std::string_view trigger{"write_through"};
  };

  Impl(ContinuationDiskStoreOptions store_options, EventSink sink,
       KeyHashFunction hasher)
      : options(std::move(store_options)),
        event_sink(std::move(sink)),
        key_hash(std::move(hasher)) {
    options.persist_evict_idle =
        options.persist_evict_idle.value_or(DiskWriterSettings().evict_idle);
    options.slice = options.slice.value_or(DiskWriterSettings().slice);
    options.idle_wait = options.idle_wait.value_or(DiskWriterSettings().idle);
    options.evict_backlog_bytes = options.evict_backlog_bytes.value_or(
        DiskWriterSettings().evict_backlog_bytes);
    options.evict_skip_tokens = options.evict_skip_tokens.value_or(
        DiskWriterSettings().evict_skip_tokens);
    if (options.persist_evict_idle.value())
      superseded = std::make_shared<std::atomic<std::size_t>>(0);
    ValidateOptions();
    InitializeDirectory();
    IndexExistingFiles();
    EvictToCapacity();
    writer = std::thread([this] { PersistQueued(); });
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    Shutdown();
    try {
      std::ostringstream line;
      line << "event=disk_cache_writer_summary captures_seen="
           << captures.load()
           << " persisted_write_through=" << persisted[0].load()
           << " persisted_evict=" << persisted[1].load()
           << " persisted_idle=" << persisted[2].load()
           << " persisted_shutdown=" << persisted[3].load()
           << " superseded=" << (superseded ? superseded->load() : 0)
           << " already_persisted=" << already_persisted.load()
           << " dropped_evict=" << dropped_evict.load()
           << " skipped=" << skipped->load()
           << " bytes_written=" << bytes_written.load();
      Logger::Info("cache", line.str());
    } catch (...) {
    }
    if (directory_fd >= 0) {
      ::close(directory_fd);
    }
  }

  void Shutdown() {
    {
      std::lock_guard lock(queue_mutex);
      if (!stopping) {
        shutdown_deadline =
            std::chrono::steady_clock::now() + options.shutdown_wait;
        stopping.store(true, std::memory_order_release);
      }
    }
    queue_changed.notify_all();
    if (writer.joinable())
      writer.join();
    // Defensive drain: every accepted offer must leave Queued at shutdown.
    std::lock_guard lock(queue_mutex);
    for (auto& job : pending) {
      SkipPersistence(job.snapshot,
                      ContinuationDiskEventReason::kShutdownDeadline,
                      job.trigger);
      if (job.snapshot->persistence)
        job.snapshot->persistence->state.store(SnapshotPersistence::kAbandoned);
    }
    pending.clear();
    for (auto& job : evict_backlog) {
      DropEvict(job.snapshot, ContinuationDiskEventReason::kShutdownDeadline);
    }
    evict_backlog.clear();
    backlog_bytes = queued_bytes = 0;
    queue_changed.notify_all();
  }

  void SkipPersistence(
      const std::shared_ptr<const ContinuationSnapshot>& snapshot,
      ContinuationDiskEventReason reason, std::string_view trigger) {
    if (!snapshot || !snapshot->persistence) {
      Emit(ContinuationDiskEventAction::kDroppedEvict, reason, 0,
           snapshot ? snapshot->PayloadBytes() : 0, 0, 0, 0, trigger);
      return;
    }
    auto& record = *snapshot->persistence;
    record.state.store(SnapshotPersistence::kFailed);
    Emit(ContinuationDiskEventAction::kSkipped, reason, 0,
         snapshot->PayloadBytes(), record.tokens.size(), 0, 0, trigger);
  }

  void DropEvict(const std::shared_ptr<const ContinuationSnapshot>& snapshot,
                 ContinuationDiskEventReason reason) {
    if (snapshot->persistence)
      snapshot->persistence->state.store(SnapshotPersistence::kAbandoned);
    Emit(ContinuationDiskEventAction::kDroppedEvict, reason, 0,
         snapshot->PayloadBytes(),
         snapshot->persistence ? snapshot->persistence->tokens.size() : 0, 0, 0,
         "evict");
  }

  // queue_mutex held. Transfer the charge; never exceed active+queued staging.
  void PromoteBacklog() {
    if (stopping && std::chrono::steady_clock::now() >= shutdown_deadline) {
      for (auto& job : evict_backlog)
        DropEvict(job.snapshot, ContinuationDiskEventReason::kShutdownDeadline);
      evict_backlog.clear();
      backlog_bytes = 0;
      return;
    }
    const auto limit =
        std::min(options.capacity_bytes, options.staging_capacity_bytes);
    while (!evict_backlog.empty()) {
      auto& job = evict_backlog.front();
      const auto used = queued_bytes + capture_bytes->load();
      if (used > limit - job.retained_bytes)
        break;
      const auto charge = job.retained_bytes;
      try {
        pending.push_back(std::move(job));
        queued_bytes += charge;
      } catch (...) {
        DropEvict(job.snapshot,
                  ContinuationDiskEventReason::kSerializationFailure);
      }
      evict_backlog.pop_front();
      backlog_bytes -= charge;
    }
  }

  void PersistQueued() noexcept {
    for (;;) {
      PendingSave job;
      {
        std::unique_lock lock(queue_mutex);
        PromoteBacklog();
        while (pending.empty()) {
          if (options.persist_evict_idle.value() && persistence_source) {
            const auto now = std::chrono::steady_clock::now();
            const auto idle_since =
                std::chrono::steady_clock::time_point(std::chrono::nanoseconds(
                    last_activity.load(std::memory_order_acquire)));
            if (stopping ||
                (active_generations.load(std::memory_order_acquire) == 0 &&
                 now - idle_since >= options.idle_wait.value())) {
              // Source locks only the existing RAM metadata mutex. Never hold
              // queue_mutex across it: eviction may concurrently enqueue.
              auto source = persistence_source;
              const bool shutdown = stopping;
              lock.unlock();
              std::shared_ptr<const ContinuationSnapshot> snapshot;
              try {
                snapshot = source(shutdown);
              } catch (...) {
                Emit(ContinuationDiskEventAction::kSkipped,
                     ContinuationDiskEventReason::kSerializationFailure, 0, 0,
                     0);
              }
              if (snapshot) {
                if (shutdown &&
                    std::chrono::steady_clock::now() >= shutdown_deadline) {
                  // Terminal for this shutdown: source must advance past it.
                  SkipPersistence(
                      snapshot, ContinuationDiskEventReason::kShutdownDeadline,
                      "shutdown");
                  snapshot->persistence->state.store(
                      SnapshotPersistence::kAbandoned);
                } else if (shutdown || active_generations.load(
                                           std::memory_order_acquire) == 0) {
                  EnqueuePersistence(snapshot, shutdown ? "shutdown" : "idle");
                }
              }
              lock.lock();
              if (!pending.empty())
                break;
              if (shutdown && !snapshot && evict_backlog.empty()) {
                writer_exited = true;
                queue_changed.notify_all();
                return;
              }
              if (stopping && !shutdown)
                continue;  // Re-query with failed entries included at shutdown.
              if (snapshot)
                continue;
            }
          }
          if (stopping && evict_backlog.empty()) {
            writer_exited = true;
            queue_changed.notify_all();
            return;
          }
          if (options.persist_evict_idle.value())
            queue_changed.wait_for(lock, std::chrono::milliseconds(250));
          else
            queue_changed.wait(lock,
                               [this] { return stopping || !pending.empty(); });
          PromoteBacklog();
        }
        if (defer_enabled && !stopping) {
          const auto deadline = pending.front().enqueued +
                                DiskDeferMaxWait(options.defer_max_wait);
          while (flush_waiters == 0 && !stopping &&
                 active_generations.load(std::memory_order_acquire) != 0 &&
                 queued_bytes + capture_bytes->load(std::memory_order_relaxed) <
                     std::min(options.capacity_bytes,
                              options.staging_capacity_bytes) &&
                 std::chrono::steady_clock::now() < deadline) {
            queue_changed.wait_until(
                lock, std::min(deadline, std::chrono::steady_clock::now() +
                                             std::chrono::milliseconds(5)));
          }
        }
        // Idle work stays pending if admission raced its final preflight.
        if (pending.front().trigger == "idle" && !stopping &&
            flush_waiters == 0 &&
            (active_generations.load(std::memory_order_acquire) != 0 ||
             std::chrono::steady_clock::now().time_since_epoch() -
                     std::chrono::nanoseconds(
                         last_activity.load(std::memory_order_acquire)) <
                 options.idle_wait.value())) {
          queue_changed.wait_for(lock, std::chrono::milliseconds(250));
          continue;
        }
        job = std::move(pending.front());
        pending.pop_front();
      }
      const auto record = job.snapshot->persistence;
      bool stored = false;
      try {
        if (stopping && options.persist_evict_idle.value() &&
            std::chrono::steady_clock::now() >= shutdown_deadline) {
          SkipPersistence(job.snapshot,
                          ContinuationDiskEventReason::kShutdownDeadline,
                          job.trigger);
        } else {
          // Existing exact files count as persisted without serializing again.
          bool exists = false;
          if (record) {
            const ScopedOperationPermit permit(operation_gate);
            exists = Touch(*job.runner, job.tokens, job.identity);
          }
          // Evict-skip policy: a snapshot whose persisted strict prefix is within GUFO_DISK_EVICT_SKIP_TOKENS tokens is covered by it (restore uses the
          // prefix and prefills the tail), so the ~1 GB write is skipped; logged as already_persisted/prefix_persisted and counted.
          bool covered = false;
          if (!exists && record && job.trigger == "evict" &&
              options.evict_skip_tokens.value() > 0) {
            const ScopedOperationPermit permit(operation_gate);
            const auto prefix = PersistedPrefixEntry(*job.runner, job.tokens, job.identity);
            covered = prefix.has_value() &&
                      job.tokens.size() - (*prefix)->tokens.size() <= options.evict_skip_tokens.value();
            // The covering entry now stands in for this snapshot: refresh its LRU position (as the exact-dedup Touch does) so disk LRU does not
            // evict it first and leave the session with nothing on disk (review F1).
            if (covered)
              TouchEntry(*prefix);
          }
          if (covered) {
            stored = true;
            Emit(ContinuationDiskEventAction::kAlreadyPersisted,
                 ContinuationDiskEventReason::kPrefixPersisted, 0,
                 job.snapshot->PayloadBytes(), job.tokens.size(), 0, 0,
                 job.trigger);
          } else if (exists) {
            stored = true;
          } else {
            stored = Save(*job.runner, job.tokens, *job.snapshot, job.identity,
                          job.shared_prefix, job.trigger)
                         .stored;
            if (!stored && record) {
              const ScopedOperationPermit permit(operation_gate);
              exists = Touch(*job.runner, job.tokens, job.identity);
              stored = exists;
            }
          }
          if (exists)
            Emit(ContinuationDiskEventAction::kAlreadyPersisted,
                 ContinuationDiskEventReason::kAlreadyPersisted, 0,
                 job.snapshot->PayloadBytes(), job.tokens.size(), 0, 0,
                 job.trigger);
        }
      } catch (...) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kIoFailure, 0, 0, job.tokens.size(),
             0, 0, job.trigger);
      }
      if (record)
        record->state.store(stored ? SnapshotPersistence::kPersisted
                                   : (stopping ? SnapshotPersistence::kAbandoned
                                               : SnapshotPersistence::kFailed));
      const auto released = job.retained_bytes;
      job = {};
      {
        std::lock_guard lock(queue_mutex);
        queued_bytes -= released;
      }
      queue_changed.notify_all();
    }
  }

  void ValidateOptions() {
    if (options.directory.empty()) {
      throw std::invalid_argument(
          "continuation disk directory must not be empty");
    }
    if (options.capacity_bytes == 0) {
      throw std::invalid_argument("continuation disk capacity must be nonzero");
    }
    if (options.staging_capacity_bytes == 0) {
      // The host snapshot budget already leaves half of available RAM free.
      // Disk staging gets a quarter of that budget. It allocates nothing
      // upfront, so a fixed cap would only refuse checkpoints larger than it.
      options.staging_capacity_bytes =
          std::min(options.capacity_bytes, HostSnapshotBudgetBytes() / 4);
    }
    if (options.staging_capacity_bytes < kHeaderBytes) {
      throw std::invalid_argument(
          "continuation disk staging capacity is too small");
    }
    if (!key_hash) {
      key_hash = DefaultKeyHash;
    }
  }

  void InitializeDirectory() {
    std::error_code error;
    const bool existed = std::filesystem::exists(options.directory, error);
    if (error) {
      throw std::runtime_error("failed to inspect continuation disk directory");
    }
    if (!existed) {
      std::filesystem::create_directories(options.directory, error);
      if (error) {
        throw std::runtime_error(
            "failed to create continuation disk directory");
      }
    }

    struct stat status{};
    if (::lstat(options.directory.c_str(), &status) != 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to inspect continuation disk directory"));
    }
    if (!S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) ||
        status.st_uid != ::geteuid()) {
      throw std::runtime_error(
          "continuation disk directory is not a private owned directory");
    }
    if (::chmod(options.directory.c_str(), S_IRWXU) != 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to secure continuation disk directory"));
    }

    directory_fd = ::open(options.directory.c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory_fd < 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to open continuation disk directory"));
    }
  }

  // Phase timings of the publish that is currently being reported (Save holds write_mutex across PublishImage and its Emit).
  mutable double phase_gather_ms_{0.0}, phase_write_ms_{0.0}, phase_hash_ms_{0.0}, phase_fsync_ms_{0.0};

  void Emit(ContinuationDiskEventAction action,
            ContinuationDiskEventReason reason, std::size_t file_bytes,
            std::size_t payload_bytes, std::size_t token_count,
            double elapsed_ms = 0.0, std::size_t staging_used_bytes = 0,
            std::string_view trigger = "write_through") const noexcept {
    if (action == ContinuationDiskEventAction::kSkipped &&
        (trigger == "evict" || trigger == "idle" || trigger == "shutdown") &&
        reason != ContinuationDiskEventReason::kExactReplacement)
      action = ContinuationDiskEventAction::kDroppedEvict;
    if (action == ContinuationDiskEventAction::kDroppedEvict)
      dropped_evict.fetch_add(1, std::memory_order_relaxed);
    if (action == ContinuationDiskEventAction::kAlreadyPersisted)
      already_persisted.fetch_add(1, std::memory_order_relaxed);
    if (action == ContinuationDiskEventAction::kSkipped)
      skipped->fetch_add(1, std::memory_order_relaxed);
    if (action == ContinuationDiskEventAction::kStored) {
      bytes_written.fetch_add(payload_bytes, std::memory_order_relaxed);
      const auto index = trigger == "evict"      ? 1
                         : trigger == "idle"     ? 2
                         : trigger == "shutdown" ? 3
                                                 : 0;
      persisted[index].fetch_add(1, std::memory_order_relaxed);
    }
    if (!event_sink) {
      return;
    }
    try {
      event_sink({
          .action = action,
          .reason = reason,
          .file_bytes = file_bytes,
          .payload_bytes = payload_bytes,
          .token_count = token_count,
          .retained_bytes = retained,
          .capacity_bytes = options.capacity_bytes,
          .staging_capacity_bytes = options.staging_capacity_bytes,
          .staging_used_bytes = staging_used_bytes,
          .elapsed_ms = elapsed_ms,
          .trigger = trigger,
          .slice_yields = action == ContinuationDiskEventAction::kStored
                              ? phase_slice_yields
                              : 0,
          .slice_wait_ms = action == ContinuationDiskEventAction::kStored
                               ? phase_slice_wait_ms
                               : 0,
          .phase_gather_ms = action == ContinuationDiskEventAction::kStored
                                 ? phase_gather_ms_
                                 : 0,
          .phase_write_ms = action == ContinuationDiskEventAction::kStored
                                ? phase_write_ms_
                                : 0,
          .phase_hash_ms = action == ContinuationDiskEventAction::kStored
                               ? phase_hash_ms_
                               : 0,
          .phase_fsync_ms = action == ContinuationDiskEventAction::kStored
                                ? phase_fsync_ms_
                                : 0,
      });
    } catch (...) {
      return;
    }
  }

  /// Called with queue_mutex held. Report capacity rejection before allocation
  /// or enqueue, where it would otherwise look like a successful cache capture.
  [[nodiscard]] bool CanStage(std::size_t bytes, std::size_t payload_bytes,
                              std::size_t token_count,
                              std::string_view trigger = "write_through") const {
    if (writer_exited ||
        (stopping && (!options.persist_evict_idle.value() ||
                      std::chrono::steady_clock::now() >= shutdown_deadline))) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kShutdownDeadline, bytes, payload_bytes,
           token_count, 0.0, 0, trigger);
      return false;
    }
    const auto used =
        queued_bytes + capture_bytes->load(std::memory_order_relaxed);
    if (bytes > options.capacity_bytes) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kByteCapacity, bytes, payload_bytes,
           token_count, 0.0, used, trigger);
      return false;
    }
    const auto limit =
        std::min(options.capacity_bytes, options.staging_capacity_bytes);
    if (bytes > limit || used > limit - bytes) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kStagingCapacity, bytes, payload_bytes,
           token_count, 0.0, used, trigger);
      return false;
    }
    return true;
  }

  [[nodiscard]] std::string Hash(std::span<const std::uint8_t> bytes) const {
    const std::string digest = key_hash(bytes);
    if (!IsLowerHexDigest(digest)) {
      throw std::runtime_error(
          "continuation disk key hash must be 64 lowercase hex characters");
    }
    return digest;
  }

  [[nodiscard]] std::string HashKey(
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> tokens) const {
    return Hash(MakeKeyBytes(persistence, tokens));
  }

  [[nodiscard]] bool SafeRegularFile(std::string_view filename,
                                     struct stat* status) const noexcept {
    if (filename.empty() || filename.find('/') != std::string_view::npos ||
        filename.find("..") != std::string_view::npos || status == nullptr) {
      return false;
    }
    if (::fstatat(directory_fd, std::string(filename).c_str(), status,
                  AT_SYMLINK_NOFOLLOW) != 0) {
      return false;
    }
    return S_ISREG(status->st_mode) && !S_ISLNK(status->st_mode) &&
           status->st_uid == ::geteuid() &&
           (status->st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
           (status->st_mode & S_IRUSR) != 0;
  }

  [[nodiscard]] bool ReadImage(std::string_view filename,
                               std::vector<std::uint8_t>* image,
                               ContinuationDiskEventReason* failure_reason,
                               std::size_t* file_bytes = nullptr) const {
    if (image == nullptr || failure_reason == nullptr) {
      return false;
    }
    struct stat status{};
    if (!SafeRegularFile(filename, &status)) {
      *failure_reason = ContinuationDiskEventReason::kUnsafeFile;
      return false;
    }
    if (file_bytes != nullptr && status.st_size > 0)
      *file_bytes = static_cast<std::size_t>(status.st_size);
    if (status.st_size <= 0 ||
        static_cast<std::uint64_t>(status.st_size) >
            static_cast<std::uint64_t>(options.staging_capacity_bytes)) {
      *failure_reason = status.st_size > 0
                            ? ContinuationDiskEventReason::kStagingCapacity
                            : ContinuationDiskEventReason::kCorrupt;
      return false;
    }
    const ScopedFileDescriptor file(
        ::openat(directory_fd, std::string(filename).c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!file) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    image->resize(static_cast<std::size_t>(status.st_size));
    if (!ReadAll(file.get(), *image)) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    std::uint8_t trailing = 0;
    while (true) {
      const ssize_t count = ::read(file.get(), &trailing, 1);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count != 0) {
        *failure_reason = count < 0 ? ContinuationDiskEventReason::kIoFailure
                                    : ContinuationDiskEventReason::kCorrupt;
        return false;
      }
      break;
    }
    return true;
  }

  void RemoveFileOnly(std::string_view filename) const noexcept {
    if (!filename.empty()) {
      (void)::unlinkat(directory_fd, std::string(filename).c_str(), 0);
    }
  }

  void IndexExistingFiles() {
    std::error_code error;
    const std::filesystem::directory_iterator iterator(
        options.directory,
        std::filesystem::directory_options::skip_permission_denied, error);
    if (error) {
      throw std::runtime_error(
          "failed to enumerate continuation disk directory");
    }
    for (const auto& directory_entry : iterator) {
      const std::string filename = directory_entry.path().filename().string();
      if (filename.starts_with(kTemporaryPrefix)) {
        // Publishers hold a shared directory lock from before temporary-file
        // creation through rename. Defer orphan cleanup while any is active;
        // readers must not wait for another store's potentially long write.
        const ScopedFileLock cleanup_lock(directory_fd, LOCK_EX | LOCK_NB);
        if (cleanup_lock)
          RemoveFileOnly(filename);
        continue;
      }
      if (!HasSuffix(filename, kFileSuffix)) {
        continue;
      }

      std::vector<std::uint8_t> image;
      ContinuationDiskEventReason failure_reason =
          ContinuationDiskEventReason::kCorrupt;
      ParsedImage parsed;
      std::size_t file_bytes = 0;
      if (!ReadImage(filename, &image, &failure_reason, &file_bytes)) {
        if (failure_reason == ContinuationDiskEventReason::kStagingCapacity) {
          // A smaller RAM budget must not destroy a previously valid cache.
          // Account the file for LRU/retention, but never index unverified
          // tokens. A restart with enough staging can validate and reuse it.
          auto access = directory_entry.last_write_time(error);
          if (error) {
            error.clear();
            access = std::filesystem::file_time_type::min();
          }
          entries.push_back({.filename = filename,
                             .key_hash = {},
                             .persistence = {},
                             .tokens = {},
                             .file_bytes = file_bytes,
                             .last_access = access});
          retained += file_bytes;
          retained_entries.fetch_add(1, std::memory_order_relaxed);
          Emit(ContinuationDiskEventAction::kSkipped, failure_reason,
               file_bytes, 0, 0);
          continue;
        }
        RemoveFileOnly(filename);
        Emit(ContinuationDiskEventAction::kRemoved, failure_reason, 0, 0, 0);
        continue;
      }
      const ParseFailure parse_failure = ParseAndVerifyImage(&image, &parsed);
      if (parse_failure != ParseFailure::kNone) {
        RemoveFileOnly(filename);
        Emit(ContinuationDiskEventAction::kRemoved,
             parse_failure == ParseFailure::kChecksum
                 ? ContinuationDiskEventReason::kChecksumMismatch
                 : ContinuationDiskEventReason::kCorrupt,
             image.size(), 0, 0);
        continue;
      }

      std::filesystem::file_time_type last_access =
          directory_entry.last_write_time(error);
      if (error) {
        error.clear();
        last_access = std::filesystem::file_time_type::min();
      }
      const std::string digest = HashKey(parsed.persistence, parsed.tokens);
      const EntryIterator duplicate =
          FindExact(digest, parsed.persistence, parsed.tokens);
      if (duplicate != entries.end()) {
        if (duplicate->last_access >= last_access) {
          RemoveFileOnly(filename);
          Emit(ContinuationDiskEventAction::kRemoved,
               ContinuationDiskEventReason::kExactReplacement, image.size(),
               parsed.payload_bytes, parsed.tokens.size());
          continue;
        }
        RemoveEntry(duplicate, ContinuationDiskEventReason::kExactReplacement);
      }
      entries.push_back({
          .filename = filename,
          .key_hash = digest,
          .persistence = std::move(parsed.persistence),
          .tokens = std::move(parsed.tokens),
          .file_bytes = image.size(),
          .payload_bytes = parsed.payload_bytes,
          .last_access = last_access,
      });
      const EntryIterator added = std::prev(entries.end());
      index.emplace(added->key_hash, added);
      prefixes[PrefixKey(added->persistence)].Insert(added->tokens, added);
      retained += added->file_bytes;
      retained_entries.fetch_add(1, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] EntryIterator FindExact(
      const std::string& digest,
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> tokens) {
    const auto [begin, end] = index.equal_range(digest);
    for (auto current = begin; current != end; ++current) {
      const EntryIterator entry = current->second;
      if (entry->persistence == persistence &&
          std::ranges::equal(entry->tokens, tokens)) {
        return entry;
      }
    }
    return entries.end();
  }

  void EraseIndex(EntryIterator entry) {
    const auto prefix = prefixes.find(PrefixKey(entry->persistence));
    if (prefix != prefixes.end()) {
      prefix->second.Erase(entry->tokens);
      if (prefix->second.children.empty() && !prefix->second.entry) {
        prefixes.erase(prefix);
      }
    }
    const auto [begin, end] = index.equal_range(entry->key_hash);
    for (auto current = begin; current != end; ++current) {
      if (current->second == entry) {
        index.erase(current);
        return;
      }
    }
  }

  bool RemoveEntry(EntryIterator entry,
                   ContinuationDiskEventReason reason) noexcept {
    if (entry == entries.end()) {
      return true;
    }
    if (::unlinkat(directory_fd, entry->filename.c_str(), 0) != 0 &&
        errno != ENOENT) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kIoFailure, entry->file_bytes,
           entry->payload_bytes, entry->tokens.size());
      return false;
    }
    const std::size_t file_bytes = entry->file_bytes;
    const std::size_t payload_bytes = entry->payload_bytes;
    const std::size_t token_count = entry->tokens.size();
    EraseIndex(entry);
    retained -= file_bytes;
    entries.erase(entry);
    retained_entries.fetch_sub(1, std::memory_order_relaxed);
    Emit(ContinuationDiskEventAction::kRemoved, reason, file_bytes,
         payload_bytes, token_count);
    return true;
  }

  [[nodiscard]] EntryIterator LeastRecentlyUsed() {
    EntryIterator selected = entries.end();
    for (auto current = entries.begin(); current != entries.end(); ++current) {
      if (selected == entries.end() ||
          current->last_access < selected->last_access ||
          (current->last_access == selected->last_access &&
           current->filename < selected->filename)) {
        selected = current;
      }
    }
    return selected;
  }

  void EvictToCapacity() {
    while (retained > options.capacity_bytes) {
      const EntryIterator victim = LeastRecentlyUsed();
      if (victim == entries.end() ||
          !RemoveEntry(victim, ContinuationDiskEventReason::kLru)) {
        break;
      }
    }
  }

  bool MakeCapacity(std::size_t file_bytes) {
    if (file_bytes > options.capacity_bytes) {
      return false;
    }
    while (retained > options.capacity_bytes - file_bytes) {
      const EntryIterator victim = LeastRecentlyUsed();
      if (victim == entries.end() ||
          !RemoveEntry(victim, ContinuationDiskEventReason::kLru)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::string UniqueSuffix() {
    const std::uint64_t random_value =
        (static_cast<std::uint64_t>(random_device()) << 32U) ^
        static_cast<std::uint64_t>(random_device()) ^
        static_cast<std::uint64_t>(++unique_counter) ^
        static_cast<std::uint64_t>(::getpid());
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << random_value;
    return output.str();
  }

  [[nodiscard]] std::string NewFinalFilename(const std::string& digest) {
    for (int attempt = 0; attempt < 64; ++attempt) {
      const std::string filename =
          digest + "-" + UniqueSuffix() + std::string(kFileSuffix);
      struct stat status{};
      if (::fstatat(directory_fd, filename.c_str(), &status,
                    AT_SYMLINK_NOFOLLOW) != 0 &&
          errno == ENOENT) {
        return filename;
      }
    }
    throw std::runtime_error("failed to allocate a continuation disk filename");
  }

  [[nodiscard]] bool PublishImage(std::string_view final_filename,
                                  std::span<const std::uint8_t> header,
                                  const TextModelRunner& runner,
                                  const TextRunnerSnapshot& snapshot,
                                  std::size_t payload_bytes) {
    // Save serializes publishers in this instance with write_mutex, and
    // startup indexing completes before its worker starts. Thus no two lock
    // owners in this instance can share directory_fd's open file description.
    const ScopedFileLock publication_lock(directory_fd, LOCK_SH);
    if (!publication_lock)
      return false;
    const std::string temporary_filename =
        std::string(kTemporaryPrefix) + UniqueSuffix();
    ScopedFileDescriptor temporary(
        ::openat(directory_fd, temporary_filename.c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                 S_IRUSR | S_IWUSR));
    if (!temporary) {
      return false;
    }
    bool valid = true;
    try {
      crypto::Sha256Hasher hasher;
      std::size_t written = 0;
      using Clock = std::chrono::steady_clock;
      const auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
      double write_ms = 0.0, hash_ms = 0.0;
      phase_gather_ms_ = phase_write_ms_ = phase_hash_ms_ = phase_fsync_ms_ = 0.0;
      phase_slice_yields = 0;
      phase_slice_wait_ms = 0;
      std::size_t slice_bytes = 0;
      const auto sink = [&](std::span<const std::uint8_t> bytes) {
        // Keep write sizes bounded and checksum each byte exactly once.
        while (!bytes.empty()) {
          const auto chunk =
              bytes.first(std::min(bytes.size(), std::size_t{1024 * 1024}));
          const auto t0 = Clock::now();
          if (!WriteAll(temporary.get(), chunk)) {
            throw std::runtime_error("continuation disk write failed");
          }
          const auto t1 = Clock::now();
          hasher.Update(chunk);
          write_ms += ms(t1 - t0);
          hash_ms += ms(Clock::now() - t1);
          bytes = bytes.subspan(chunk.size());
          if (options.slice.value()) {
            slice_bytes += chunk.size();
            if (slice_bytes >= 16 * 1024 * 1024) {
              slice_bytes = 0;
              while (active_generations.load(std::memory_order_acquire) != 0 &&
                     phase_slice_wait_ms < options.slice_max_wait.count() &&
                     !stopping) {
                const auto wait_start = Clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                ++phase_slice_yields;
                phase_slice_wait_ms += ms(Clock::now() - wait_start);
              }
            }
          }
        }
      };
      sink(header);
      const auto stream_start = Clock::now();
      runner.StreamPersistentSnapshot(snapshot, [&](auto bytes) {
        if (bytes.size() > payload_bytes - written) {
          throw std::runtime_error("snapshot serialization overflow");
        }
        sink(bytes);
        written += bytes.size();
      });
      if (written != payload_bytes) {
        throw std::runtime_error("snapshot serialization truncated");
      }
      phase_write_ms_ = write_ms;
      phase_hash_ms_ = hash_ms;
      phase_gather_ms_ =
          std::max(0.0, ms(Clock::now() - stream_start) - write_ms - hash_ms -
                            phase_slice_wait_ms);
      const auto checksum = hasher.FinishHex();
      if (::lseek(temporary.get(), kChecksumOffset, SEEK_SET) < 0 ||
          !WriteAll(temporary.get(),
                    {reinterpret_cast<const std::uint8_t*>(checksum.data()),
                     checksum.size()})) {
        throw std::runtime_error("snapshot checksum write failed");
      }
    } catch (...) {
      valid = false;
    }
    const auto fsync_start = std::chrono::steady_clock::now();
    valid = valid && ::fsync(temporary.get()) == 0;
    const int raw_descriptor = temporary.release();
    if (::close(raw_descriptor) != 0) {
      valid = false;
    }
    if (!valid) {
      RemoveFileOnly(temporary_filename);
      return false;
    }
    if (::renameat(directory_fd, temporary_filename.c_str(), directory_fd,
                   std::string(final_filename).c_str()) != 0) {
      RemoveFileOnly(temporary_filename);
      return false;
    }
    if (::fsync(directory_fd) != 0) {
      RemoveFileOnly(final_filename);
      (void)::fsync(directory_fd);
      return false;
    }
    phase_fsync_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fsync_start).count();
    return true;
  }

  [[nodiscard]] SaveResult Save(
      const TextModelRunner& runner,
      std::span<const TextRunnerToken> checkpoint_tokens,
      const TextRunnerSnapshot& snapshot,
      std::span<const std::uint8_t> input_identity, bool shared_prefix = false,
      std::string_view trigger = "write_through") {
    // Serialize writers, but keep existing entries readable during payload
    // serialization, hashing and filesystem durability operations.
    const std::lock_guard write_lock(write_mutex);
    ScopedOperationPermit metadata_lock(operation_gate);
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kUnsupported, 0, 0,
           checkpoint_tokens.size(), 0, 0, trigger);
      return {};
    }
    if (checkpoint_tokens.empty()) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kSerializationFailure, 0, 0, 0, 0, 0,
           trigger);
      return {};
    }
    // The same prefix is offered again whenever a restored continuation is
    // committed or several conversations share one system prompt. The stored
    // file already represents these tokens; refresh its recency and skip the
    // serialization and write.
    const std::string digest =
        HashKey(*descriptor.persistence, checkpoint_tokens);
    const EntryIterator existing =
        FindExact(digest, *descriptor.persistence, checkpoint_tokens);
    if (existing != entries.end()) {
      TouchEntry(existing);
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kExactReplacement, existing->file_bytes,
           existing->payload_bytes, checkpoint_tokens.size(), 0, 0, trigger);
      return {};
    }

    // A continuation advances by a few hundred tokens per turn, so persisting
    // every turn rewrites a largely identical snapshot. Skipping writes that
    // barely move past a stored prefix trades that write for re-prefilling the
    // short gap, which the caller does anyway when no checkpoint matches.
    //
    // Shared prefixes are exempt: they exist to give a new conversation a place
    // to start from, and they sit only a few tokens past a deeper entry by
    // construction.
    if (!shared_prefix &&
        WithinCheckpointStep(runner, checkpoint_tokens, input_identity)) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kMinStep, 0, 0,
           checkpoint_tokens.size(), 0, 0, trigger);
      return {};
    }

    std::size_t payload_bytes = 0;
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> header;
    std::size_t file_bytes = 0;
    try {
      payload_bytes = runner.PersistentSnapshotPayloadBytes(snapshot);
      if (payload_bytes == 0) {
        throw std::runtime_error("persistent snapshot payload is empty");
      }
      file_bytes = CheckedFileBytes(
          descriptor.persistence->compatibility_identity.size(),
          checkpoint_tokens.size(), payload_bytes);
      if (file_bytes > options.staging_capacity_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kStagingCapacity, file_bytes,
             payload_bytes, checkpoint_tokens.size(), 0, 0, trigger);
        return {};
      }
      if (file_bytes > options.capacity_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kByteCapacity, file_bytes,
             payload_bytes, checkpoint_tokens.size(), 0, 0, trigger);
        return {};
      }
      header = BuildHeader(*descriptor.persistence, checkpoint_tokens,
                           payload_bytes);
    } catch (...) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kSerializationFailure, file_bytes,
           payload_bytes, checkpoint_tokens.size(), 0, 0, trigger);
      return {};
    }

    const std::string filename = NewFinalFilename(digest);
    metadata_lock.Unlock();
    const bool published =
        PublishImage(filename, header, runner, snapshot, payload_bytes);
    metadata_lock.Lock();
    if (!published) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kIoFailure, file_bytes, payload_bytes,
           checkpoint_tokens.size(), 0, 0, trigger);
      return {};
    }
    // No visible entry is evicted until its replacement is fully durable.
    if (!MakeCapacity(file_bytes)) {
      RemoveFileOnly(filename);
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kByteCapacity, file_bytes,
           payload_bytes, checkpoint_tokens.size(), 0, 0, trigger);
      return {};
    }

    entries.push_back({
        .filename = filename,
        .key_hash = digest,
        .persistence = *descriptor.persistence,
        .tokens = std::vector<TextRunnerToken>(checkpoint_tokens.begin(),
                                               checkpoint_tokens.end()),
        .file_bytes = file_bytes,
        .payload_bytes = payload_bytes,
        .last_access = std::filesystem::file_time_type::clock::now(),
    });
    const EntryIterator added = std::prev(entries.end());
    index.emplace(digest, added);
    prefixes[PrefixKey(added->persistence)].Insert(added->tokens, added);
    retained += file_bytes;
    retained_entries.fetch_add(1, std::memory_order_relaxed);
    Emit(ContinuationDiskEventAction::kStored,
         ContinuationDiskEventReason::kSaved, file_bytes, payload_bytes,
         checkpoint_tokens.size(),
         std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - started)
             .count(),
         0, trigger);
    return {
        .stored = true,
        .file_bytes = file_bytes,
        .payload_bytes = payload_bytes,
    };
  }

  [[nodiscard]] EntryIterator FindLongestCandidate(
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> prompt) {
    const auto root = prefixes.find(PrefixKey(persistence));
    if (root == prefixes.end())
      return entries.end();
    return root->second.Find(prompt).longest.value_or(entries.end());
  }

  [[nodiscard]] EntryIterator FindLongestInputCandidate(
      const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
      std::span<const std::uint8_t> input_identity,
      std::span<const ContinuationInputPrefix> input_prefixes) {
    auto best = entries.end();
    VisitInputRanges(
        prompt.size(), input_identity, input_prefixes,
        [&](auto begin, auto end, auto identity) {
          const auto descriptor = DescriptorForInput(runner, identity);
          const auto candidate =
              FindLongestCandidate(*descriptor.persistence, prompt.first(end));
          if (candidate != entries.end() && candidate->tokens.size() >= begin &&
              (best == entries.end() ||
               candidate->tokens.size() > best->tokens.size()))
            best = candidate;
        });
    return best;
  }

  [[nodiscard]] RestoreResult RestoreLongestPrefix(
      const TextModelRunner& runner, TextRunnerState& state,
      std::span<const TextRunnerToken> prompt,
      std::span<const std::uint8_t> input_identity,
      std::size_t stable_prefix_tokens,
      std::span<const ContinuationInputPrefix> input_prefixes) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      Emit(ContinuationDiskEventAction::kMiss,
           ContinuationDiskEventReason::kUnsupported, 0, 0, 0);
      return {};
    }

    while (true) {
      const EntryIterator candidate = FindLongestInputCandidate(
          runner, prompt, input_identity, input_prefixes);
      if (candidate == entries.end()) {
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kNotFound, 0, 0, 0);
        return {};
      }
      if (stable_prefix_tokens != 0 &&
          candidate->tokens.size() > stable_prefix_tokens &&
          FindLongestInputCandidate(runner, prompt.first(stable_prefix_tokens),
                                    input_identity,
                                    input_prefixes) == entries.end()) {
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kNotFound, 0, 0, 0);
        return {};
      }

      std::vector<std::uint8_t> image;
      ContinuationDiskEventReason failure_reason =
          ContinuationDiskEventReason::kCorrupt;
      if (!ReadImage(candidate->filename, &image, &failure_reason)) {
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        const bool removed = RemoveEntry(candidate, failure_reason);
        Emit(ContinuationDiskEventAction::kMiss, failure_reason, file_bytes,
             payload_bytes, token_count);
        if (!removed)
          return {};
        continue;
      }

      ParsedImage parsed;
      const ParseFailure parse_failure = ParseAndVerifyImage(&image, &parsed);
      if (parse_failure != ParseFailure::kNone ||
          parsed.persistence != candidate->persistence ||
          parsed.tokens != candidate->tokens ||
          parsed.payload_bytes != candidate->payload_bytes) {
        const ContinuationDiskEventReason reason =
            parse_failure == ParseFailure::kChecksum
                ? ContinuationDiskEventReason::kChecksumMismatch
                : ContinuationDiskEventReason::kCorrupt;
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        const bool removed = RemoveEntry(candidate, reason);
        Emit(ContinuationDiskEventAction::kMiss, reason, file_bytes,
             payload_bytes, token_count);
        if (!removed)
          return {};
        continue;
      }

      try {
        runner.RestorePersistentSnapshot(
            state, std::span<const std::uint8_t>(image).subspan(
                       parsed.payload_offset, parsed.payload_bytes));
      } catch (...) {
        state.Invalidate();
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        (void)RemoveEntry(candidate,
                          ContinuationDiskEventReason::kRestoreFailure);
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kRestoreFailure, file_bytes,
             payload_bytes, token_count);
        return {};
      }

      TouchEntry(candidate);
      Emit(ContinuationDiskEventAction::kRestored,
           ContinuationDiskEventReason::kHit, candidate->file_bytes,
           candidate->payload_bytes, candidate->tokens.size());
      return {
          .restored = true,
          .token_count = candidate->tokens.size(),
          .file_bytes = candidate->file_bytes,
          .payload_bytes = candidate->payload_bytes,
      };
    }
  }

  void TouchEntry(EntryIterator entry) {
    entry->last_access = std::filesystem::file_time_type::clock::now();
    (void)::utimensat(directory_fd, entry->filename.c_str(), nullptr,
                      AT_SYMLINK_NOFOLLOW);
  }

  [[nodiscard]] std::vector<std::size_t> SharedPrefixBoundaries(
      const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
      std::size_t min_tokens, std::size_t max_boundaries,
      std::span<const std::uint8_t> input_identity,
      std::span<const ContinuationInputPrefix> input_prefixes) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value() || max_boundaries == 0) {
      return {};
    }
    std::vector<std::size_t> boundaries;
    VisitInputRanges(
        prompt.size(), input_identity, input_prefixes,
        [&](auto begin, auto end, auto identity) {
          const auto scoped = DescriptorForInput(runner, identity);
          const auto root = prefixes.find(PrefixKey(*scoped.persistence));
          if (root == prefixes.end())
            return;
          for (const auto count :
               root->second.Find(prompt.first(end), true).shared_boundaries) {
            if (count >= begin && count < prompt.size())
              boundaries.push_back(count);
          }
        });
    std::ranges::sort(boundaries);
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                     boundaries.end());
    // Prefill chunk boundaries are numerically free only at multiples of 256 tokens (lead 1 quality gate T2b, 2026-10-06: grids 3584 / 3840 identical to
    // 4096, 4160 / 3500 not): a split at any other position changes the resulting state and so the later output. Round every learned boundary DOWN to a
    // multiple of the alignment (cost < 256 re-prefilled tokens per hit) so a disk-cache run stays bit-identical to a run without it.
    // GUFO_DISK_BOUNDARY_ALIGN=0 (or 1) disables the rounding (A/B), any other value N rounds down to a multiple of N.
    const std::size_t boundary_align = [] {   // read per call: tests switch GUFO_DISK_BOUNDARY_ALIGN
      const char* e = std::getenv("GUFO_DISK_BOUNDARY_ALIGN");
      if (e == nullptr || *e == '\0') return std::size_t{256};
      const auto v = std::strtoull(e, nullptr, 10);
      return static_cast<std::size_t>(v);
    }();
    if (boundary_align > 1) {
      for (auto& length : boundaries) length -= length % boundary_align;
      std::ranges::sort(boundaries);
      boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                       boundaries.end());
    }
    std::erase_if(boundaries, [min_tokens](auto length) {
      return length == 0 || length < min_tokens;
    });
    // Keep the longest ones: they save the most prefill when they hit.
    if (boundaries.size() > max_boundaries) {
      boundaries.erase(
          boundaries.begin(),
          boundaries.end() - static_cast<std::ptrdiff_t>(max_boundaries));
    }
    return boundaries;
  }

  [[nodiscard]] bool WithinCheckpointStep(
      const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
      std::span<const std::uint8_t> input_identity) {
    if (options.min_checkpoint_step_tokens == 0 ||
        !DescriptorForInput(runner, input_identity).persistence.has_value())
      return false;
    const auto base =
        FindLongestInputCandidate(runner, tokens, input_identity, {});
    return base != entries.end() && base->tokens.size() < tokens.size() &&
           tokens.size() - base->tokens.size() <
               options.min_checkpoint_step_tokens;
  }

  [[nodiscard]] bool Touch(const TextModelRunner& runner,
                           std::span<const TextRunnerToken> tokens,
                           std::span<const std::uint8_t> input_identity) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      return false;
    }
    const std::string digest = HashKey(*descriptor.persistence, tokens);
    const EntryIterator entry =
        FindExact(digest, *descriptor.persistence, tokens);
    if (entry == entries.end()) {
      return false;
    }
    TouchEntry(entry);
    return true;
  }

  // Length in tokens of the longest persisted entry (same compatibility identity) that is a STRICT prefix of `tokens`; 0 when none. Caller holds the
  // operation gate. Used by the evict-skip policy.
  [[nodiscard]] std::optional<EntryIterator> PersistedPrefixEntry(
      const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
      std::span<const std::uint8_t> input_identity) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      return std::nullopt;
    }
    const auto found = prefixes.find(PrefixKey(*descriptor.persistence));
    if (found == prefixes.end()) {
      return std::nullopt;
    }
    const auto match = found->second.Find(tokens);
    if (!match.longest || (*match.longest)->tokens.size() >= tokens.size()) {
      return std::nullopt;
    }
    return match.longest;
  }

  ContinuationDiskStoreOptions options;
  EventSink event_sink;
  KeyHashFunction key_hash;
  int directory_fd{-1};
  mutable std::binary_semaphore operation_gate{1};
  std::mutex write_mutex;
  std::list<Entry> entries;
  std::unordered_multimap<std::string, EntryIterator> index;
  std::map<PersistenceKey, PrefixNode> prefixes;
  std::atomic<std::size_t> retained{0};
  std::atomic<std::size_t> retained_entries{0};
  std::random_device random_device;
  std::uint64_t unique_counter{0};
  mutable std::mutex queue_mutex;
  std::condition_variable queue_changed;
  std::deque<PendingSave> pending;
  std::list<PendingSave> evict_backlog;
  std::size_t backlog_bytes{0};
  bool writer_exited{false};  // queue_mutex serializes exit against offers.
  std::size_t queued_bytes{0};
  std::shared_ptr<std::atomic<std::size_t>> capture_bytes =
      std::make_shared<std::atomic<std::size_t>>(0);
  const bool defer_enabled{DiskDeferEnabled()};
  std::atomic<std::size_t> active_generations{0};
  std::size_t flush_waiters{0};
  std::atomic<bool> stopping{false};
  std::chrono::steady_clock::time_point shutdown_deadline;
  std::atomic<std::int64_t> last_activity{ActivityTime()};
  std::shared_ptr<const TextModelRunner> persistence_runner;
  PersistenceSource persistence_source;
  std::atomic<std::size_t> captures{0};
  std::shared_ptr<std::atomic<std::size_t>> skipped =
      std::make_shared<std::atomic<std::size_t>>(0);
  mutable std::array<std::atomic<std::size_t>, 4> persisted{};
  mutable std::atomic<std::size_t> bytes_written{0};
  std::shared_ptr<std::atomic<std::size_t>> superseded;
  mutable std::atomic<std::size_t> dropped_evict{0}, already_persisted{0};
  mutable std::size_t phase_slice_yields{0};
  mutable double phase_slice_wait_ms{0};
  std::size_t SaveAsync(std::shared_ptr<const TextModelRunner> runner,
                        std::vector<TextRunnerToken> tokens,
                        std::shared_ptr<const TextRunnerSnapshot> snapshot,
                        std::vector<std::uint8_t> identity,
                        std::unique_ptr<CaptureReservation> reservation,
                        bool shared_prefix, std::string_view trigger);
  void EnqueuePersistence(std::shared_ptr<const ContinuationSnapshot> snapshot,
                          std::string_view trigger);
  std::thread writer;
};

ContinuationDiskStore::ContinuationDiskStore(
    ContinuationDiskStoreOptions options, EventSink event_sink,
    KeyHashFunction key_hash)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(event_sink),
                                   std::move(key_hash))) {}

ContinuationDiskStore::~ContinuationDiskStore() = default;

ContinuationDiskStore::GenerationActivity::GenerationActivity(Impl* owner)
    : owner_(owner) {
  if (owner_)
    owner_->active_generations.fetch_add(1, std::memory_order_acq_rel);
}

ContinuationDiskStore::GenerationActivity::~GenerationActivity() {
  if (owner_) {
    if (owner_->options.persist_evict_idle.value())
      owner_->last_activity.store(ActivityTime(), std::memory_order_release);
    owner_->active_generations.fetch_sub(1, std::memory_order_acq_rel);
    owner_->queue_changed.notify_all();
  }
}

ContinuationDiskStore::GenerationActivity
ContinuationDiskStore::BeginGeneration() {
  return GenerationActivity(impl_->defer_enabled ||
                                    impl_->options.slice.value() ||
                                    impl_->options.persist_evict_idle.value()
                                ? impl_.get()
                                : nullptr);
}

ContinuationDiskStore::GenerationActivity
ContinuationDiskStore::BeginAdmission() {
  return GenerationActivity(
      impl_->options.persist_evict_idle.value() ? impl_.get() : nullptr);
}

ContinuationDiskStore::CaptureReservation::~CaptureReservation() {
  counter_->fetch_sub(bytes_, std::memory_order_relaxed);
}

std::unique_ptr<ContinuationDiskStore::CaptureReservation>
ContinuationDiskStore::ReserveCapture(
    const TextModelRunner& runner, std::size_t token_count,
    std::size_t snapshot_bytes, std::span<const std::uint8_t> input_identity) {
  const auto descriptor = DescriptorForInput(runner, input_identity);
  if (!descriptor.persistence || token_count == 0 || snapshot_bytes == 0)
    return {};
  const auto bytes =
      CheckedFileBytes(descriptor.persistence->compatibility_identity.size(),
                       token_count, snapshot_bytes);
  std::lock_guard lock(impl_->queue_mutex);
  if (!impl_->CanStage(bytes, snapshot_bytes, token_count))
    return {};
  auto reservation = std::unique_ptr<CaptureReservation>(
      new CaptureReservation(impl_->capture_bytes, bytes));
  impl_->capture_bytes->fetch_add(bytes, std::memory_order_relaxed);
  return reservation;
}

bool ContinuationDiskStore::CanSave(
    const TextModelRunner& runner, std::size_t token_count,
    std::size_t snapshot_bytes,
    std::span<const std::uint8_t> input_identity) const {
  const auto descriptor = DescriptorForInput(runner, input_identity);
  if (!descriptor.persistence || token_count == 0 || snapshot_bytes == 0) {
    return false;
  }
  const auto bytes =
      CheckedFileBytes(descriptor.persistence->compatibility_identity.size(),
                       token_count, snapshot_bytes);
  std::lock_guard lock(impl_->queue_mutex);
  return impl_->CanStage(bytes, snapshot_bytes, token_count);
}

std::size_t ContinuationDiskStore::SaveAsync(
    std::shared_ptr<const TextModelRunner> runner,
    std::vector<TextRunnerToken> checkpoint_tokens,
    std::shared_ptr<const TextRunnerSnapshot> snapshot,
    std::vector<std::uint8_t> input_identity,
    std::unique_ptr<CaptureReservation> reservation, bool shared_prefix,
    std::string_view trigger) {
  return impl_->SaveAsync(std::move(runner), std::move(checkpoint_tokens),
                          std::move(snapshot), std::move(input_identity),
                          std::move(reservation), shared_prefix, trigger);
}

std::size_t ContinuationDiskStore::Impl::SaveAsync(
    std::shared_ptr<const TextModelRunner> runner,
    std::vector<TextRunnerToken> checkpoint_tokens,
    std::shared_ptr<const TextRunnerSnapshot> snapshot,
    std::vector<std::uint8_t> input_identity,
    std::unique_ptr<CaptureReservation> reservation, bool shared_prefix,
    std::string_view trigger) {
  if (!runner || !snapshot || checkpoint_tokens.empty()) {
    Emit(ContinuationDiskEventAction::kSkipped,
         ContinuationDiskEventReason::kSerializationFailure, 0, 0,
         checkpoint_tokens.size(), 0, 0, trigger);
    return 0;
  }
  const auto descriptor = DescriptorForInput(*runner, input_identity);
  if (!descriptor.persistence) {
    Emit(ContinuationDiskEventAction::kSkipped,
         ContinuationDiskEventReason::kUnsupported, 0, 0,
         checkpoint_tokens.size(), 0, 0, trigger);
    return 0;
  }
  const auto file_bytes =
      CheckedFileBytes(descriptor.persistence->compatibility_identity.size(),
                       checkpoint_tokens.size(),
                       runner->PersistentSnapshotPayloadBytes(*snapshot));
  const auto charge = CheckedFileBytes(
      descriptor.persistence->compatibility_identity.size(),
      checkpoint_tokens.size(),
      std::max(snapshot->PayloadBytes(),
               runner->PersistentSnapshotPayloadBytes(*snapshot)));
  {
    std::lock_guard lock(queue_mutex);
    if (trigger == "idle" &&
        active_generations.load(std::memory_order_acquire) != 0) {
      if (snapshot->persistence)
        snapshot->persistence->state.store(SnapshotPersistence::kDirty);
      return 0;
    }
    if (reservation) {
      if (reservation->counter_ != capture_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kSerializationFailure, 0, 0,
             checkpoint_tokens.size(), 0, 0, trigger);
        return 0;
      }
      reservation.reset();
    }
    if (options.persist_evict_idle.value() && trigger == "evict") {
      const auto limit =
          std::min(options.capacity_bytes, options.staging_capacity_bytes);
      if (writer_exited ||
          (stopping && std::chrono::steady_clock::now() >= shutdown_deadline)) {
        DropEvict(snapshot, writer_exited
                                ? ContinuationDiskEventReason::kWriterStopped
                                : ContinuationDiskEventReason::kShutdownDeadline);
        return 0;
      }
      if (charge > limit) {
        DropEvict(snapshot, charge > options.capacity_bytes
                                ? ContinuationDiskEventReason::kByteCapacity
                                : ContinuationDiskEventReason::kStagingCapacity);
        return 0;
      }
      const auto used = queued_bytes + capture_bytes->load();
      if (!evict_backlog.empty() || used > limit - charge) {
        const auto bound = options.evict_backlog_bytes.value();
        if (evict_backlog.size() >= options.evict_backlog_count ||
            charge > bound || backlog_bytes > bound - charge) {
          DropEvict(snapshot, ContinuationDiskEventReason::kEvictBacklogCapacity);
          return 0;
        }
        evict_backlog.push_back({std::move(runner), std::move(checkpoint_tokens),
                                std::move(snapshot), std::move(input_identity),
                                charge, shared_prefix,
                                std::chrono::steady_clock::now(), trigger});
        backlog_bytes += charge;
        queue_changed.notify_all();
        return file_bytes;
      }
    } else if (!CanStage(charge, snapshot->PayloadBytes(),
                         checkpoint_tokens.size(), trigger)) {
      return 0;
    }
    pending.push_back({std::move(runner), std::move(checkpoint_tokens),
                       std::move(snapshot), std::move(input_identity), charge,
                       shared_prefix,
                       defer_enabled ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{},
                       trigger});
    queued_bytes += charge;
  }
  queue_changed.notify_one();
  return file_bytes;
}

bool ContinuationDiskStore::persist_on_evict_idle() const noexcept {
  return impl_->options.persist_evict_idle.value();
}
void ContinuationDiskStore::SetPersistenceSource(
    std::shared_ptr<const TextModelRunner> runner, PersistenceSource source) {
  std::lock_guard lock(impl_->queue_mutex);
  impl_->persistence_runner = std::move(runner);
  impl_->persistence_source = std::move(source);
  impl_->queue_changed.notify_all();
}
void ContinuationDiskStore::Shutdown() {
  impl_->Shutdown();
}
void ContinuationDiskStore::PersistEvicted(
    std::shared_ptr<const ContinuationSnapshot> snapshot) {
  if (persist_on_evict_idle())
    impl_->EnqueuePersistence(std::move(snapshot), "evict");
}

std::size_t ContinuationDiskStore::Capture(
    std::shared_ptr<const TextModelRunner> runner,
    std::vector<TextRunnerToken> tokens,
    std::shared_ptr<const TextRunnerSnapshot> snapshot,
    std::vector<std::uint8_t> identity,
    std::unique_ptr<CaptureReservation> reservation, bool shared_prefix,
    bool already_persisted, bool retained_in_ram) {
  if (snapshot)
    impl_->captures.fetch_add(1, std::memory_order_relaxed);
  const auto count = tokens.size();
  try {
    if (!persist_on_evict_idle())
      return already_persisted
                 ? 0
                 : SaveAsync(std::move(runner), std::move(tokens),
                             std::move(snapshot), std::move(identity),
                             std::move(reservation), shared_prefix);
    reservation.reset();
    if (!snapshot) {
      impl_->Emit(ContinuationDiskEventAction::kSkipped,
                  ContinuationDiskEventReason::kRamNotRetained, 0,
                  snapshot ? snapshot->PayloadBytes() : 0, tokens.size());
      return 0;
    }
    auto record = std::make_shared<SnapshotPersistence>();
    record->tokens = std::move(tokens);
    record->input_identity = std::move(identity);
    record->shared_prefix = shared_prefix;
    record->state.store(already_persisted ? SnapshotPersistence::kPersisted
                                          : SnapshotPersistence::kDirty);
    auto sink = impl_->event_sink;
    auto skipped = impl_->skipped;
    const auto count = record->tokens.size(), bytes = snapshot->PayloadBytes();
    record->on_discard = [sink, skipped, count, bytes] {
      skipped->fetch_add(1, std::memory_order_relaxed);
      if (sink)
        sink({.action = ContinuationDiskEventAction::kSkipped,
              .reason = ContinuationDiskEventReason::kRamNotRetained,
              .payload_bytes = bytes,
              .token_count = count});
    };
    auto superseded = impl_->superseded;
    record->on_superseded = [sink, superseded, count, bytes] {
      superseded->fetch_add(1, std::memory_order_relaxed);
      if (sink)
        sink({.action = ContinuationDiskEventAction::kSuperseded,
              .reason = ContinuationDiskEventReason::kSuperseded,
              .payload_bytes = bytes, .token_count = count, .trigger = "evict"});
    };
    snapshot->persistence = std::move(record);
    impl_->queue_changed.notify_all();
    if (!retained_in_ram && !already_persisted)
      impl_->EnqueuePersistence(snapshot, "evict");
    return 0;
  } catch (...) {
    impl_->Emit(ContinuationDiskEventAction::kSkipped,
                ContinuationDiskEventReason::kSerializationFailure, 0, 0,
                count);
    return 0;
  }
}

void ContinuationDiskStore::Impl::EnqueuePersistence(
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::string_view trigger) {
  if (!snapshot || !snapshot->persistence)
    return;
  auto record = snapshot->persistence;
  auto state = record->state.load();
  if ((state != SnapshotPersistence::kDirty &&
       state != SnapshotPersistence::kFailed) ||
      !record->state.compare_exchange_strong(state,
                                             SnapshotPersistence::kQueued))
    return;
  bool accepted = false;
  try {
    auto text = std::dynamic_pointer_cast<const TextRunnerSnapshot>(snapshot);
    if (!persistence_runner || !text)
      throw std::runtime_error("persistent RAM snapshot has no text runner");
    accepted = SaveAsync(persistence_runner, record->tokens, std::move(text),
                         record->input_identity, {}, record->shared_prefix,
                         trigger) != 0;
  } catch (...) {
    Emit(ContinuationDiskEventAction::kSkipped,
         ContinuationDiskEventReason::kSerializationFailure, 0,
         snapshot->PayloadBytes(), record->tokens.size(), 0, 0, trigger);
  }
  if (!accepted && record->state.load() == SnapshotPersistence::kQueued)
    record->state.store(stopping ? SnapshotPersistence::kAbandoned
                                 : SnapshotPersistence::kFailed);
  queue_changed.notify_all();
}

void ContinuationDiskStore::Flush() {
  std::unique_lock lock(impl_->queue_mutex);
  if (impl_->defer_enabled || impl_->options.persist_evict_idle.value()) {
    ++impl_->flush_waiters;
    impl_->queue_changed.notify_all();
  }
  impl_->queue_changed.wait(lock, [this] {
    return impl_->queued_bytes == 0 && impl_->evict_backlog.empty();
  });
  if (impl_->defer_enabled || impl_->options.persist_evict_idle.value())
    --impl_->flush_waiters;
}

ContinuationDiskStore::SaveResult ContinuationDiskStore::Save(
    const TextModelRunner& runner,
    std::span<const TextRunnerToken> checkpoint_tokens,
    const TextRunnerSnapshot& snapshot,
    std::span<const std::uint8_t> input_identity) {
  return impl_->Save(runner, checkpoint_tokens, snapshot, input_identity);
}

ContinuationDiskStore::RestoreResult
ContinuationDiskStore::RestoreLongestPrefix(
    const TextModelRunner& runner, TextRunnerState& state,
    std::span<const TextRunnerToken> prompt,
    std::span<const std::uint8_t> input_identity,
    std::size_t stable_prefix_tokens,
    std::span<const ContinuationInputPrefix> input_prefixes) {
  if (stable_prefix_tokens > prompt.size())
    throw std::invalid_argument("stable cache prefix exceeds prompt length");
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit) {
    impl_->Emit(ContinuationDiskEventAction::kMiss,
                ContinuationDiskEventReason::kBusy, 0, 0, prompt.size());
    return {};
  }
  return impl_->RestoreLongestPrefix(runner, state, prompt, input_identity,
                                     stable_prefix_tokens, input_prefixes);
}

std::vector<std::size_t> ContinuationDiskStore::SharedPrefixBoundaries(
    const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
    std::size_t min_tokens, std::size_t max_boundaries,
    std::span<const std::uint8_t> input_identity,
    std::span<const ContinuationInputPrefix> input_prefixes) {
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit)
    return {};
  return impl_->SharedPrefixBoundaries(runner, prompt, min_tokens,
                                       max_boundaries, input_identity,
                                       input_prefixes);
}

bool ContinuationDiskStore::Touch(
    const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
    std::span<const std::uint8_t> input_identity) {
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit)
    return false;
  return impl_->Touch(runner, tokens, input_identity);
}

bool ContinuationDiskStore::WithinCheckpointStep(
    const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
    std::span<const std::uint8_t> input_identity) {
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit)
    return false;
  return impl_->WithinCheckpointStep(runner, tokens, input_identity);
}

std::size_t ContinuationDiskStore::entry_count() const noexcept {
  return impl_->retained_entries.load(std::memory_order_relaxed);
}

std::size_t ContinuationDiskStore::retained_bytes() const noexcept {
  return impl_->retained.load(std::memory_order_relaxed);
}

std::size_t ContinuationDiskStore::capacity_bytes() const noexcept {
  return impl_->options.capacity_bytes;
}

std::size_t ContinuationDiskStore::staging_capacity_bytes() const noexcept {
  return impl_->options.staging_capacity_bytes;
}

const ContinuationDiskStoreOptions& ContinuationDiskStore::options() const noexcept {
  return impl_->options;
}

}  // namespace gufo::server
