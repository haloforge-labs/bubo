#include "src/cli/serve/continuation_disk_store.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::ContinuationDiskEvent;
using gufo::server::ContinuationDiskEventAction;
using gufo::server::ContinuationDiskEventReason;
using gufo::server::ContinuationDiskStore;
using gufo::server::ContinuationDiskStoreOptions;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPersistenceDescriptor;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;

constexpr std::size_t kFakePayloadBytes = 16;
constexpr std::size_t kDiskHeaderBytes = 96;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "gufo-continuation-disk-XXXXXX")
                              .string();
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create temporary directory");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

template<typename Integer>
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     Integer value) {
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename Integer>
Integer GetLittleEndian(std::span<const std::uint8_t> source,
                        std::size_t offset) {
  Integer result = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    result |= static_cast<Integer>(source[offset + byte]) << (byte * 8U);
  }
  return result;
}

class FakeState final : public TextRunnerState {
public:
  void Invalidate() noexcept override {
    value = 0;
    position = 0;
  }

  std::uint64_t value{0};
  std::uint64_t position{0};
};

class FakeSnapshot final : public TextRunnerSnapshot {
public:
  FakeSnapshot(std::uint64_t snapshot_value, std::uint64_t snapshot_position)
      : value(snapshot_value), position(snapshot_position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return kFakePayloadBytes;
  }

  std::uint64_t value;
  std::uint64_t position;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::invalid_argument("unexpected fake state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::invalid_argument("unexpected fake state");
  }
  return *fake;
}

const FakeSnapshot& RequireFakeSnapshot(const TextRunnerSnapshot& snapshot) {
  const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
  if (fake == nullptr) {
    throw std::invalid_argument("unexpected fake snapshot");
  }
  return *fake;
}

class FakeRunner final : public TextModelRunner {
public:
  explicit FakeRunner(std::string identity, std::string model_id = "fake-model")
      : identity_(identity.begin(), identity.end()),
        model_id_(std::move(model_id)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_id_,
        .state_abi = "fake-state-v1",
        .max_context = 4096,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
            },
        .persistence =
            TextRunnerPersistenceDescriptor{
                .compatibility_identity = identity_,
                .payload_version = 7,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {};
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{
        .kind = TextExecutionPlanKind::kSerial,
        .physical_width = 1,
    }};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(text.size());
    for (const char value : text) {
      tokens.push_back(static_cast<unsigned char>(value));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken>) const override {
    return {};
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<FakeState>();
  }

  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState&,
                                        std::span<const TextRunnerToken>,
                                        std::size_t,
                                        std::size_t) const override {
    throw std::logic_error("fake persistence test does not prefill");
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState&, gufo::sampling::SamplerState&) const override {
    throw std::logic_error("fake persistence test does not decode");
  }

  void Advance(TextRunnerState&, TextRunnerToken) const override {
    throw std::logic_error("fake persistence test does not advance");
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState&) const override {
    return kFakePayloadBytes;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& fake = RequireFakeState(state);
    return std::make_unique<FakeSnapshot>(fake.value, fake.position);
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto& saved = RequireFakeSnapshot(snapshot);
    auto& destination = RequireFakeState(state);
    destination.value = saved.value;
    destination.position = saved.position;
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    (void)RequireFakeSnapshot(snapshot);
    return kFakePayloadBytes;
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& saved = RequireFakeSnapshot(snapshot);
    if (destination.size() != kFakePayloadBytes) {
      throw std::invalid_argument("fake destination size mismatch");
    }
    PutLittleEndian<std::uint64_t>(destination, 0, saved.value);
    PutLittleEndian<std::uint64_t>(destination, 8, saved.position);
    return destination.size();
  }

  std::function<void()> before_stream;
  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    if (before_stream)
      before_stream();
    std::array<std::uint8_t, kFakePayloadBytes> bytes{};
    (void)SerializePersistentSnapshot(snapshot, bytes);
    sink(std::span(bytes).first(7));
    sink(std::span(bytes).subspan(7));
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != kFakePayloadBytes) {
      throw std::invalid_argument("fake payload size mismatch");
    }
    auto& destination = RequireFakeState(state);
    destination.value = GetLittleEndian<std::uint64_t>(payload, 0);
    destination.position = GetLittleEndian<std::uint64_t>(payload, 8);
  }

private:
  std::vector<std::uint8_t> identity_;
  std::string model_id_;
};

std::unique_ptr<TextRunnerSnapshot> MakeSnapshot(const FakeRunner& runner,
                                                 std::uint64_t value,
                                                 std::uint64_t position) {
  auto state = runner.CreateState();
  auto& fake = RequireFakeState(*state);
  fake.value = value;
  fake.position = position;
  return runner.Snapshot(*state);
}

ContinuationDiskStoreOptions StoreOptions(
    const std::filesystem::path& directory,
    std::size_t capacity_bytes = 1024U * 1024U,
    std::size_t staging_bytes = 1024U * 1024U) {
  return {
      .directory = directory,
      .capacity_bytes = capacity_bytes,
      .staging_capacity_bytes = staging_bytes,
  };
}

std::vector<std::filesystem::path> CacheFiles(
    const std::filesystem::path& directory) {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".kvc") {
      files.push_back(entry.path());
    }
  }
  return files;
}

std::size_t ExpectedFileBytes(std::size_t identity_bytes,
                              std::size_t token_count) {
  return kDiskHeaderBytes + identity_bytes +
         token_count * sizeof(std::uint32_t) + kFakePayloadBytes;
}

ContinuationDiskStore::SaveResult SaveTokens(
    ContinuationDiskStore& store, const FakeRunner& runner,
    std::initializer_list<TextRunnerToken> tokens,
    const TextRunnerSnapshot& snapshot) {
  const std::vector<TextRunnerToken> owned(tokens);
  return store.Save(runner, owned, snapshot);
}

ContinuationDiskStore::RestoreResult RestoreTokens(
    ContinuationDiskStore& store, const FakeRunner& runner,
    TextRunnerState& state, std::initializer_list<TextRunnerToken> tokens) {
  const std::vector<TextRunnerToken> owned(tokens);
  return store.RestoreLongestPrefix(runner, state, owned);
}

void TestSha256KnownVector() {
  const std::array<std::uint8_t, 3> input = {'a', 'b', 'c'};
  Expect(gufo::crypto::Sha256Hex(input) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "shared SHA-256 implementation matches the standard vector");
  gufo::crypto::Sha256Hasher incremental;
  incremental.Update(std::span(input).first(1));
  gufo::crypto::Sha256Hasher first;
  first.Update(std::span(input).first(1));
  Expect(incremental.Digest() == first.Finish(),
         "prefix digest matches an independently finished hash");
  incremental.Update(std::span(input).subspan(1));
  const auto complete = incremental.Digest();
  Expect(complete == incremental.Finish(),
         "reading a prefix digest does not consume incremental state");

  TemporaryDirectory directory;
  const auto path = directory.path() / "artifact.gguf";
  {
    std::ofstream output(path, std::ios::binary);
    output.write("abc", 3);
  }
  Expect(gufo::crypto::Sha256FileHex(path) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "streamed artifact SHA-256 matches the in-memory implementation");
}

void TestRestartRestoreAndCompatibilityIdentity() {
  TemporaryDirectory directory;
  const FakeRunner writer("artifact-A", "fake-old-executable");
  auto snapshot = MakeSnapshot(writer, 0x123456789ABCDEF0ULL, 3);
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    const auto saved = SaveTokens(store, writer, {1, 2, 3}, *snapshot);
    Expect(saved.stored && saved.payload_bytes == kFakePayloadBytes,
           "first process atomically stores a persistent snapshot");
    Expect(
        store.entry_count() == 1 && store.retained_bytes() == saved.file_bytes,
        "store accounts exact retained file bytes");
  }

  const FakeRunner compatible_reader("artifact-A", "fake-new-executable");
  auto compatible_state = compatible_reader.CreateState();
  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  const auto restored = RestoreTokens(restarted, compatible_reader,
                                      *compatible_state, {1, 2, 3, 4, 5});
  Expect(restored.restored && restored.token_count == 3,
         "new executable restores the longest compatible saved prefix");
  Expect(RequireFakeState(*compatible_state).value == 0x123456789ABCDEF0ULL &&
             RequireFakeState(*compatible_state).position == 3,
         "restart restore reconstructs exact model-owned state");

  const FakeRunner incompatible_reader("artifact-B");
  auto incompatible_state = incompatible_reader.CreateState();
  Expect(!RestoreTokens(restarted, incompatible_reader, *incompatible_state,
                        {1, 2, 3, 4})
              .restored,
         "different compatibility identity is a deterministic cache miss");
}

void TestLongestPrefixAndForcedHashCollision() {
  TemporaryDirectory directory;
  const FakeRunner runner("collision-model");
  const auto forced_collision = [](std::span<const std::uint8_t>) {
    return std::string(64, '0');
  };
  ContinuationDiskStore store(StoreOptions(directory.path()), {},
                              forced_collision);

  auto short_snapshot = MakeSnapshot(runner, 11, 2);
  auto long_snapshot = MakeSnapshot(runner, 22, 4);
  auto unrelated_snapshot = MakeSnapshot(runner, 33, 3);
  Expect(SaveTokens(store, runner, {1, 2}, *short_snapshot).stored,
         "collision bucket stores first key");
  Expect(SaveTokens(store, runner, {1, 2, 3, 4}, *long_snapshot).stored,
         "collision bucket stores longer exact key");
  Expect(SaveTokens(store, runner, {9, 8, 7}, *unrelated_snapshot).stored,
         "collision bucket stores unrelated exact key");

  auto state = runner.CreateState();
  const auto restored = RestoreTokens(store, runner, *state, {1, 2, 3, 4, 5});
  Expect(restored.restored && restored.token_count == 4 &&
             RequireFakeState(*state).value == 22,
         "full token verification selects longest exact key after collision");

  auto unrelated_state = runner.CreateState();
  Expect(
      RestoreTokens(store, runner, *unrelated_state, {9, 8, 7, 6}).restored &&
          RequireFakeState(*unrelated_state).value == 33,
      "forced collision cannot return another prompt's state");

  auto miss_state = runner.CreateState();
  Expect(!RestoreTokens(store, runner, *miss_state, {5, 5, 5}).restored,
         "forced collision without exact tokens remains a miss");
}

void TestCorruptionBecomesDeterministicMissAndRemoval() {
  TemporaryDirectory directory;
  const FakeRunner runner("corruption-model");
  std::vector<ContinuationDiskEvent> events;
  ContinuationDiskStore store(
      StoreOptions(directory.path()),
      [&](const ContinuationDiskEvent& event) { events.push_back(event); });
  auto snapshot = MakeSnapshot(runner, 44, 3);
  Expect(SaveTokens(store, runner, {4, 4, 4}, *snapshot).stored,
         "corruption test stores an entry");
  const auto files = CacheFiles(directory.path());
  Expect(files.size() == 1, "one published cache file exists");

  std::fstream stream(files.front(),
                      std::ios::binary | std::ios::in | std::ios::out);
  stream.seekg(-1, std::ios::end);
  char value = 0;
  stream.read(&value, 1);
  value ^= static_cast<char>(0x5A);
  stream.seekp(-1, std::ios::end);
  stream.write(&value, 1);
  stream.close();

  if (::geteuid() != 0) {
    Expect(::chmod(directory.path().c_str(), S_IRUSR | S_IXUSR) == 0,
           "cache directory can simulate an unlink failure");
    auto lookup = std::async(std::launch::async, [&] {
      auto state = runner.CreateState();
      return RestoreTokens(store, runner, *state, {4, 4, 4, 5});
    });
    const bool returned =
        lookup.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    Expect(::chmod(directory.path().c_str(), S_IRWXU) == 0,
           "cache directory permissions restored");
    const auto result = lookup.get();
    Expect(returned && !result.restored && store.entry_count() == 1,
           "unremovable corrupt entries miss without an infinite retry");
  }

  auto state = runner.CreateState();
  Expect(!RestoreTokens(store, runner, *state, {4, 4, 4, 5}).restored,
         "checksum-invalid entry becomes a cache miss");
  Expect(store.entry_count() == 0 && CacheFiles(directory.path()).empty(),
         "checksum-invalid entry is removed from index and disk");
  Expect(std::ranges::any_of(
             events,
             [](const ContinuationDiskEvent& event) {
               return event.reason ==
                      ContinuationDiskEventReason::kChecksumMismatch;
             }),
         "sanitized event reports checksum removal reason");
}

void TestByteAndStagingLimits() {
  TemporaryDirectory staging_directory;
  const FakeRunner runner("limits");
  const std::size_t file_bytes = ExpectedFileBytes(6, 2);
  auto snapshot = MakeSnapshot(runner, 1, 2);

  ContinuationDiskStore staging_limited(
      StoreOptions(staging_directory.path(), 4096, file_bytes - 1));
  Expect(!SaveTokens(staging_limited, runner, {1, 2}, *snapshot).stored &&
             staging_limited.entry_count() == 0,
         "full file image must fit the bounded staging allocation");

  TemporaryDirectory byte_directory;
  ContinuationDiskStore byte_limited(
      StoreOptions(byte_directory.path(), file_bytes - 1, 4096));
  Expect(!SaveTokens(byte_limited, runner, {1, 2}, *snapshot).stored &&
             byte_limited.entry_count() == 0,
         "single entry larger than total disk budget is skipped");
}

void TestAutomaticStagingAndAdmissionDiagnostics() {
  TemporaryDirectory default_directory;
  ContinuationDiskStore defaults({.directory = default_directory.path()});
  Expect(defaults.capacity_bytes() == std::size_t{8} * 1024U * 1024U * 1024U &&
             defaults.staging_capacity_bytes() > kDiskHeaderBytes &&
             defaults.staging_capacity_bytes() <= defaults.capacity_bytes(),
         "defaults bound RAM staging by available RAM and disk retention");

  TemporaryDirectory directory;
  std::vector<ContinuationDiskEvent> events;
  const FakeRunner runner("automatic");
  ContinuationDiskStore store(
      {.directory = directory.path(), .capacity_bytes = 1024U * 1024U},
      [&](const auto& event) { events.push_back(event); });
  const auto limit = store.staging_capacity_bytes();
  Expect(limit > kDiskHeaderBytes && limit <= store.capacity_bytes(),
         "automatic staging resolves to a usable bounded RAM budget");
  const auto payload = limit / 2;
  auto first = store.ReserveCapture(runner, 2, payload);
  Expect(first != nullptr, "automatic staging admits a snapshot");
  Expect(!store.CanSave(runner, 2, payload) &&
             !store.ReserveCapture(runner, 2, payload),
         "automatic staging still accounts for all in-flight captures");
  Expect(!events.empty() &&
             events.back().reason ==
                 ContinuationDiskEventReason::kStagingCapacity &&
             events.back().staging_capacity_bytes == limit &&
             events.back().staging_used_bytes > payload &&
             events.back().payload_bytes == payload,
         "preflight rejection reports staging usage and required size");
  first.reset();
  Expect(store.CanSave(runner, 2, payload),
         "released staging is reusable without allocating the whole budget");

  Expect(!store.CanSave(runner, 2, store.capacity_bytes()) &&
             events.back().reason == ContinuationDiskEventReason::kByteCapacity,
         "disk retention rejection is distinguished from staging pressure");

  TemporaryDirectory explicit_directory;
  ContinuationDiskStore explicit_limit(
      StoreOptions(explicit_directory.path(), 4096, 256),
      [&](const auto& event) { events.push_back(event); });
  Expect(explicit_limit.staging_capacity_bytes() == 256 &&
             !explicit_limit.CanSave(runner, 2, 512) &&
             events.back().reason ==
                 ContinuationDiskEventReason::kStagingCapacity &&
             events.back().staging_used_bytes == 0,
         "an explicit too-small staging limit remains enforced and visible");
}

void TestSmallerStagingPreservesExistingFiles() {
  TemporaryDirectory directory;
  const FakeRunner runner("staging-restart");
  const auto file_bytes = ExpectedFileBytes(15, 2);
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(
        SaveTokens(store, runner, {1, 2}, *MakeSnapshot(runner, 42, 2)).stored,
        "restart fixture stores");
  }
  {
    std::vector<ContinuationDiskEvent> events;
    ContinuationDiskStore store(
        StoreOptions(directory.path(), 4096, file_bytes - 1),
        [&](const auto& event) { events.push_back(event); });
    Expect(store.entry_count() == 1 && store.retained_bytes() == file_bytes &&
               CacheFiles(directory.path()).size() == 1,
           "a smaller RAM budget preserves and accounts for existing files");
    auto state = runner.CreateState();
    Expect(!RestoreTokens(store, runner, *state, {1, 2, 3}).restored,
           "unverified oversized files are not exposed to lookup");
    Expect(std::ranges::any_of(
               events,
               [&](const auto& event) {
                 return event.action == ContinuationDiskEventAction::kSkipped &&
                        event.reason ==
                            ContinuationDiskEventReason::kStagingCapacity &&
                        event.file_bytes == file_bytes;
               }),
           "oversized startup files report a staging skip, not corruption");
  }
  {
    ContinuationDiskStore store(
        {.directory = directory.path(), .capacity_bytes = 4096});
    auto state = runner.CreateState();
    Expect(RestoreTokens(store, runner, *state, {1, 2, 3}).restored &&
               RequireFakeState(*state).value == 42,
           "automatic staging validates and restores the preserved file");
  }
  {
    ContinuationDiskStore store(
        StoreOptions(directory.path(), file_bytes - 1, file_bytes - 1));
    Expect(store.retained_bytes() == 0 && CacheFiles(directory.path()).empty(),
           "unindexed files remain subject to the disk retention budget");
  }
}

void TestLruEvictionUsesActualFileBytes() {
  TemporaryDirectory directory;
  const FakeRunner runner("12345678");
  const std::size_t entry_bytes = ExpectedFileBytes(8, 2);
  ContinuationDiskStore store(
      StoreOptions(directory.path(), entry_bytes * 2, 4096));

  auto first = MakeSnapshot(runner, 1, 2);
  auto second = MakeSnapshot(runner, 2, 2);
  auto third = MakeSnapshot(runner, 3, 2);
  Expect(SaveTokens(store, runner, {1, 1}, *first).file_bytes == entry_bytes,
         "store reports exact first file size");
  Expect(SaveTokens(store, runner, {2, 2}, *second).stored,
         "second entry fills exact two-entry budget");

  auto touched = runner.CreateState();
  Expect(RestoreTokens(store, runner, *touched, {1, 1, 9}).restored,
         "restore refreshes first entry's LRU position");
  Expect(SaveTokens(store, runner, {3, 3}, *third).stored,
         "third entry evicts one least-recently-used file");
  Expect(store.entry_count() == 2 && store.retained_bytes() == entry_bytes * 2,
         "retained bytes never exceed configured total budget");

  auto first_state = runner.CreateState();
  auto second_state = runner.CreateState();
  auto third_state = runner.CreateState();
  Expect(RestoreTokens(store, runner, *first_state, {1, 1, 8}).restored,
         "recently touched entry survives LRU pressure");
  Expect(!RestoreTokens(store, runner, *second_state, {2, 2, 8}).restored,
         "oldest entry is the deterministic LRU victim");
  Expect(RestoreTokens(store, runner, *third_state, {3, 3, 8}).restored,
         "new entry survives LRU pressure");
}

void TestAtomicPublicationAndPrivatePermissions() {
  TemporaryDirectory directory;
  const auto orphan = directory.path() / ".tmp-orphan";
  {
    std::ofstream output(orphan, std::ios::binary);
    output << "partial";
  }

  const FakeRunner runner("permissions");
  ContinuationDiskStore store(StoreOptions(directory.path()));
  Expect(!std::filesystem::exists(orphan),
         "startup removes an interrupted private temporary file");
  auto snapshot = MakeSnapshot(runner, 77, 2);
  Expect(SaveTokens(store, runner, {7, 7}, *snapshot).stored,
         "permission test stores one entry");

  const auto files = CacheFiles(directory.path());
  Expect(files.size() == 1, "atomic publication exposes one final file");
  for (const auto& entry :
       std::filesystem::directory_iterator(directory.path())) {
    Expect(!entry.path().filename().string().starts_with(".tmp-"),
           "successful publication leaves no temporary file");
  }
  struct stat status{};
  Expect(::lstat(files.front().c_str(), &status) == 0 &&
             S_ISREG(status.st_mode) &&
             (status.st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
             (status.st_mode & S_IRUSR) != 0 && (status.st_mode & S_IWUSR) != 0,
         "published cache file is private and owner read-write");
}

void TestStartupRejectsUnsafeAndInvalidFiles() {
  TemporaryDirectory directory;
  const FakeRunner runner("startup-valid");
  const FakeRunner incompatible("startup-incompatible");
  std::filesystem::path truncated_file;
  std::filesystem::path checksum_file;
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    const auto save_and_find_file =
        [&](const FakeRunner& entry_runner,
            std::initializer_list<TextRunnerToken> tokens,
            const TextRunnerSnapshot& snapshot) {
          const auto before = CacheFiles(directory.path());
          Expect(SaveTokens(store, entry_runner, tokens, snapshot).stored,
                 "startup fixture publishes one immutable entry");
          const auto after = CacheFiles(directory.path());
          const auto added = std::find_if(
              after.begin(), after.end(), [&](const auto& candidate) {
                return std::find(before.begin(), before.end(), candidate) ==
                       before.end();
              });
          Expect(added != after.end(),
                 "startup fixture identifies the newly published file");
          return *added;
        };
    auto first = MakeSnapshot(runner, 1, 2);
    auto second = MakeSnapshot(runner, 2, 2);
    auto third = MakeSnapshot(runner, 3, 2);
    auto other = MakeSnapshot(incompatible, 4, 2);
    truncated_file = save_and_find_file(runner, {1, 1}, *first);
    checksum_file = save_and_find_file(runner, {2, 2}, *second);
    (void)save_and_find_file(runner, {3, 3}, *third);
    (void)save_and_find_file(incompatible, {4, 4}, *other);
  }

  auto files = CacheFiles(directory.path());
  Expect(files.size() == 4, "startup fixture exposes four immutable files");
  std::filesystem::resize_file(truncated_file, 12);
  {
    std::fstream stream(checksum_file,
                        std::ios::binary | std::ios::in | std::ios::out);
    stream.seekg(-1, std::ios::end);
    char value = 0;
    stream.read(&value, 1);
    value ^= static_cast<char>(0x5A);
    stream.seekp(-1, std::ios::end);
    stream.write(&value, 1);
  }
  const auto target = directory.path() / "external-target";
  {
    std::ofstream output(target, std::ios::binary);
    output << "must survive";
  }
  const auto unsafe_link = directory.path() / "unsafe.kvc";
  // Windows needs Developer Mode or elevation to create symlinks; without
  // one the symlink-specific expectations below are vacuous.
  std::error_code link_error;
  std::filesystem::create_symlink(target.filename(), unsafe_link, link_error);
  const bool have_link = !link_error;

  std::vector<ContinuationDiskEvent> events;
  ContinuationDiskStore restarted(
      StoreOptions(directory.path()),
      [&](const ContinuationDiskEvent& event) { events.push_back(event); });
  Expect(
      restarted.entry_count() == 2 && CacheFiles(directory.path()).size() == 2,
      "startup indexes only checksum-valid regular entries");
  Expect(std::filesystem::exists(target) &&
             (!have_link || !std::filesystem::exists(unsafe_link)),
         "unsafe symlink is removed without following its target");
  Expect(
      std::ranges::any_of(events,
                          [](const ContinuationDiskEvent& event) {
                            return event.reason ==
                                   ContinuationDiskEventReason::kCorrupt;
                          }) &&
          std::ranges::any_of(
              events,
              [](const ContinuationDiskEvent& event) {
                return event.reason ==
                       ContinuationDiskEventReason::kChecksumMismatch;
              }) &&
          (!have_link || std::ranges::any_of(
                             events,
                             [](const ContinuationDiskEvent& event) {
                               return event.reason ==
                                      ContinuationDiskEventReason::kUnsafeFile;
                             })),
      "startup reports sanitized invalid-file reasons");

  std::size_t compatible_hits = 0;
  for (const std::array<TextRunnerToken, 3> prompt :
       {std::array<TextRunnerToken, 3>{1, 1, 9},
        std::array<TextRunnerToken, 3>{2, 2, 9},
        std::array<TextRunnerToken, 3>{3, 3, 9}}) {
    auto state = runner.CreateState();
    compatible_hits +=
        restarted.RestoreLongestPrefix(runner, *state, prompt).restored ? 1 : 0;
  }
  Expect(compatible_hits == 1,
         "the remaining compatible startup entry restores exactly");
  auto incompatible_state = incompatible.CreateState();
  Expect(
      RestoreTokens(restarted, incompatible, *incompatible_state, {4, 4, 9})
          .restored,
      "a valid different compatibility identity remains independently usable");
}

void TestRootSymlinkIsRejected() {
  TemporaryDirectory directory;
  const auto target = directory.path() / "target";
  const auto link = directory.path() / "cache-link";
  std::filesystem::create_directory(target);
  std::error_code link_error;
  std::filesystem::create_directory_symlink(target.filename(), link,
                                            link_error);
  if (link_error) {
    std::cout << "skipping root symlink check: " << link_error.message()
              << '\n';
    return;
  }
  bool rejected = false;
  try {
    ContinuationDiskStore store(StoreOptions(link));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected, "configured cache-directory symlinks are rejected");
}

void TestConcurrentCallersRemainBoundedAndExact() {
  TemporaryDirectory directory;
  const FakeRunner runner("concurrent");
  constexpr std::size_t kThreadCount = 8;
  ContinuationDiskStore store(
      StoreOptions(directory.path(), 1024U * 1024U, 4096));
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (std::size_t index = 0; index < kThreadCount; ++index) {
    workers.emplace_back([&, index] {
      auto snapshot = MakeSnapshot(runner, 100 + index, 2);
      const std::array<TextRunnerToken, 2> tokens = {
          static_cast<TextRunnerToken>(index),
          static_cast<TextRunnerToken>(index + 100)};
      Expect(store.Save(runner, tokens, *snapshot).stored,
             "concurrent save succeeds under serialized I/O bound");
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  Expect(store.entry_count() == kThreadCount,
         "all concurrent saves publish distinct exact entries");
  for (std::size_t index = 0; index < kThreadCount; ++index) {
    auto state = runner.CreateState();
    const std::array<TextRunnerToken, 3> prompt = {
        static_cast<TextRunnerToken>(index),
        static_cast<TextRunnerToken>(index + 100), 999};
    Expect(store.RestoreLongestPrefix(runner, *state, prompt).restored &&
               RequireFakeState(*state).value == 100 + index,
           "concurrent store preserves exact per-key payloads");
  }
}

void TestConcurrentStoreInstancesPublishSafely() {
  TemporaryDirectory directory;
  const FakeRunner runner("multi-process");
  {
    ContinuationDiskStore first(StoreOptions(directory.path()));
    ContinuationDiskStore second(StoreOptions(directory.path()));
    auto first_snapshot = MakeSnapshot(runner, 101, 2);
    auto second_snapshot = MakeSnapshot(runner, 202, 2);
    std::thread first_writer([&] {
      Expect(SaveTokens(first, runner, {1, 0}, *first_snapshot).stored,
             "first store instance publishes atomically");
    });
    std::thread second_writer([&] {
      Expect(SaveTokens(second, runner, {2, 0}, *second_snapshot).stored,
             "second store instance publishes atomically");
    });
    first_writer.join();
    second_writer.join();
  }

  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  Expect(restarted.entry_count() == 2,
         "restart indexes files from concurrent store instances");
  auto first_state = runner.CreateState();
  auto second_state = runner.CreateState();
  Expect(
      RestoreTokens(restarted, runner, *first_state, {1, 0, 9}).restored &&
          RequireFakeState(*first_state).value == 101 &&
          RestoreTokens(restarted, runner, *second_state, {2, 0, 9}).restored &&
          RequireFakeState(*second_state).value == 202,
      "concurrent publishers expose no partial or confused entry");
}

void TestStartupPreservesActivePublisher() {
  TemporaryDirectory directory;
  auto runner = std::make_shared<FakeRunner>("startup-publisher");
  std::binary_semaphore entered(0), release(0);
  const auto orphan = directory.path() / ".tmp-orphan";
  {
    ContinuationDiskStore writer(StoreOptions(directory.path()));
    runner->before_stream = [&] {
      entered.release();
      release.acquire();
    };
    Expect(writer.SaveAsync(runner, {1, 2}, MakeSnapshot(*runner, 42, 2)) != 0,
           "writer accepts a checkpoint before another store starts");
    const bool started = entered.try_acquire_for(std::chrono::seconds(2));
    if (!started)
      release.release();
    Expect(started, "writer reaches the temporary-file serialization barrier");
    {
      std::ofstream output(orphan, std::ios::binary);
      output << "interrupted older write";
    }
    auto startup = std::async(std::launch::async, [&] {
      ContinuationDiskStore reader(StoreOptions(directory.path()));
    });
    const bool ready =
        startup.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    release.release();
    startup.get();
    writer.Flush();
    Expect(ready, "startup does not wait for an active checkpoint writer");
    Expect(
        writer.entry_count() == 1 && CacheFiles(directory.path()).size() == 1,
        "overlapping startup does not discard the active writer's checkpoint");
  }
  runner->before_stream = {};

  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  auto state = runner->CreateState();
  Expect(RestoreTokens(restarted, *runner, *state, {1, 2, 3}).restored &&
             RequireFakeState(*state).value == 42,
         "checkpoint published across startup restores its exact payload");
  Expect(!std::filesystem::exists(orphan),
         "startup without a publisher removes abandoned temporary files");
}

void TestSharedPrefixBoundariesAndExactDedup() {
  TemporaryDirectory directory;
  const FakeRunner runner("shared-prefix-model");
  ContinuationDiskStore store(StoreOptions(directory.path()), {});

  // Conversation A: shared system prefix {7, 7, 7} plus its own turn.
  auto first = MakeSnapshot(runner, 11, 5);
  Expect(SaveTokens(store, runner, {7, 7, 7, 1, 2}, *first).stored,
         "first conversation frontier is stored");

  // Conversation B shares the first three tokens only.
  const std::vector<TextRunnerToken> second_prompt = {7, 7, 7, 3, 4, 5};
  Expect(store.SharedPrefixBoundaries(runner, second_prompt, 1, 4) ==
             std::vector<std::size_t>{3},
         "common prefix with a stored entry becomes a boundary");
  Expect(store.SharedPrefixBoundaries(runner, second_prompt, 4, 4).empty(),
         "boundaries below the minimum length are dropped");
  const std::vector<TextRunnerToken> first_prompt = {7, 7, 7, 1, 2};
  Expect(store.SharedPrefixBoundaries(runner, first_prompt, 1, 4).empty(),
         "an exact stored prompt yields no boundary");

  // Persist the shared prefix; a third conversation restores it.
  auto shared = MakeSnapshot(runner, 77, 3);
  Expect(SaveTokens(store, runner, {7, 7, 7}, *shared).stored,
         "shared prefix snapshot is stored");
  Expect(store.SharedPrefixBoundaries(runner, second_prompt, 1, 4).empty(),
         "a stored shared prefix is no longer a boundary");
  auto state = runner.CreateState();
  const auto restored = RestoreTokens(store, runner, *state, {7, 7, 7, 9});
  Expect(restored.restored && restored.token_count == 3 &&
             RequireFakeState(*state).value == 77,
         "new conversation restores the shared prefix");

  // Saving the same tokens again touches instead of rewriting.
  const std::size_t retained_before = store.retained_bytes();
  const std::size_t entries_before = store.entry_count();
  auto duplicate = MakeSnapshot(runner, 78, 3);
  Expect(!SaveTokens(store, runner, {7, 7, 7}, *duplicate).stored,
         "exact duplicate save is skipped");
  Expect(store.retained_bytes() == retained_before &&
             store.entry_count() == entries_before,
         "exact duplicate save leaves the store unchanged");
  Expect(store.Touch(runner, std::vector<TextRunnerToken>{7, 7, 7}),
         "touch finds the exact entry");
  Expect(!store.Touch(runner, std::vector<TextRunnerToken>{7, 7}),
         "touch misses a non-stored prefix");

  // Longest boundaries win when more than max_boundaries are shared.
  auto deeper = MakeSnapshot(runner, 12, 5);
  Expect(SaveTokens(store, runner, {7, 7, 7, 3, 8}, *deeper).stored,
         "deeper entry is stored");
  const std::vector<TextRunnerToken> third_prompt = {7, 7, 7, 3, 4, 6};
  Expect(store.SharedPrefixBoundaries(runner, third_prompt, 1, 4) ==
             std::vector<std::size_t>{4},
         "only the unstored shared length is reported");
  Expect(store.SharedPrefixBoundaries(runner, third_prompt, 1, 1) ==
             std::vector<std::size_t>{4},
         "capping keeps the longest boundary");
}

void TestBoundedAsyncPersistenceDoesNotBlockLookup() {
  TemporaryDirectory directory;
  auto runner = std::make_shared<FakeRunner>("async-store");
  std::binary_semaphore entered(0), release(0);
  const auto bytes = ExpectedFileBytes(
      runner->Descriptor().persistence->compatibility_identity.size(), 2);
  {
    ContinuationDiskStore store(
        StoreOptions(directory.path(), 4096, bytes + sizeof(FakeSnapshot)));
    Expect(store
               .Save(*runner, std::vector<TextRunnerToken>{7, 8},
                     *MakeSnapshot(*runner, 99, 2))
               .stored,
           "an existing entry is available before the new write");
    runner->before_stream = [&] {
      entered.release();
      release.acquire();
    };
    auto snapshot = MakeSnapshot(*runner, 42, 2);
    Expect(store.CanSave(*runner, 2, snapshot->PayloadBytes()),
           "empty persistence queue admits the snapshot before capture");
    auto capture = store.ReserveCapture(*runner, 2, snapshot->PayloadBytes());
    Expect(capture != nullptr &&
               !store.CanSave(*runner, 2, snapshot->PayloadBytes()) &&
               !store.ReserveCapture(*runner, 2, snapshot->PayloadBytes()),
           "in-flight captures reserve staging before any allocation");
    capture.reset();
    Expect(store.CanSave(*runner, 2, snapshot->PayloadBytes()),
           "discarded captures release their reservation");
    capture = store.ReserveCapture(*runner, 2, snapshot->PayloadBytes());
    Expect(store.SaveAsync(runner, {1, 2}, std::move(snapshot), {},
                           std::move(capture)) == bytes,
           "immutable snapshot is queued without waiting for serialization");
    const bool started = entered.try_acquire_for(std::chrono::seconds(2));
    Expect(started, "persistence worker starts independently");
    Expect(
        !store.CanSave(*runner, 2, sizeof(FakeSnapshot)) &&
            store.SaveAsync(runner, {3, 4}, MakeSnapshot(*runner, 43, 2)) == 0,
        "active writes count against the bounded queue");
    auto lookup = std::async(std::launch::async, [&] {
      auto state = runner->CreateState();
      return RestoreTokens(store, *runner, *state, {7, 8, 9}).restored &&
             RequireFakeState(*state).value == 99 &&
             store.Touch(*runner, std::vector<TextRunnerToken>{7, 8}) &&
             !RestoreTokens(store, *runner, *state, {1, 2, 9}).restored &&
             store
                 .SharedPrefixBoundaries(
                     *runner, std::vector<TextRunnerToken>{1, 9}, 1, 4)
                 .empty() &&
             !store.Touch(*runner, std::vector<TextRunnerToken>{1, 2});
    });
    const bool ready =
        lookup.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    release.release();
    Expect(ready && lookup.get(),
           "existing entries remain readable while a writer is blocked");
    store.Flush();
    runner->before_stream = {};
    auto state = runner->CreateState();
    Expect(RestoreTokens(store, *runner, *state, {1, 2, 9}).restored &&
               RequireFakeState(*state).value == 42,
           "streamed chunks form a checksummed restorable snapshot");
    Expect(store.SaveAsync(runner, {3, 4}, MakeSnapshot(*runner, 43, 2)) != 0,
           "completed writes release queue admission");
  }
  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  Expect(restarted.entry_count() == 3,
         "shutdown drains accepted persistence jobs");
}

bool DiskDeferExpected() {
  const char* value = std::getenv("GUFO_DISK_DEFER");
  return value && std::string_view(value) == "1";
}

void TestDeferredPersistenceIdleAndBytes() {
  TemporaryDirectory directory, reference_directory;
  auto runner = std::make_shared<FakeRunner>("defer-idle");
  // Synchronous Save is the unchanged immediate-serialization byte reference.
  {
    ContinuationDiskStore reference(StoreOptions(reference_directory.path()));
    Expect(SaveTokens(reference, *runner, {1, 2}, *MakeSnapshot(*runner, 42, 2))
               .stored,
           "immediate reference is stored");
  }
  std::binary_semaphore entered(0);
  runner->before_stream = [&] { entered.release(); };
  ContinuationDiskStore store(StoreOptions(directory.path()));
  {
    auto active = store.BeginGeneration();
    {
      auto second_active = store.BeginGeneration();
      Expect(store.SaveAsync(runner, {1, 2}, MakeSnapshot(*runner, 42, 2)) != 0,
             "busy generation admits persistence");
      if (DiskDeferExpected())
        Expect(!entered.try_acquire_for(std::chrono::milliseconds(100)),
               "serialization cannot begin during active decode");
      else
        Expect(entered.try_acquire_for(std::chrono::seconds(2)),
               "unset/0 retain immediate serialization despite active decode");
    }
    if (DiskDeferExpected())
      Expect(!entered.try_acquire_for(std::chrono::milliseconds(100)),
             "one remaining generation still gates serialization");
  }
  if (DiskDeferExpected())
    Expect(entered.try_acquire_for(std::chrono::seconds(2)),
           "serialization begins when the last generation becomes idle");
  store.Flush();
  const auto files = CacheFiles(directory.path());
  const auto reference_files = CacheFiles(reference_directory.path());
  Expect(files.size() == 1 && reference_files.size() == 1,
         "both modes persist exactly one file");
  const auto read = [](const auto& path) {
    std::ifstream input(path, std::ios::binary);
    Expect(input.good(), "persisted file is readable");
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
  };
  Expect(
      read(files.front()) == read(reference_files.front()),
      "deferred/immediate complete persisted bytes equal the same reference");
}

void TestDeferredPersistenceDeadline() {
  TemporaryDirectory directory;
  auto runner = std::make_shared<FakeRunner>("defer-deadline");
  std::binary_semaphore entered(0);
  runner->before_stream = [&] { entered.release(); };
  auto options = StoreOptions(directory.path());
  options.defer_max_wait = std::chrono::milliseconds(250);
  ContinuationDiskStore store(options);
  auto active = store.BeginGeneration();
  const auto start = std::chrono::steady_clock::now();
  Expect(store.SaveAsync(runner, {1, 2}, MakeSnapshot(*runner, 42, 2)) != 0,
         "bounded busy write is admitted");
  if (DiskDeferExpected()) {
    Expect(!entered.try_acquire_for(std::chrono::milliseconds(50)),
           "deadline does not bypass the initial busy gate");
  }
  Expect(entered.try_acquire_for(std::chrono::seconds(2)),
         "deadline starts serialization even while generation stays active");
  if (DiskDeferExpected())
    Expect(std::chrono::steady_clock::now() - start >= options.defer_max_wait,
           "deadline is measured from admission, without early expiry");
  store.Flush();
  Expect(store.entry_count() == 1, "deadline does not discard persistence");
}

void TestDeferredPersistencePressure() {
  for (const bool capture_pressure : {false, true}) {
    TemporaryDirectory directory;
    auto runner = std::make_shared<FakeRunner>("defer-pressure");
    std::binary_semaphore entered(0), release(0);
    bool first = true;
    runner->before_stream = [&] {
      if (std::exchange(first, false)) {
        entered.release();
        release.acquire();
      }
    };
    const auto bytes = ExpectedFileBytes(
        runner->Descriptor().persistence->compatibility_identity.size(), 2);
    ContinuationDiskStore store(
        StoreOptions(directory.path(), 4096, 2 * bytes));
    auto active = store.BeginGeneration();
    Expect(
        store.SaveAsync(runner, {1, 2}, MakeSnapshot(*runner, 42, 2)) == bytes,
        "first pressure payload is queued");
    if (DiskDeferExpected())
      Expect(!entered.try_acquire_for(std::chrono::milliseconds(50)),
             "partial staging still defers busy materialization");
    std::unique_ptr<ContinuationDiskStore::CaptureReservation> reservation;
    if (capture_pressure) {
      reservation = store.ReserveCapture(*runner, 2, kFakePayloadBytes);
      Expect(reservation != nullptr, "capture fills remaining staging");
    } else {
      Expect(store.SaveAsync(runner, {3, 4}, MakeSnapshot(*runner, 43, 2)) ==
                 bytes,
             "second queued payload fills remaining staging");
    }
    Expect(entered.try_acquire_for(std::chrono::seconds(2)),
           "full queued/capture staging bypasses the idle gate");
    Expect(
        !store.CanSave(*runner, 2, kFakePayloadBytes) &&
            !store.ReserveCapture(*runner, 2, kFakePayloadBytes) &&
            store.SaveAsync(runner, {5, 6}, MakeSnapshot(*runner, 44, 2)) == 0,
        "deferred and active bytes retain the original staging bound");
    reservation.reset();
    release.release();
    store.Flush();
    Expect(store.entry_count() == (capture_pressure ? 1U : 2U) &&
               store.CanSave(*runner, 2, kFakePayloadBytes),
           "accepted payloads persist and release staging after completion");
  }
}

void TestDeferredPersistenceFlushAndShutdown() {
  for (const bool explicit_flush : {false, true}) {
    TemporaryDirectory directory;
    auto runner = std::make_shared<FakeRunner>("defer-drain");
    std::binary_semaphore entered(0);
    runner->before_stream = [&] { entered.release(); };
    auto store =
        std::make_unique<ContinuationDiskStore>(StoreOptions(directory.path()));
    {
      auto active = store->BeginGeneration();
      Expect(
          store->SaveAsync(runner, {1, 2}, MakeSnapshot(*runner, 42, 2)) != 0,
          "drain payload is admitted");
      if (DiskDeferExpected())
        Expect(!entered.try_acquire_for(std::chrono::milliseconds(50)),
               "drain starts with a deferred payload");
      if (explicit_flush) {
        auto drained = std::async(std::launch::async, [&] { store->Flush(); });
        Expect(drained.wait_for(std::chrono::seconds(2)) ==
                   std::future_status::ready,
               "Flush bypasses deferral even while generation remains active");
        drained.get();
        Expect(store->entry_count() == 1,
               "Flush persists the deferred payload");
      }
    }  // Token must end before the store; idle notification may race shutdown.
    const auto start = std::chrono::steady_clock::now();
    store.reset();
    Expect(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
           "shutdown drains without waiting for the ten-second deadline");
    ContinuationDiskStore restarted(StoreOptions(directory.path()));
    auto state = runner->CreateState();
    Expect(
        RestoreTokens(restarted, *runner, *state, {1, 2, 3}).restored &&
            RequireFakeState(*state).value == 42,
        "shutdown loses no deferred persist and bytes restore after restart");
  }
}

void TestIndexedPrefixLookup() {
  TemporaryDirectory directory;
  FakeRunner runner("prefix-index");
  ContinuationDiskStore store(StoreOptions(directory.path()));
  // Many independent branches, with terminal prefixes inside compressed edges.
  for (TextRunnerToken branch = 0; branch < 96; ++branch) {
    Expect(SaveTokens(store, runner, {7, branch, 8, 9},
                      *MakeSnapshot(runner, branch, 4))
               .stored,
           "indexed branch stores");
  }
  Expect(
      SaveTokens(store, runner, {7, 45}, *MakeSnapshot(runner, 999, 2)).stored,
      "shorter prefix splits an existing compressed edge");
  auto state = runner.CreateState();
  Expect(RestoreTokens(store, runner, *state, {7, 45, 8, 9, 10}).token_count ==
                 4 &&
             RequireFakeState(*state).value == 45,
         "longest terminal on the matching path wins");
  Expect(RestoreTokens(store, runner, *state, {7, 45, 10}).token_count == 2 &&
             RequireFakeState(*state).value == 999,
         "a mismatched edge falls back to its terminal ancestor");
  Expect(store.SharedPrefixBoundaries(
             runner, std::vector<TextRunnerToken>{7, 46, 8, 10}, 1, 4) ==
             std::vector<std::size_t>({1, 3}),
         "index returns branch and partial-edge boundaries in prompt order");
}

}  // namespace

void TestImageIdentitySurvivesRestart() {
  TemporaryDirectory directory;
  FakeRunner runner("image-cache-test");
  const std::vector<TextRunnerToken> prompt{1, 248056, 3};
  const std::vector<std::uint8_t> image_a{1, 2}, image_b{1, 3};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(store
               .Save(runner, prompt, *MakeSnapshot(runner, 10, prompt.size()),
                     image_a)
               .stored,
           "first image is persisted");
    Expect(store
               .Save(runner, prompt, *MakeSnapshot(runner, 20, prompt.size()),
                     image_b)
               .stored,
           "second image has an independent disk entry");
  }
  ContinuationDiskStore store(StoreOptions(directory.path()));
  auto state = runner.CreateState();
  Expect(store.RestoreLongestPrefix(runner, *state, prompt, image_a).restored &&
             RequireFakeState(*state).value == 10,
         "restart restores first image exactly");
  Expect(store.RestoreLongestPrefix(runner, *state, prompt, image_b).restored &&
             RequireFakeState(*state).value == 20,
         "restart restores second image exactly");
  Expect(!store.RestoreLongestPrefix(runner, *state, prompt).restored,
         "text cannot restore a disk image continuation");
  Expect(!store.Touch(runner, prompt), "disk dedup respects image identity");
}

void TestLearnedBoundaryRoundsDownToTheAlignment() {
  // Lead 1: a learned shared-prefix boundary is rounded DOWN to a multiple of 256 tokens by default (prefill splits at any other position change the
  // resulting state), GUFO_DISK_BOUNDARY_ALIGN=0 or 1 disables it, any other N rounds down to a multiple of N; a boundary that rounds to 0 is dropped.
  TemporaryDirectory directory;
  FakeRunner runner("boundary-align");
  ContinuationDiskStore store(StoreOptions(directory.path()));
  const auto build = [](std::size_t shared, TextRunnerToken tail) {
    std::vector<TextRunnerToken> tokens(shared);
    for (std::size_t i = 0; i < shared; ++i) tokens[i] = static_cast<TextRunnerToken>(10 + i % 50);
    for (TextRunnerToken k = 0; k < 4; ++k) tokens.push_back(tail + k);
    return tokens;
  };
  const auto stored = build(600, 1000);
  Expect(store.Save(runner, stored, *MakeSnapshot(runner, 1, stored.size())).stored, "long prompt stores");
  const auto query = build(600, 2000);
  const auto learned = [&](const char* align) {
    ::setenv("GUFO_DISK_BOUNDARY_ALIGN", align, 1);
    return store.SharedPrefixBoundaries(runner, query, 1, 4);
  };
  Expect(learned("") == std::vector<std::size_t>({512}), "default alignment rounds the 600-token boundary down to 512");
  Expect(learned("0") == std::vector<std::size_t>({600}), "alignment 0 keeps the exact boundary");
  Expect(learned("1") == std::vector<std::size_t>({600}), "alignment 1 keeps the exact boundary");
  Expect(learned("64") == std::vector<std::size_t>({576}), "alignment 64 rounds down to a multiple of 64");
  Expect(learned("250") == std::vector<std::size_t>({500}), "alignment 250 rounds the 600-token boundary down to 500");
  const auto short_stored = build(200, 3000);
  Expect(store.Save(runner, short_stored, *MakeSnapshot(runner, 2, short_stored.size())).stored, "short-prefix prompt stores");
  const auto short_query = build(200, 4000);
  ::setenv("GUFO_DISK_BOUNDARY_ALIGN", "", 1);
  Expect(store.SharedPrefixBoundaries(runner, short_query, 1, 4).empty(), "a boundary that rounds down to 0 is dropped");
  ::setenv("GUFO_DISK_BOUNDARY_ALIGN", "0", 1);
  Expect(store.SharedPrefixBoundaries(runner, short_query, 1, 4) == std::vector<std::size_t>({200}), "alignment 0 keeps the short boundary");
}

void TestAppendedImagePrefixesSurviveRestart() {
  using gufo::server::ContinuationInputPrefix;
  TemporaryDirectory directory;
  FakeRunner runner("image-prefix-cache");
  const std::vector<TextRunnerToken> prompt{1, 2, 3, 248056, 4, 5, 248056, 6};
  const auto tokens = std::span<const TextRunnerToken>(prompt);
  const std::vector<std::uint8_t> a{10}, b{20}, ab{30}, ba{40};
  const std::vector<ContinuationInputPrefix> prefixes{{3, {}}, {6, a}};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(store.Save(runner, tokens.first(3), *MakeSnapshot(runner, 10, 3))
               .stored,
           "text prefix persists without future image identity");
    Expect(store.Save(runner, tokens.first(5), *MakeSnapshot(runner, 20, 5), a)
               .stored,
           "first-image prefix persists independently of the second image");
    Expect(store.Save(runner, tokens, *MakeSnapshot(runner, 30, 8), ab).stored,
           "full two-image checkpoint is persisted");
  }
  ContinuationDiskStore store(StoreOptions(directory.path()));
  auto state = runner.CreateState();
  auto result =
      store.RestoreLongestPrefix(runner, *state, tokens, ab, 7, prefixes);
  Expect(result.restored && result.token_count == 8 &&
             RequireFakeState(*state).value == 30,
         "exact disk retry finds its fallback under an earlier image identity");
  const std::vector<std::uint8_t> ac{50};
  result = store.RestoreLongestPrefix(runner, *state, tokens, ac, 7, prefixes);
  Expect(result.restored && result.token_count == 5 &&
             RequireFakeState(*state).value == 20,
         "changing only the second image restores the first-image checkpoint");
  const std::vector<ContinuationInputPrefix> reordered{{3, {}}, {6, b}};
  result = store.RestoreLongestPrefix(runner, *state, tokens, ba, 7, reordered);
  Expect(result.restored && result.token_count == 3 &&
             RequireFakeState(*state).value == 10,
         "reordered or changed first image restores only the text prefix");
  result = store.RestoreLongestPrefix(runner, *state, tokens);
  Expect(result.restored && result.token_count == 3,
         "literal image-pad text cannot recover visual state");
  const std::vector<ContinuationInputPrefix> moved{{2, {}}};
  result = store.RestoreLongestPrefix(runner, *state, tokens, b, 0, moved);
  Expect(!result.restored,
         "moving the first image before the checkpoint invalidates reuse");
}

void TestMinimumCheckpointStep() {
  TemporaryDirectory directory;
  auto runner = std::make_shared<FakeRunner>("min-step-model");
  std::vector<ContinuationDiskEvent> events;
  auto options = StoreOptions(directory.path());
  options.min_checkpoint_step_tokens = 3;
  ContinuationDiskStore store(
      options, [&](const auto& event) { events.push_back(event); });

  Expect(
      SaveTokens(store, *runner, {1, 2}, *MakeSnapshot(*runner, 1, 2)).stored,
      "the first checkpoint has no covering prefix and is stored");
  Expect(store.WithinCheckpointStep(*runner,
                                    std::vector<TextRunnerToken>{1, 2, 3, 4}) &&
             !store.WithinCheckpointStep(
                 *runner, std::vector<TextRunnerToken>{1, 2, 3, 4, 5}) &&
             !store.WithinCheckpointStep(*runner,
                                         std::vector<TextRunnerToken>{8, 8}),
         "callers can skip a capture the step would discard");
  Expect(!SaveTokens(store, *runner, {1, 2, 3, 4}, *MakeSnapshot(*runner, 2, 4))
                 .stored &&
             events.back().reason == ContinuationDiskEventReason::kMinStep,
         "a checkpoint less than one step past a stored prefix is skipped");
  Expect(
      SaveTokens(store, *runner, {9, 9}, *MakeSnapshot(*runner, 3, 2)).stored,
      "an unrelated checkpoint is not covered by another prompt");
  Expect(
      SaveTokens(store, *runner, {1, 2, 3, 4, 5}, *MakeSnapshot(*runner, 4, 5))
          .stored,
      "a checkpoint one full step past the prefix is stored");

  Expect(store.SaveAsync(runner, {1, 2, 3, 4, 5, 6},
                         MakeSnapshot(*runner, 5, 6), {}, {},
                         /*shared_prefix=*/true) != 0,
         "a shared prefix is queued");
  store.Flush();
  auto state = runner->CreateState();
  Expect(
      RestoreTokens(store, *runner, *state, {1, 2, 3, 4, 5, 6, 7}).restored &&
          RequireFakeState(*state).value == 5,
      "shared prefixes are exempt from the step");

  options.min_checkpoint_step_tokens = 0;
  TemporaryDirectory ungated_directory;
  options.directory = ungated_directory.path();
  ContinuationDiskStore ungated(options);
  Expect(
      SaveTokens(ungated, *runner, {1, 2}, *MakeSnapshot(*runner, 1, 2))
              .stored &&
          SaveTokens(ungated, *runner, {1, 2, 3}, *MakeSnapshot(*runner, 2, 3))
              .stored,
      "zero disables the step");
}

// Policy tests use per-store overrides: environment settings remain read once.
std::vector<char> ReadWriterFile(const std::filesystem::path& file) {
  std::ifstream input(file, std::ios::binary);
  Expect(input.good(), "writer comparison file opens");
  return {std::istreambuf_iterator<char>(input), {}};
}

struct WriterHarness {
  std::shared_ptr<FakeRunner> runner =
      std::make_shared<FakeRunner>("writer-policy");
  gufo::server::ContinuationCache cache;
  ContinuationDiskStore store;

  WriterHarness(const ContinuationDiskStoreOptions& options,
                ContinuationDiskStore::EventSink sink = {},
                std::size_t ram_entries = 4, std::size_t ram_bytes = 64)
      : cache(
            1, [this] { return runner->CreateState(); },
            {.restore =
                 [this](auto& state, const auto& snapshot) {
                   runner->RestoreOrFork(
                       dynamic_cast<TextRunnerState&>(state),
                       dynamic_cast<const TextRunnerSnapshot&>(snapshot));
                 },
             .capacity_bytes = [ram_bytes] { return ram_bytes; }},
            ram_entries),
        store(options, std::move(sink)) {
    if (store.persist_on_evict_idle()) {
      cache.SetPersistenceEvictionHook(
          [this](auto snapshot) { store.PersistEvicted(std::move(snapshot)); });
      store.SetPersistenceSource(runner, [this](bool failed) {
        return cache.NextPendingPersistence(failed);
      });
    }
  }

  std::shared_ptr<const TextRunnerSnapshot> Capture(
      std::vector<TextRunnerToken> tokens, std::uint64_t value,
      bool restored = false, bool shared = false) {
    auto lease = cache.Acquire(tokens);
    Expect(lease.TryReserveSnapshot(
               kFakePayloadBytes, tokens.size(), false,
               gufo::server::SnapshotPurpose::kContinuation, tokens),
           "policy capture obtains RAM reservation");
    std::shared_ptr<const TextRunnerSnapshot> snapshot =
        MakeSnapshot(*runner, value, tokens.size());
    (void)store.Capture(runner, tokens, snapshot, {}, {}, shared, restored);
    Expect(lease.Commit(std::move(tokens), snapshot) == kFakePayloadBytes,
           "policy capture retains snapshot");
    return snapshot;
  }
};

ContinuationDiskStoreOptions WriterOptions(
    const std::filesystem::path& directory) {
  auto options = StoreOptions(directory);
  options.persist_evict_idle = true;
  options.slice = false;
  options.idle_wait = std::chrono::minutes(10);
  return options;
}

void TestWriterEvictionBytesAndDedup() {
  // Exercise both entry-count and byte-pressure eviction, without changing LRU.
  for (const bool byte_pressure : {false, true}) {
    TemporaryDirectory directory, reference_directory;
    WriterHarness writer(WriterOptions(directory.path()), {},
                         byte_pressure ? 4 : 2, byte_pressure ? 32 : 64);
    auto first = writer.Capture({1}, 41);
    writer.Capture({2}, 42);
    Expect(writer.store.entry_count() == 0,
           "RAM captures do not write through");
    writer.Capture({3}, 43);
    writer.store.Flush();
    Expect(writer.store.entry_count() == 1,
           "only the LRU-evicted entry persists");
    Expect(first->persistence->state ==
               gufo::server::SnapshotPersistence::kPersisted,
           "evicted entry marked only after successful persistence");
    writer.store.PersistEvicted(first);
    writer.store.Flush();
    Expect(writer.store.entry_count() == 1,
           "second eviction offer cannot write twice");
    auto state = writer.runner->CreateState();
    Expect(
        RestoreTokens(writer.store, *writer.runner, *state, {1, 99}).restored &&
            RequireFakeState(*state).value == 41,
        "the exactly evicted state restores");
    Expect(
        !RestoreTokens(writer.store, *writer.runner, *state, {2, 99}).restored,
        "retained RAM entry has not persisted");
    auto reference_options = WriterOptions(reference_directory.path());
    reference_options.persist_evict_idle = false;
    WriterHarness reference(reference_options);
    reference.Capture({1}, 41);
    reference.store.Flush();
    Expect(ReadWriterFile(CacheFiles(directory.path()).front()) ==
               ReadWriterFile(CacheFiles(reference_directory.path()).front()),
           "eviction and write-through complete file bytes are identical");
    writer.store.Shutdown();
    Expect(writer.store.entry_count() == 3,
           "shutdown flushes remaining RAM entries");
  }
}

void TestWriterIdleBoundNewestFirstAndAdmission() {
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  options.idle_wait = std::chrono::milliseconds(100);
  std::counting_semaphore<16> entered(0), stored(0);
  std::binary_semaphore release(0);
  std::atomic<std::size_t> idle_count{0}, first_tokens{0};
  WriterHarness writer(options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kStored &&
        event.trigger == "idle") {
      if (idle_count.fetch_add(1) == 0)
        first_tokens.store(event.token_count);
      stored.release();
    }
  });
  bool first = true;
  std::chrono::steady_clock::time_point idle_start;
  writer.runner->before_stream = [&] {
    entered.release();
    if (std::exchange(first, false))
      release.acquire();
  };
  {
    auto pending_admission = writer.store.BeginAdmission();
    writer.Capture({1}, 41);
    writer.Capture({2, 2}, 42);
    writer.Capture({3, 3, 3}, 43);
    Expect(!entered.try_acquire_for(std::chrono::milliseconds(150)),
           "pending admission prevents idle persistence even after idle bound");
    idle_start = std::chrono::steady_clock::now();
  }
  Expect(entered.try_acquire_for(std::chrono::seconds(10)),
         "idle persistence eventually fires");
  Expect(std::chrono::steady_clock::now() - idle_start >=
             options.idle_wait.value(),
         "idle deadline starts when admission ends");
  {
    auto admitted = writer.store.BeginAdmission();
    release.release();
    Expect(stored.try_acquire_for(std::chrono::seconds(10)),
           "already-started idle write finishes");
    Expect(first_tokens == 3, "idle drains newest RAM entry first");
    Expect(!entered.try_acquire_for(std::chrono::milliseconds(150)),
           "admission stops the next idle entry");
    Expect(writer.store.entry_count() == 1,
           "idle persists one entry at a time");
  }
  writer.store.Shutdown();
  Expect(writer.store.entry_count() == 3 && idle_count == 1,
         "shutdown flushes the two remaining entries with shutdown trigger");
}

void TestWriterRestoredAndAlreadyStoredEntries() {
  TemporaryDirectory directory;
  std::atomic<std::size_t> deduped{0};
  WriterHarness writer(WriterOptions(directory.path()), [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kAlreadyPersisted)
      ++deduped;
  });
  auto saved = MakeSnapshot(*writer.runner, 41, 1);
  Expect(SaveTokens(writer.store, *writer.runner, {1}, *saved).stored,
         "seed represents restart-persistent state");
  auto restored = writer.Capture({1}, 41, true);
  Expect(restored->persistence->state ==
             gufo::server::SnapshotPersistence::kPersisted,
         "disk-restored entry is marked persisted at capture");
  auto exact = writer.Capture({2}, 42);
  Expect(SaveTokens(writer.store, *writer.runner, {2}, *exact).stored,
         "another caller has already stored the exact RAM entry");
  std::atomic<std::size_t> serializations{0};
  writer.runner->before_stream = [&] { ++serializations; };
  writer.store.PersistEvicted(restored);
  writer.store.PersistEvicted(exact);
  writer.store.Flush();
  auto replacement = writer.Capture({2}, 42);
  Expect(replacement->persistence->state ==
             gufo::server::SnapshotPersistence::kPersisted,
         "identical-token replacement inherits persisted status");
  writer.store.Shutdown();
  Expect(serializations == 0 && deduped == 1 && writer.store.entry_count() == 2,
         "restore and exact-file dedup cause no payload rewrite");
}

void TestWriterShutdownDeadlineAndDiscardLogging() {
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  options.shutdown_wait = std::chrono::milliseconds(0);
  std::atomic<std::size_t> deadline_skips{0}, ram_skips{0};
  WriterHarness writer(options, [&](const auto& event) {
    if (event.reason == ContinuationDiskEventReason::kShutdownDeadline) {
      Expect(event.action == ContinuationDiskEventAction::kDroppedEvict &&
                 event.trigger == "shutdown", "shutdown deadline counts a triggered drop");
      ++deadline_skips;
    }
    if (event.reason == ContinuationDiskEventReason::kRamNotRetained)
      ++ram_skips;
  });
  writer.Capture({1}, 41);
  writer.Capture({2}, 42);
  {
    auto abandoned = MakeSnapshot(*writer.runner, 44, 1);
    (void)writer.store.Capture(writer.runner, {4}, std::move(abandoned));
    // No cache publication: the tag logs this discard itself.
  }
  writer.store.Shutdown();
  Expect(
      deadline_skips == 2 && ram_skips == 1 && writer.store.entry_count() == 0,
      "bounded shutdown and unpublished captures each log their "
      "skip");
}

void TestWriterDefaultsStillWriteThrough() {
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  options.persist_evict_idle = false;
  WriterHarness writer(options);
  auto active = writer.store.BeginGeneration();
  auto snapshot = writer.Capture({1}, 41);
  Expect(!snapshot->persistence,
         "write-through does not allocate persistence tags");
  writer.store.Flush();
  Expect(writer.store.entry_count() == 1,
         "default writes capture without eviction or idle");
}

class LargeWriterRunner final : public TextModelRunner {
public:
  TextRunnerDescriptor Descriptor() const override {
    return base_.Descriptor();
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return base_.ResourceClaim();
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return base_.SupportedPlans();
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    return base_.Tokenize(text);
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return base_.RenderAndTokenize(request);
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    return base_.Decode(tokens);
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    return base_.CreateState();
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> tokens,
                          std::size_t offset,
                          std::size_t budget) const override {
    return base_.Prefill(state, tokens, offset, budget);
  }
  TextDecodeSelection SelectNext(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    return base_.SelectNext(state, sampler);
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    base_.Advance(state, token);
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return base_.CheckpointPosition(state);
  }
  std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return base_.SnapshotPayloadBytes(state);
  }
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    return base_.Snapshot(state);
  }
  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    base_.RestoreOrFork(state, snapshot);
  }
  std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot&) const override {
    return kFakePayloadBytes + 32 * 1024 * 1024;
  }
  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    base_.StreamPersistentSnapshot(snapshot, sink);
    // Deliberately irregular provider segments cross multiple writer slices.
    std::vector<std::uint8_t> bytes(5 * 1024 * 1024, 0x5a);
    std::size_t left = 32 * 1024 * 1024;
    while (left) {
      const auto size = std::min(left, bytes.size());
      sink(std::span(bytes).first(size));
      left -= size;
    }
  }

private:
  FakeRunner base_{"writer-slice"};
};

void TestWriterSlicingBytesAndActiveYield() {
  TemporaryDirectory sliced_directory, reference_directory;
  auto runner = std::make_shared<LargeWriterRunner>();
  auto reference_options = WriterOptions(reference_directory.path());
  reference_options.persist_evict_idle = false;
  reference_options.capacity_bytes = reference_options.staging_capacity_bytes =
      64 * 1024 * 1024;
  ContinuationDiskStore reference(reference_options);
  auto state = runner->CreateState();
  RequireFakeState(*state).value = 41;
  RequireFakeState(*state).position = 1;
  auto snapshot = runner->Snapshot(*state);
  const std::vector<TextRunnerToken> tokens{1};
  Expect(reference.Save(*runner, tokens, *snapshot).stored,
         "unsliced reference stores");
  auto sliced_options = reference_options;
  sliced_options.directory = sliced_directory.path();
  sliced_options.slice = true;
  sliced_options.slice_max_wait = std::chrono::milliseconds(20);
  std::size_t yields = 0;
  double wait_ms = 0;
  ContinuationDiskStore sliced(sliced_options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kStored) {
      yields = event.slice_yields;
      wait_ms = event.slice_wait_ms;
    }
  });
  {
    auto active = sliced.BeginGeneration();
    Expect(sliced.Save(*runner, tokens, *snapshot).stored,
           "sliced active write stores");
  }
  Expect(yields > 0 && wait_ms >= 20 && wait_ms < 1000,
         "active generation causes bounded between-slice yields");
  Expect(ReadWriterFile(CacheFiles(sliced_directory.path()).front()) ==
             ReadWriterFile(CacheFiles(reference_directory.path()).front()),
         "slicing preserves all header, payload and checksum bytes");
}

void TestWriterStagingAndSharedSpacing() {
  // One active payload, two extra pinned eviction offers, then explicit drop.
  TemporaryDirectory directory;
  auto backlog_options = WriterOptions(directory.path());
  auto runner = std::make_shared<FakeRunner>("backlog");
  const auto charge = ExpectedFileBytes(
      runner->Descriptor().persistence->compatibility_identity.size(), 1);
  backlog_options.staging_capacity_bytes = charge;
  backlog_options.evict_backlog_bytes = 2 * charge;
  std::binary_semaphore entered(0), release(0);
  std::atomic<std::size_t> writes{0}, drops{0};
  ContinuationDiskStore store(backlog_options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kStored) ++writes;
    if (event.action == ContinuationDiskEventAction::kDroppedEvict) {
      Expect(event.reason == ContinuationDiskEventReason::kEvictBacklogCapacity,
             "overflow identifies bounded backlog exhaustion");
      ++drops;
    }
  });
  store.SetPersistenceSource(runner, [](bool) {
    return std::shared_ptr<const gufo::server::ContinuationSnapshot>{};
  });
  std::atomic<bool> first{true};
  runner->before_stream = [&] {
    if (first.exchange(false)) { entered.release(); release.acquire(); }
  };
  auto offer = [&](TextRunnerToken token) {
    std::shared_ptr<const TextRunnerSnapshot> snapshot =
        MakeSnapshot(*runner, token, 1);
    (void)store.Capture(runner, {token}, snapshot);
    store.PersistEvicted(snapshot);
    return snapshot;
  };
  auto active = offer(1);
  Expect(entered.try_acquire_for(std::chrono::seconds(10)),
         "first eviction is actively streaming");
  auto second = offer(2), third = offer(3), overflow = offer(4);
  std::weak_ptr<const TextRunnerSnapshot> pinned_second = second;
  std::weak_ptr<const TextRunnerSnapshot> pinned_third = third;
  second.reset(); third.reset();
  Expect(!pinned_second.expired() && !pinned_third.expired() && drops == 1 &&
             overflow->persistence->state ==
                 gufo::server::SnapshotPersistence::kAbandoned,
         "busy writer pins two eviction offers and explicitly drops overflow");
  Expect(!store.CanSave(*runner, 1, kFakePayloadBytes),
         "backlog does not weaken active-plus-queued staging admission");
  release.release();
  store.Flush();
  Expect(writes == 3 && store.entry_count() == 3 &&
             pinned_second.expired() && pinned_third.expired(),
         "all bounded eviction offers drain and release their memory");
  store.Shutdown();

  // Write-through has no extra backlog, even if its byte bound is configured.
  TemporaryDirectory through_directory;
  backlog_options.directory = through_directory.path();
  backlog_options.persist_evict_idle = false;
  ContinuationDiskStore through(backlog_options);
  first.store(true);
  Expect(through.SaveAsync(runner, {5}, MakeSnapshot(*runner, 5, 1)) != 0,
         "write-through stages first payload");
  Expect(entered.try_acquire_for(std::chrono::seconds(10)),
         "write-through payload holds its staging charge while running");
  Expect(through.SaveAsync(runner, {6}, MakeSnapshot(*runner, 6, 1)) == 0,
         "write-through still refuses second payload at the staging limit");
  release.release(); through.Flush();
  Expect(through.entry_count() == 1, "write-through bound remains strict");

  TemporaryDirectory shared_directory;
  auto options = WriterOptions(shared_directory.path());
  options.min_checkpoint_step_tokens = 10;
  WriterHarness shared(options);
  Expect(SaveTokens(shared.store, *shared.runner, {1},
                    *MakeSnapshot(*shared.runner, 40, 1))
             .stored,
         "seed minimum-step prefix");
  auto boundary = shared.Capture({1, 2}, 41, false, true);
  auto continuation = shared.Capture({1, 3}, 42);
  shared.store.Shutdown();
  Expect(shared.store.entry_count() == 2 &&
             boundary->persistence->state ==
                 gufo::server::SnapshotPersistence::kPersisted &&
             continuation->persistence->state !=
                 gufo::server::SnapshotPersistence::kPersisted,
         "deferred shared boundary keeps exemption while continuation keeps "
         "min-step gate");
}

void TestWriterBatchEvictionsAndBacklogByteBound() {
  for (const bool byte_bound : {false, true}) {
    TemporaryDirectory directory;
    auto options = WriterOptions(directory.path());
    FakeRunner sizing("writer-policy");
    const auto charge = ExpectedFileBytes(
        sizing.Descriptor().persistence->compatibility_identity.size(), 1);
    options.staging_capacity_bytes = charge;
    // A single reserve offers three snapshots: bytes, not the old count=2, limit it.
    options.evict_backlog_bytes = byte_bound ? charge : 3 * charge;
    std::atomic<std::size_t> drops{0}, writes{0};
    WriterHarness writer(options, [&](const auto& event) {
      if (event.action == ContinuationDiskEventAction::kDroppedEvict) {
        Expect(event.reason == ContinuationDiskEventReason::kEvictBacklogCapacity,
               "batch overflow reports byte capacity");
        ++drops;
      }
      if (event.action == ContinuationDiskEventAction::kStored) ++writes;
    }, 5, 4 * kFakePayloadBytes);
    auto one = writer.Capture({1}, 1);
    auto two = writer.Capture({2}, 2);
    auto three = writer.Capture({3}, 3);
    auto four = writer.Capture({4}, 4);
    std::binary_semaphore entered(0), release(0);
    std::atomic<bool> first{true};
    writer.runner->before_stream = [&] {
      if (first.exchange(false)) { entered.release(); release.acquire(); }
    };
    std::shared_ptr<const TextRunnerSnapshot> active =
        MakeSnapshot(*writer.runner, 5, 1);
    (void)writer.store.Capture(writer.runner, {5}, active);
    writer.store.PersistEvicted(active);
    Expect(entered.try_acquire_for(std::chrono::seconds(10)), "active batch writer");
    {
      auto lease = writer.cache.Acquire(std::vector<TextRunnerToken>{9});
      Expect(lease.TryReserveSnapshot(3 * kFakePayloadBytes, 1),
             "one reservation evicts three snapshots");
      Expect(one->persistence->state == gufo::server::SnapshotPersistence::kQueued &&
                 two->persistence->state == (byte_bound
                     ? gufo::server::SnapshotPersistence::kAbandoned
                     : gufo::server::SnapshotPersistence::kQueued) &&
                 three->persistence->state == (byte_bound
                     ? gufo::server::SnapshotPersistence::kAbandoned
                     : gufo::server::SnapshotPersistence::kQueued) &&
                 drops == (byte_bound ? 2 : 0),
             "batch evictions wait in bounded backlog instead of staging loss");
      lease.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure,
                         3 * kFakePayloadBytes, 1);
    }
    release.release(); writer.store.Flush();
    Expect(writes == (byte_bound ? 2 : 4), "batch accepted offers all persist");
    writer.store.Shutdown();
    Expect(four->persistence->state == gufo::server::SnapshotPersistence::kPersisted,
           "remaining RAM snapshot persists at shutdown");
  }
}

void TestWriterBacklogSafetyCount() {
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  auto runner = std::make_shared<FakeRunner>("backlog-safety");
  const auto charge = ExpectedFileBytes(
      runner->Descriptor().persistence->compatibility_identity.size(), 1);
  options.staging_capacity_bytes = charge;
  // Keep the default 2 GiB byte bound; these tiny offers reach the safety guard.
  std::atomic<std::size_t> writes{0}, drops{0};
  ContinuationDiskStore store(options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kStored) ++writes;
    if (event.action == ContinuationDiskEventAction::kDroppedEvict) {
      Expect(event.reason == ContinuationDiskEventReason::kEvictBacklogCapacity &&
                 event.trigger == "evict", "safety overflow has drop reason and trigger");
      ++drops;
    }
  });
  store.SetPersistenceSource(runner, [](bool) {
    return std::shared_ptr<const gufo::server::ContinuationSnapshot>{};
  });
  std::binary_semaphore entered(0), release(0);
  std::atomic<bool> first{true};
  runner->before_stream = [&] {
    if (first.exchange(false)) { entered.release(); release.acquire(); }
  };
  auto offer = [&](TextRunnerToken token) {
    std::shared_ptr<const TextRunnerSnapshot> snapshot = MakeSnapshot(*runner, token, 1);
    (void)store.Capture(runner, {token}, snapshot);
    store.PersistEvicted(snapshot);
    return snapshot;
  };
  auto active = offer(1);
  Expect(entered.try_acquire_for(std::chrono::seconds(10)), "safety writer blocks");
  std::vector<std::shared_ptr<const TextRunnerSnapshot>> backlog;
  for (TextRunnerToken token = 2; token <= 9; ++token) {
    backlog.push_back(offer(token));
    Expect(backlog.back()->persistence->state == gufo::server::SnapshotPersistence::kQueued &&
               drops == 0, "eight small snapshots backlog without loss");
  }
  auto overflow = offer(10);
  Expect(drops == 1 && overflow->persistence->state == gufo::server::SnapshotPersistence::kAbandoned,
         "ninth backlogged snapshot reaches absolute safety guard");
  release.release(); store.Flush(); store.Shutdown();
  Expect(writes == 9 && drops == 1, "all accepted small burst offers persist");
  for (const auto& snapshot : backlog)
    Expect(snapshot->persistence->state == gufo::server::SnapshotPersistence::kPersisted,
           "accepted safety burst reaches persisted state");
}

void TestWriterRefusedTriggerAccounting() {
  for (const bool byte_capacity : {false, true}) {
    TemporaryDirectory directory;
    auto options = WriterOptions(directory.path());
    auto runner = std::make_shared<FakeRunner>("refused-trigger");
    const auto charge = ExpectedFileBytes(
        runner->Descriptor().persistence->compatibility_identity.size(), 1);
    if (byte_capacity) options.capacity_bytes = charge - 1;
    else options.staging_capacity_bytes = charge - 1;
    std::vector<ContinuationDiskEvent> events;
    ContinuationDiskStore store(options, [&](const auto& event) { events.push_back(event); });
    for (const auto trigger : {"write_through", "evict", "idle", "shutdown"}) {
      Expect(store.SaveAsync(runner, {1}, MakeSnapshot(*runner, 1, 1), {}, {}, false, trigger) == 0,
             "oversize offer is refused");
      Expect(events.back().trigger == trigger &&
                 events.back().reason == (byte_capacity ? ContinuationDiskEventReason::kByteCapacity
                                                        : ContinuationDiskEventReason::kStagingCapacity) &&
                 events.back().action == (std::string_view(trigger) == "write_through"
                     ? ContinuationDiskEventAction::kSkipped : ContinuationDiskEventAction::kDroppedEvict),
             "capacity refusal retains trigger and counts deferred drop");
    }
    store.Shutdown();
    Expect(store.SaveAsync(runner, {1}, MakeSnapshot(*runner, 1, 1), {}, {}, false, "shutdown") == 0 &&
               events.back().action == ContinuationDiskEventAction::kDroppedEvict &&
               events.back().trigger == "shutdown" &&
               events.back().reason == ContinuationDiskEventReason::kShutdownDeadline,
           "shutdown deadline refusal is a shutdown drop");
  }
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  options.capacity_bytes = 1;
  std::atomic<std::size_t> drops{0};
  WriterHarness writer(options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kDroppedEvict) {
      Expect(event.trigger == "shutdown" && event.reason == ContinuationDiskEventReason::kByteCapacity,
             "source refusal is labelled shutdown drop");
      ++drops;
    }
  });
  auto snapshot = writer.Capture({1}, 1);
  writer.store.Shutdown();
  Expect(drops == 1 && snapshot->persistence->state == gufo::server::SnapshotPersistence::kAbandoned,
         "refused shutdown source is counted and terminal");
}

void TestWriterMissingRecordDeadlineDrop() {
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  options.shutdown_wait = std::chrono::milliseconds(0);
  auto runner = std::make_shared<FakeRunner>("missing-record");
  std::atomic<std::size_t> drops{0};
  ContinuationDiskStore store(options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kDroppedEvict) {
      Expect(event.reason == ContinuationDiskEventReason::kShutdownDeadline &&
                 event.trigger == "shutdown" && event.payload_bytes == kFakePayloadBytes,
             "untagged deadline job emits fallback drop");
      ++drops;
    }
  });
  std::binary_semaphore entered(0), release(0);
  runner->before_stream = [&] { entered.release(); release.acquire(); };
  Expect(store.SaveAsync(runner, {1}, MakeSnapshot(*runner, 1, 1)) != 0, "block writer with first job");
  Expect(entered.try_acquire_for(std::chrono::seconds(10)), "missing-record writer blocks");
  std::shared_ptr<const TextRunnerSnapshot> untagged = MakeSnapshot(*runner, 2, 1);
  Expect(!untagged->persistence && store.SaveAsync(runner, {2}, untagged, {}, {}, false, "shutdown") != 0,
         "queue an untagged shutdown job");
  auto shutdown = std::async(std::launch::async, [&] { store.Shutdown(); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (store.CanSave(*runner, 1, kFakePayloadBytes) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  Expect(std::chrono::steady_clock::now() < deadline, "shutdown reaches deadline before writer release");
  release.release(); shutdown.get();
  Expect(drops == 1, "missing record cannot silently lose queued persist");
}

void TestWriterExactReplacementSharedBoundary() {
  TemporaryDirectory directory;
  auto options = WriterOptions(directory.path());
  options.min_checkpoint_step_tokens = 10;
  std::atomic<std::size_t> superseded{0};
  WriterHarness writer(options, [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kSuperseded) ++superseded;
  }, 1, kFakePayloadBytes);
  Expect(SaveTokens(writer.store, *writer.runner, {7}, *MakeSnapshot(*writer.runner, 1, 1)).stored,
         "seed nearby persisted prefix for min-step gate");
  auto boundary = writer.Capture({7, 2}, 2, false, true);
  auto continuation = writer.Capture({7, 2}, 2);
  writer.store.Flush();
  Expect(boundary->persistence->state == gufo::server::SnapshotPersistence::kPersisted &&
             boundary->persistence->shared_prefix && !continuation->persistence->shared_prefix &&
             superseded == 0 && writer.store.entry_count() == 2,
         "exact non-shared replacement independently persists learned boundary despite min-step");
  writer.store.Shutdown();
  Expect(continuation->persistence->state == gufo::server::SnapshotPersistence::kPersisted,
         "replacement deduplicates against independently persisted boundary");
}

void TestWriterGrowingConversationAndRealDrops() {
  for (bool pressure : {false, true}) {
    TemporaryDirectory directory;
    std::atomic<std::size_t> writes{0}, superseded{0};
    WriterHarness writer(WriterOptions(directory.path()), [&](const auto& event) {
      if (event.action == ContinuationDiskEventAction::kStored) ++writes;
      if (event.action == ContinuationDiskEventAction::kSuperseded) ++superseded;
    }, pressure ? 4 : 1, pressure ? kFakePayloadBytes : 64);
    std::vector<TextRunnerToken> tokens{7};
    auto latest = writer.Capture(tokens, 1);
    for (std::uint64_t turn = 2; turn <= 10; ++turn) {
      tokens.push_back(static_cast<TextRunnerToken>(turn));
      latest = writer.Capture(tokens, turn);
    }
    auto exact = writer.Capture(tokens, 10);
    writer.store.Flush();
    Expect(writes == 0 && superseded == 10,
           "ten growing turns and identical replacement do not write per turn");
    writer.store.Shutdown();
    Expect(writes == 1 && writer.store.entry_count() == 1 &&
               exact->persistence->state ==
                   gufo::server::SnapshotPersistence::kPersisted,
           "shutdown saves only latest conversation checkpoint");
    auto state = writer.runner->CreateState();
    Expect(writer.store.RestoreLongestPrefix(*writer.runner, *state, tokens).restored &&
               RequireFakeState(*state).value == 10,
           "latest turn retains restart persistence");
  }
  // Learned shared boundary remains valuable despite a longer survivor.
  TemporaryDirectory directory;
  WriterHarness writer(WriterOptions(directory.path()), {}, 1, kFakePayloadBytes);
  auto boundary = writer.Capture({7}, 1, false, true);
  writer.Capture({7, 2}, 2);
  writer.store.Flush();
  Expect(boundary->persistence->state ==
             gufo::server::SnapshotPersistence::kPersisted,
         "learned system boundary persists when pressure removes it");
  writer.Capture({7, 3}, 3); // Diverged branch is a real drop.
  writer.store.Flush();
  Expect(writer.store.entry_count() == 2, "diverged tail persists on removal");
  writer.Capture({7}, 1); // Rewind discards the longer history.
  writer.store.Flush();
  Expect(writer.store.entry_count() == 3, "rewound history persists on removal");
  writer.store.Shutdown();
}

void TestWriterEvictSkipCoveredByPersistedPrefix() {
  // GUFO_DISK_EVICT_SKIP_TOKENS policy: an evicted snapshot whose persisted strict prefix is within N tokens is not written again.
  for (const std::size_t skip : {std::size_t{0}, std::size_t{4}, std::size_t{1}}) {
    TemporaryDirectory directory;
    std::atomic<std::size_t> writes{0}, covered{0}, already{0}, covering{0}, gap{0};
    auto options = WriterOptions(directory.path());
    options.evict_skip_tokens = skip;
    WriterHarness writer(options, [&](const auto& event) {
      if (event.action == ContinuationDiskEventAction::kStored) ++writes;
      if (event.action == ContinuationDiskEventAction::kAlreadyPersisted) {
        ++already;
        if (event.reason == ContinuationDiskEventReason::kPrefixPersisted) {
          ++covered;
          covering = event.covering_tokens;
          gap = event.gap_tokens;
        }
      }
    }, 1, 64);
    writer.Capture({1}, 1);        // RAM: {1}
    writer.Capture({2}, 2);        // evicts {1}: persisted (nothing covers it)
    writer.store.Flush();
    writer.Capture({1, 5, 5}, 3);  // evicts {2}: persisted
    writer.store.Flush();
    Expect(writes == 2, "two unrelated evictions persist");
    writer.Capture({3}, 4);        // evicts {1,5,5}: its persisted prefix {1} is 2 tokens shorter
    writer.store.Flush();
    if (skip >= 2) {
      Expect(writes == 2 && covered == 1 && already == 1,
             "an eviction covered by a persisted prefix within the bound is skipped and logged");
      Expect(covering == 1 && gap == 2,
             "the skip event reports the covering prefix length and the gap");
    } else {
      Expect(writes == 3 && covered == 0,
             "bound smaller than the gap (or off) still persists the eviction");
    }
    writer.store.Shutdown();
  }
}

void TestWriterOptionalCopyRefusedIsNotWritten() {
  TemporaryDirectory directory;
  std::atomic<std::size_t> writes{0}, refused{0}, dropped{0};
  WriterHarness writer(WriterOptions(directory.path()), [&](const auto& event) {
    if (event.action == ContinuationDiskEventAction::kStored) ++writes;
    if (event.action == ContinuationDiskEventAction::kDroppedEvict) ++dropped;
    if (event.action == ContinuationDiskEventAction::kSkipped &&
        event.reason == ContinuationDiskEventReason::kRamNotRetained &&
        event.token_count == 7) ++refused;
  }, 1, 64);
  writer.store.NoteOptionalCopyRefused(7);
  writer.store.Flush();
  Expect(writes == 0 && dropped == 0 && refused == 1 && writer.store.entry_count() == 0,
         "a RAM-refused optional copy is logged as skipped/ram_not_retained and never written");
  writer.store.Shutdown();
}

void TestWriterFailedReplacementAndRamRefusal() {
  TemporaryDirectory directory;
  WriterHarness writer(WriterOptions(directory.path()), {}, 1, kFakePayloadBytes);
  auto old = writer.Capture({1}, 1);
  {
    auto lease = writer.cache.Acquire(std::vector<TextRunnerToken>{1, 2});
    Expect(lease.TryReserveSnapshot(kFakePayloadBytes, 2, false,
               gufo::server::SnapshotPurpose::kContinuation,
               std::vector<TextRunnerToken>{1, 2}), "replacement reservation");
    lease.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure,
                       kFakePayloadBytes, 2);
  }
  writer.store.Flush();
  Expect(old->persistence->state == gufo::server::SnapshotPersistence::kPersisted,
         "failed family capture offers displaced original");
  {
    auto lease = writer.cache.Acquire(std::vector<TextRunnerToken>{2});
    std::shared_ptr<const TextRunnerSnapshot> refused =
        MakeSnapshot(*writer.runner, 2, 1);
    (void)writer.store.Capture(writer.runner, {2}, refused, {}, {}, true);
    Expect(lease.Commit({2}, refused) == 0,
           "tagged snapshot publication without reservation refuses RAM");
    writer.store.Flush();
    Expect(refused->persistence->state ==
               gufo::server::SnapshotPersistence::kPersisted,
           "RAM-refused learned boundary falls back to bounded writer");
  }
  (void)writer.store.Capture(writer.runner, {3},
      MakeSnapshot(*writer.runner, 3, 1), {}, {}, true, false, false);
  writer.store.Flush();
  Expect(writer.store.entry_count() == 3,
         "disk-only fallback survives inability to reserve RAM");
  writer.store.Shutdown();
}

void TestWriterConcurrentTouchRestoreAndOffers() {
  TemporaryDirectory directory;
  WriterHarness writer(WriterOptions(directory.path()));
  std::vector<std::shared_ptr<const TextRunnerSnapshot>> snapshots;
  for (TextRunnerToken i = 1; i <= 64; ++i) {
    auto snapshot = MakeSnapshot(*writer.runner, i, 1);
    Expect(SaveTokens(writer.store, *writer.runner, {i}, *snapshot).stored,
           "threaded metadata test seeds exact entries");
    std::shared_ptr<const TextRunnerSnapshot> tagged = std::move(snapshot);
    (void)writer.store.Capture(writer.runner, {i}, tagged);
    snapshots.push_back(std::move(tagged));
  }
  // Corrupt indexed files before concurrency. Restore will erase their entries
  // while public and writer-side Touch are looking up the same metadata.
  auto files = CacheFiles(directory.path());
  for (std::size_t i = 0; i < files.size(); i += 2) {
    std::fstream file(files[i], std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(-1, std::ios::end); file.put('\xff');
  }
  std::counting_semaphore<3> start(0);
  std::thread touch([&] {
    start.acquire();
    for (int round = 0; round < 16; ++round)
      for (TextRunnerToken i = 1; i <= 64; ++i)
        (void)writer.store.Touch(*writer.runner, std::vector<TextRunnerToken>{i});
  });
  std::thread restore([&] {
    start.acquire();
    auto state = writer.runner->CreateState();
    for (int round = 0; round < 16; ++round)
      for (TextRunnerToken i = 1; i <= 64; ++i)
        (void)RestoreTokens(writer.store, *writer.runner, *state, {i});
  });
  std::thread offers([&] {
    start.acquire();
    for (const auto& snapshot : snapshots) writer.store.PersistEvicted(snapshot);
  });
  start.release(3);
  touch.join(); restore.join(); offers.join();
  writer.store.Flush(); writer.store.Shutdown();
  for (const auto& snapshot : snapshots)
    Expect(snapshot->persistence->state != gufo::server::SnapshotPersistence::kQueued,
           "concurrent lookup/erase/dedup/offer leaves no orphaned queued record");
}

void TestWriterShutdownExitOfferRace() {
  for (int round = 0; round < 32; ++round) {
    TemporaryDirectory directory;
    auto options = WriterOptions(directory.path());
    options.shutdown_wait = std::chrono::seconds(10);
    std::atomic<std::size_t> stored{0}, dropped{0};
    WriterHarness writer(options, [&](const auto& event) {
      if (event.action == ContinuationDiskEventAction::kStored) ++stored;
      if (event.action == ContinuationDiskEventAction::kDroppedEvict) ++dropped;
    });
    std::shared_ptr<const TextRunnerSnapshot> snapshot =
        MakeSnapshot(*writer.runner, 1, 1);
    (void)writer.store.Capture(writer.runner, {1}, snapshot);
    std::binary_semaphore start(0);
    std::thread offer([&] { start.acquire(); writer.store.PersistEvicted(snapshot); });
    start.release(); writer.store.Shutdown(); offer.join(); writer.store.Flush();
    Expect(stored + dropped == 1 && snapshot->persistence->state !=
               gufo::server::SnapshotPersistence::kQueued,
           "offer racing exit is durably flushed or explicitly dropped");
    std::shared_ptr<const TextRunnerSnapshot> after =
        MakeSnapshot(*writer.runner, 2, 1);
    (void)writer.store.Capture(writer.runner, {2}, after);
    writer.store.PersistEvicted(after); writer.store.Flush();
    Expect(dropped >= 1 && after->persistence->state ==
               gufo::server::SnapshotPersistence::kAbandoned,
           "offer after join is rejected and never orphaned");
  }
}

int main() {
  // Lead 1 (learned shared-prefix boundaries rounded down to a multiple of 256) is off in these tests: they assert exact tiny boundaries.
  ::setenv("GUFO_DISK_BOUNDARY_ALIGN", "0", 1);
  TestWriterBacklogSafetyCount();
  TestWriterRefusedTriggerAccounting();
  TestWriterMissingRecordDeadlineDrop();
  TestWriterExactReplacementSharedBoundary();
  TestWriterBatchEvictionsAndBacklogByteBound();
  TestWriterGrowingConversationAndRealDrops();
  TestWriterEvictSkipCoveredByPersistedPrefix();
  TestWriterOptionalCopyRefusedIsNotWritten();
  TestWriterFailedReplacementAndRamRefusal();
  TestWriterConcurrentTouchRestoreAndOffers();
  TestWriterShutdownExitOfferRace();
  TestWriterStagingAndSharedSpacing();
  TestWriterEvictionBytesAndDedup();
  TestWriterIdleBoundNewestFirstAndAdmission();
  TestWriterRestoredAndAlreadyStoredEntries();
  TestWriterShutdownDeadlineAndDiscardLogging();
  TestWriterDefaultsStillWriteThrough();
  TestWriterSlicingBytesAndActiveYield();
  TestDeferredPersistenceIdleAndBytes();
  TestDeferredPersistenceDeadline();
  TestDeferredPersistencePressure();
  TestDeferredPersistenceFlushAndShutdown();
  TestAppendedImagePrefixesSurviveRestart();
  TestBoundedAsyncPersistenceDoesNotBlockLookup();
  TestIndexedPrefixLookup();
  TestLearnedBoundaryRoundsDownToTheAlignment();
  TestMinimumCheckpointStep();
  TestImageIdentitySurvivesRestart();
  TestSharedPrefixBoundariesAndExactDedup();
  TestSha256KnownVector();
  TestRestartRestoreAndCompatibilityIdentity();
  TestLongestPrefixAndForcedHashCollision();
  TestCorruptionBecomesDeterministicMissAndRemoval();
  TestByteAndStagingLimits();
  TestAutomaticStagingAndAdmissionDiagnostics();
  TestSmallerStagingPreservesExistingFiles();
  TestLruEvictionUsesActualFileBytes();
  TestAtomicPublicationAndPrivatePermissions();
  TestStartupRejectsUnsafeAndInvalidFiles();
  TestRootSymlinkIsRejected();
  TestConcurrentCallersRemainBoundedAndExact();
  TestConcurrentStoreInstancesPublishSafely();
  TestStartupPreservesActivePublisher();
  std::cout << "continuation disk store tests passed\n";
  return 0;
}
