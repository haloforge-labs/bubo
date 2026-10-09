#ifndef GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_
#define GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class ContinuationDiskEventAction : std::uint8_t {
  kStored,
  kRestored,
  kMiss,
  kRemoved,
  kSkipped,
  kDroppedEvict,
  kAlreadyPersisted,
  kSuperseded,
};

enum class ContinuationDiskEventReason : std::uint8_t {
  kSaved,
  kHit,
  kNotFound,
  kUnsupported,
  kByteCapacity,
  kStagingCapacity,
  kLru,
  kExactReplacement,
  /// A stored prefix already covers this checkpoint within the minimum step.
  kMinStep,
  kCorrupt,
  kChecksumMismatch,
  kUnsafeFile,
  kIoFailure,
  kSerializationFailure,
  kRestoreFailure,
  kBusy,
  kRamNotRetained,
  kShutdownDeadline,
  kEvictBacklogCapacity,
  kWriterStopped,
  kAlreadyPersisted,
  kSuperseded,
  kPrefixPersisted,
};

/// Sanitized disk-cache event. Prompt contents, token values, paths, model
/// identities, and compatibility fingerprints are deliberately absent.
struct ContinuationDiskEvent {
  ContinuationDiskEventAction action{ContinuationDiskEventAction::kMiss};
  ContinuationDiskEventReason reason{ContinuationDiskEventReason::kNotFound};
  std::size_t file_bytes{0};
  std::size_t payload_bytes{0};
  std::size_t token_count{0};
  std::size_t retained_bytes{0};
  std::size_t capacity_bytes{0};
  std::size_t staging_capacity_bytes{0};
  std::size_t staging_used_bytes{0};
  double elapsed_ms{0.0};
  std::string_view trigger{"write_through"};
  std::size_t slice_yields{0};
  double slice_wait_ms{0.0};
  // Phase split of a completed write (prefill thread 2026-10-08, writer-fix diagnostics): serialize = device->host gather inside StreamPersistentSnapshot,
  // write = ::write time, hash = SHA-256 Update time, fsync = file fsync + rename + directory fsync. Zero for other events.
  double phase_gather_ms{0.0}, phase_write_ms{0.0}, phase_hash_ms{0.0}, phase_fsync_ms{0.0};
  // reason == kPrefixPersisted only: length of the persisted prefix that covers the skipped snapshot and the tokens it does not cover (review F3).
  std::size_t covering_tokens{0}, gap_tokens{0};
};

struct ContinuationDiskStoreOptions {
  std::filesystem::path directory;
  std::size_t capacity_bytes{TextRunnerDiskCacheOptions::kDefaultCapacityBytes};
  /// Zero resolves to the host snapshot budget, capped by capacity_bytes.
  std::size_t staging_capacity_bytes{0};
  /// Minimum token advance over the longest stored prefix that is still a
  /// prefix of the checkpoint before writing it. Zero disables the gate.
  std::size_t min_checkpoint_step_tokens{0};
  /// Maximum additional idle-gate delay from enqueue when GUFO_DISK_DEFER=1.
  std::chrono::milliseconds defer_max_wait{std::chrono::seconds(10)};
  /// Unspecified values resolve from once-read environment settings.
  std::optional<bool> persist_evict_idle;
  std::optional<bool> slice;
  std::optional<std::chrono::milliseconds> idle_wait;
  /// Additional pinned evicted storage, independent of the staging budget.
  std::optional<std::size_t> evict_backlog_bytes;
  /// Evict-skip policy: skip persisting an evicted snapshot whose persisted strict prefix is at most this many tokens shorter (0 = off).
  std::optional<std::size_t> evict_skip_tokens;
  /// Absolute safety guard for tiny snapshots; bytes are the primary bound.
  std::size_t evict_backlog_count{8};
  std::chrono::milliseconds shutdown_wait{std::chrono::seconds(30)};
  std::chrono::milliseconds slice_max_wait{std::chrono::seconds(10)};
};

/// Restart-safe, provider-neutral exact-prefix snapshot store.
///
/// Model code owns compatibility identity and payload serialization. The store
/// owns checksums, exact token verification, byte limits, atomic publication,
/// startup indexing, LRU replacement, permissions, and collision-safe lookup.
///
/// Persistence uses one bounded worker. Payload writes and fsync run outside
/// the metadata gate, so existing entries remain readable during persistence.
class ContinuationDiskStore {
  struct Impl;

public:
  using EventSink = std::function<void(const ContinuationDiskEvent&)>;
  using KeyHashFunction =
      std::function<std::string(std::span<const std::uint8_t>)>;

  struct SaveResult {
    bool stored{false};
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
  };

  struct RestoreResult {
    bool restored{false};
    std::size_t token_count{0};
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
  };

  explicit ContinuationDiskStore(ContinuationDiskStoreOptions options,
                                 EventSink event_sink = {},
                                 KeyHashFunction key_hash = {});
  ~ContinuationDiskStore();

  ContinuationDiskStore(const ContinuationDiskStore&) = delete;
  ContinuationDiskStore& operator=(const ContinuationDiskStore&) = delete;
  ContinuationDiskStore(ContinuationDiskStore&&) = delete;
  ContinuationDiskStore& operator=(ContinuationDiskStore&&) = delete;

  [[nodiscard]] SaveResult Save(
      const TextModelRunner& runner,
      std::span<const TextRunnerToken> checkpoint_tokens,
      const TextRunnerSnapshot& snapshot,
      std::span<const std::uint8_t> input_identity = {});

  /// Holds disk-only capture memory against the staging budget until it is
  /// queued or discarded. Reservations can safely outlive the store.
  class CaptureReservation {
  public:
    ~CaptureReservation();
    CaptureReservation(const CaptureReservation&) = delete;
    CaptureReservation& operator=(const CaptureReservation&) = delete;

  private:
    friend class ContinuationDiskStore;
    CaptureReservation(std::shared_ptr<std::atomic<std::size_t>> counter,
                       std::size_t bytes)
        : counter_(std::move(counter)), bytes_(bytes) {}
    std::shared_ptr<std::atomic<std::size_t>> counter_;
    std::size_t bytes_;
  };

  [[nodiscard]] std::unique_ptr<CaptureReservation> ReserveCapture(
      const TextModelRunner& runner, std::size_t token_count,
      std::size_t snapshot_bytes,
      std::span<const std::uint8_t> input_identity = {});

  /// Conservative admission before capturing a snapshot. Includes queued and
  /// active snapshot bytes; failure means a harmless cache skip.
  [[nodiscard]] bool CanSave(
      const TextModelRunner& runner, std::size_t token_count,
      std::size_t snapshot_bytes,
      std::span<const std::uint8_t> input_identity = {}) const;

  /// Enqueues immutable storage, retaining both model and snapshot until done.
  /// The returned byte count is queued work, not completed disk I/O.
  [[nodiscard]] std::size_t SaveAsync(
      std::shared_ptr<const TextModelRunner> runner,
      std::vector<TextRunnerToken> checkpoint_tokens,
      std::shared_ptr<const TextRunnerSnapshot> snapshot,
      std::vector<std::uint8_t> input_identity = {},
      std::unique_ptr<CaptureReservation> reservation = {},
      bool shared_prefix = false, std::string_view trigger = "write_through");

  /// Capture offer: default delegates to SaveAsync; opt-in tags RAM storage.
  [[nodiscard]] std::size_t Capture(
      std::shared_ptr<const TextModelRunner> runner,
      std::vector<TextRunnerToken> tokens,
      std::shared_ptr<const TextRunnerSnapshot> snapshot,
      std::vector<std::uint8_t> identity = {},
      std::unique_ptr<CaptureReservation> reservation = {},
      bool shared_prefix = false, bool already_persisted = false,
      bool retained_in_ram = true);
  using PersistenceSource =
      std::function<std::shared_ptr<const ContinuationSnapshot>(
          bool include_failed)>;
  void SetPersistenceSource(std::shared_ptr<const TextModelRunner> runner,
                            PersistenceSource source);
  void PersistEvicted(std::shared_ptr<const ContinuationSnapshot> snapshot);
  /// Stops sourcing and joins the writer; deadline bounds new shutdown work.
  void Shutdown();
  [[nodiscard]] bool persist_on_evict_idle() const noexcept;
  /// evict_idle policy: the RAM tier refused an OPTIONAL copy (the whole-prompt retry checkpoint); it was never retained, so it is not an eviction and is not
  /// written. Logged as action=skipped reason=ram_not_retained and counted in `skipped` (admission trace 2026-10-09: these refused retry copies were the
  /// whole 9.6 GB of the D3F run and changed no restore).
  void NoteOptionalCopyRefused(std::size_t token_count) noexcept;

  /// Drains accepted writes. Shutdown also drains automatically.
  void Flush();

  /// Nonallocating, lock-free request-lifetime signal for the optional worker
  /// idle gate. The store must outlive the token. Disabled tokens are no-ops.
  class GenerationActivity {
  public:
    GenerationActivity() = default;
    ~GenerationActivity();
    GenerationActivity(const GenerationActivity&) = delete;
    GenerationActivity& operator=(const GenerationActivity&) = delete;

  private:
    friend class ContinuationDiskStore;
    explicit GenerationActivity(Impl* owner);
    Impl* owner_{nullptr};
  };
  [[nodiscard]] GenerationActivity BeginGeneration();
  [[nodiscard]] GenerationActivity BeginAdmission();

  /// Restores the longest exact saved prefix of prompt into state. When a
  /// stable boundary is supplied, a later checkpoint requires an earlier
  /// saved prefix too, so legacy full-prompt files cannot bypass migration.
  [[nodiscard]] RestoreResult RestoreLongestPrefix(
      const TextModelRunner& runner, TextRunnerState& state,
      std::span<const TextRunnerToken> prompt,
      std::span<const std::uint8_t> input_identity = {},
      std::size_t stable_prefix_tokens = 0,
      std::span<const ContinuationInputPrefix> input_prefixes = {});

  /// Prefix lengths that prompt shares with stored entries but that no entry
  /// holds exactly.
  ///
  /// For each stored entry of the same compatibility identity, the common
  /// prefix length with prompt is a candidate. Lengths below min_tokens, equal
  /// to prompt.size(), or already stored exactly are dropped. Returns at most
  /// max_boundaries distinct lengths in ascending order. Callers snapshot at
  /// those positions so a prefix shared across conversations (system prompt,
  /// tool schemas) becomes restorable by the next one instead of being
  /// prefilled again.
  [[nodiscard]] std::vector<std::size_t> SharedPrefixBoundaries(
      const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
      std::size_t min_tokens, std::size_t max_boundaries,
      std::span<const std::uint8_t> input_identity = {},
      std::span<const ContinuationInputPrefix> input_prefixes = {});

  /// Marks the exact entry for tokens as recently used without reading it.
  /// Returns false when no such entry exists.
  [[nodiscard]] bool Touch(const TextModelRunner& runner,
                           std::span<const TextRunnerToken> tokens,
                           std::span<const std::uint8_t> input_identity = {});

  /// True when a stored prefix lies less than min_checkpoint_step_tokens
  /// before tokens, so saving them as a continuation would be skipped. Lets
  /// callers avoid capturing a snapshot only to have it discarded.
  [[nodiscard]] bool WithinCheckpointStep(
      const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
      std::span<const std::uint8_t> input_identity = {});

  [[nodiscard]] std::size_t entry_count() const noexcept;
  [[nodiscard]] std::size_t retained_bytes() const noexcept;
  [[nodiscard]] std::size_t capacity_bytes() const noexcept;
  [[nodiscard]] std::size_t staging_capacity_bytes() const noexcept;
  [[nodiscard]] const ContinuationDiskStoreOptions& options() const noexcept;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CONTINUATION_DISK_STORE_HPP_
