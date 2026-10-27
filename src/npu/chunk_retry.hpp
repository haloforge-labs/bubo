// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_CHUNK_RETRY_HPP_
#define GUFO_NPU_CHUNK_RETRY_HPP_
// L-REL-2: a prefill chunk whose NPU result is invalid is run once again on
// the GPU. This header holds the pure parts: the retry decision, the log text
// and the internal failure-injection switch GUFO_NPU_TEST_FAIL_CHUNK.
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

namespace gufo::npu {
// Back up the state before a chunk only when the NPU may take it: the lanes are
// ready, the pass qualifies for a lane (QualifiedChunk) and no hold will still
// be on when it starts. A chunk the guard re-arms at its start still gets one.
inline bool ShouldBackupChunk(bool lanes_ready, bool qualified,
                              bool hold_stays) noexcept {
  return lanes_ready && qualified && !hold_stays;
}

// ServerPrefill::ChunkUsed() keeps the last chunk's value when a pass fails
// before its chunk start (BeginChunk not reached): it counts only if the
// chunk serial moved during this attempt.
inline bool ChunkUsedThisAttempt(std::uint64_t serial_before,
                                 std::uint64_t serial_after,
                                 bool chunk_used) noexcept {
  return serial_after != serial_before && chunk_used;
}

// The backup time belongs to this attempt's chunk only if exactly one chunk
// was noted for the request during it (NoteRequestChunk ran once). A pass that
// failed before its check noted none, and its time would land in the slot of
// the chunk before.
inline bool BackupBelongsToChunk(std::uint64_t noted_before,
                                 std::uint64_t noted_after) noexcept {
  return noted_after == noted_before + 1;
}

// Retry iff the process-failed flag flipped during this chunk, the chunk used
// the NPU, and this is the first attempt for the span.
inline bool ShouldRetryChunk(bool failed_before, bool failed_after,
                             bool chunk_used, bool first_attempt) noexcept {
  return first_attempt && chunk_used && !failed_before && failed_after;
}

// chunk is the NPU chunk number inside the request (1-based, as in the
// failure line); a negative copy_ms means the copy time was not measured.
inline std::string FormatChunkRetryWarning(std::uint64_t chunk, double copy_ms,
                                           double retry_ms) {
  char copy[32];
  if (copy_ms < 0)
    std::snprintf(copy, sizeof(copy), "na");
  else
    std::snprintf(copy, sizeof(copy), "%.2f", copy_ms);
  char text[160];
  std::snprintf(text, sizeof(text),
                "chunk %llu faulted; retried on GPU (copy_ms=%s retry_ms=%.1f)",
                static_cast<unsigned long long>(chunk), copy, retry_ms);
  return text;
}

// GUFO_NPU_TEST_FAIL_CHUNK=k (internal, honoured only with debug hooks): NPU
// chunk k of a request, counted from 0, reports invalid activations through
// the normal failure path.
inline bool ParseTestFailChunk(std::string_view text, std::uint64_t* chunk,
                               std::string* error) {
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size() || value > 1000000) {
    if (error)
      *error = "expected a chunk number from 0 to 1000000";
    return false;
  }
  *chunk = value;
  return true;
}
// chunk_number is 1-based (Stats().chunks), the configured chunk 0-based.
inline bool TestFailChunkHits(const std::optional<std::uint64_t>& configured,
                              std::uint64_t chunk_number) noexcept {
  return configured && chunk_number == *configured + 1;
}
// Without debug hooks the switch is dropped before it reaches the knobs; the
// returned text is the short INFO line saying so.
inline std::optional<std::string> TestFailChunkIgnoredNote(bool debug_hooks) {
  if (debug_hooks)
    return std::nullopt;
  return "GUFO_NPU_TEST_FAIL_CHUNK ignored: it needs debug hooks";
}
inline std::string TestFailChunkActiveText(std::uint64_t chunk) {
  return "npu test: GUFO_NPU_TEST_FAIL_CHUNK override active (internal test "
         "setting, not for normal use): NPU chunk " +
         std::to_string(chunk) +
         " of every request (counted from 0) will report invalid activations";
}
}  // namespace gufo::npu
#endif
