#ifndef GUFO_TOKENIZATION_QWEN_INCREMENTAL_TOKENIZE_HPP_
#define GUFO_TOKENIZATION_QWEN_INCREMENTAL_TOKENIZE_HPP_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::tokenization {

struct IncrementalPrompt {
  std::vector<TokenId> tokens;
  std::size_t cache_prefix_tokens{0};
  bool incremental{false};
  std::string_view reason;
  std::size_t reused_tokens{0};
  std::size_t tokenized_tokens{0};
};

/// Optional host-only cache. Owners construct it only when the switch is on.
/// Published entries hold prepared inputs, never generated tokens or KV state.
class QwenIncrementalTokenizer {
public:
  using Clock = std::chrono::steady_clock;
  using Log = std::function<void(std::string_view)>;
  explicit QwenIncrementalTokenizer(Log log = {}) : log_(std::move(log)) {}

  [[nodiscard]] IncrementalPrompt Prepare(
      const QwenTokenizer& tokenizer, std::string_view client,
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options, std::uint32_t max_context,
      std::string_view request_guard = {},
      Clock::time_point now = Clock::now()) const;

  /// Model-owned image preparation and token-count requests keep their full
  /// paths. Include them in the same counters and rate-limited fallback log.
  void RecordExternalFull(std::string_view reason,
                          std::size_t tokenized = 0) const;
  void RecordExternalTokens(std::size_t tokenized) const {
    AddTokenized(tokenized);
  }
  [[nodiscard]] json::Value Statistics() const;

private:
  struct Entry;
  static constexpr std::size_t kMaxEntries = 4;
  static constexpr std::size_t kMaxEntryBytes = 16 * 1024 * 1024;
  static constexpr auto kLifetime = std::chrono::minutes(10);

  void Publish(std::shared_ptr<const Entry> entry) const;
  void AddTokenized(std::size_t count) const;

  Log log_;
  mutable std::mutex mutex_;
  mutable std::array<std::shared_ptr<const Entry>, kMaxEntries> entries_;
  mutable std::array<std::uint64_t, kMaxEntries> used_{};
  mutable std::uint64_t serial_{0};
  mutable std::map<std::string, std::size_t> full_;
  mutable std::size_t hits_{0};
  mutable std::size_t reused_{0};
  mutable std::size_t tokenized_{0};
  mutable std::size_t evictions_{0};
  mutable std::size_t publication_skips_{0};
};

}  // namespace gufo::tokenization
#endif  // GUFO_TOKENIZATION_QWEN_INCREMENTAL_TOKENIZE_HPP_
