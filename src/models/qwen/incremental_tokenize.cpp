#include "src/models/qwen/incremental_tokenize.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace gufo::tokenization {
namespace {

bool SameMessage(const ChatMessage& a, const ChatMessage& b) {
  if (a.role != b.role || a.content != b.content ||
      a.framing_suffix != b.framing_suffix || a.name != b.name ||
      a.thought != b.thought || a.tool_call_id != b.tool_call_id ||
      !a.images.empty() || !b.images.empty() ||
      a.tool_calls.size() != b.tool_calls.size())
    return false;
  for (std::size_t index = 0; index < a.tool_calls.size(); ++index) {
    const auto& x = a.tool_calls[index];
    const auto& y = b.tool_calls[index];
    if (x.id != y.id || x.name != y.name ||
        x.arguments.size() != y.arguments.size())
      return false;
    for (std::size_t arg = 0; arg < x.arguments.size(); ++arg) {
      const auto& p = x.arguments[arg];
      const auto& q = y.arguments[arg];
      if (p.name != q.name || p.value != q.value || p.is_string != q.is_string)
        return false;
    }
  }
  return true;
}

bool SameOptions(const ChatTemplateOptions& a, const ChatTemplateOptions& b) {
  return a.add_generation_prompt == b.add_generation_prompt &&
         a.enable_thinking == b.enable_thinking &&
         a.reasoning_effort == b.reasoning_effort &&
         a.preserve_thinking == b.preserve_thinking &&
         a.add_vision_id == b.add_vision_id &&
         a.require_tool_call == b.require_tool_call && a.dialect == b.dialect &&
         a.tool_format == b.tool_format &&
         a.max_output_bytes == b.max_output_bytes;
}

bool SameTools(std::span<const ChatTool> a, std::span<const ChatTool> b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t index = 0; index < a.size(); ++index) {
    if (a[index].name != b[index].name ||
        a[index].description != b[index].description ||
        a[index].parameters_json != b[index].parameters_json ||
        a[index].definition_json != b[index].definition_json)
      return false;
  }
  return true;
}

void HashSize(crypto::Sha256Hasher& hash, std::size_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>(value >> (8 * index));
  hash.Update(bytes);
}

void HashText(crypto::Sha256Hasher& hash, std::string_view text) {
  HashSize(hash, text.size());
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

std::array<std::uint8_t, 32> PromptKey(std::span<const ChatMessage> messages,
                                       std::span<const ChatTool> tools,
                                       const ChatTemplateOptions& options,
                                       std::uint32_t max_context) {
  crypto::Sha256Hasher hash;
  if (!messages.empty() && messages.front().role == ChatRole::kSystem) {
    HashText(hash, messages.front().content);
    HashText(hash, messages.front().framing_suffix);
  }
  for (const auto value :
       {std::size_t(options.add_generation_prompt),
        std::size_t(options.enable_thinking),
        std::size_t(options.reasoning_effort),
        std::size_t(options.preserve_thinking),
        std::size_t(options.add_vision_id),
        std::size_t(options.require_tool_call), std::size_t(options.dialect),
        std::size_t(options.tool_format), options.max_output_bytes,
        std::size_t(max_context), tools.size()})
    HashSize(hash, value);
  for (const auto& tool : tools) {
    HashText(hash, tool.name);
    HashText(hash, tool.description);
    HashText(hash, tool.parameters_json);
    HashText(hash, tool.definition_json);
  }
  return hash.Finish();
}

bool SafeSpecials(const QwenTokenizer& tokenizer) {
  const auto start = tokenizer.FindSpecialToken(kImStart);
  const auto end = tokenizer.FindSpecialToken(kImEnd);
  if (!start || !end || *start == *end ||
      tokenizer.Encode(kImStart) != std::vector<TokenId>{*start} ||
      tokenizer.Encode(kImEnd) != std::vector<TokenId>{*end})
    return false;
  // Encode scans specials before NFC, pre-tokenization and BPE. Single bracket
  // pairs cannot overlap ChatML framing or extend a match across its newline.
  for (const auto& [text, id] : tokenizer.SpecialTokens()) {
    static_cast<void>(id);
    if (text.size() < 2 || text.front() != '<' || text.back() != '>' ||
        text.find('<', 1) != std::string::npos ||
        text.find('>') != text.size() - 1 ||
        text.find_first_of("\r\n") != std::string::npos)
      return false;
  }
  return true;
}

std::string_view Eligibility(std::span<const ChatMessage> messages,
                             const ChatTemplateOptions& options,
                             std::size_t& input_bytes) {
  if (messages.empty())
    return "invalid_request";
  if (options.dialect != QwenTemplateDialect::kQwen)
    return "dialect";
  if (!options.preserve_thinking)
    return "reasoning_history";
  bool assistant = false;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto& message = messages[index];
    if (!message.images.empty())
      return "images";
    if (message.role == ChatRole::kDeveloper ||
        (message.role == ChatRole::kSystem && index != 0))
      return "system_layout";
    assistant |= message.role == ChatRole::kAssistant;
    input_bytes += sizeof(ChatMessage) + message.content.size() +
                   message.framing_suffix.size() + message.name.size() +
                   message.thought.size() + message.tool_call_id.size();
    for (const auto& call : message.tool_calls) {
      input_bytes += sizeof(call) + call.id.size() + call.name.size();
      for (const auto& arg : call.arguments)
        input_bytes += sizeof(arg) + arg.name.size() + arg.value.size();
    }
  }
  if (assistant && messages.size() >= 2 &&
      messages.back().role == ChatRole::kUser &&
      (messages[messages.size() - 2].role == ChatRole::kUser ||
       messages[messages.size() - 2].role == ChatRole::kTool))
    return "replaceable_user";
  return {};
}

// Match Render's early bound as well as its actual rendered-byte bound. A
// whitespace-only message can fail the former even though Trim removes it.
bool EstimateFits(std::span<const ChatMessage> messages,
                  std::span<const ChatTool> tools,
                  const ChatTemplateOptions& options) {
  std::size_t estimate = 0;
  const auto add = [&](std::size_t bytes) {
    if (bytes > options.max_output_bytes - estimate)
      return false;
    estimate += bytes;
    return true;
  };
  for (const auto& message : messages) {
    if (!add(message.content.size()) || !add(message.framing_suffix.size()) ||
        !add(message.thought.size()) || !add(32))
      return false;
    for (const auto& call : message.tool_calls) {
      if (!add(call.name.size()) || !add(64))
        return false;
      for (const auto& arg : call.arguments)
        if (!add(arg.name.size()) || !add(arg.value.size()) || !add(40))
          return false;
    }
  }
  for (const auto& tool : tools)
    if (!add(tool.name.size()) || !add(tool.description.size()) ||
        !add(tool.parameters_json.size()) ||
        !add(tool.definition_json.size()) ||
        !add(tool.definition_json.size()) || !add(96))
      return false;
  // This upper bound deliberately sends near-bound requests to Render.
  return (!options.add_generation_prompt || add(64)) && add(1024) &&
         estimate <= std::size_t(std::numeric_limits<std::int32_t>::max());
}
}  // namespace

struct QwenIncrementalTokenizer::Entry {
  std::string client;
  std::array<std::uint8_t, 32> key;
  const QwenTokenizer* tokenizer{nullptr};
  std::vector<ChatMessage> messages;
  std::vector<ChatTool> tools;
  ChatTemplateOptions options;
  std::uint32_t max_context{0};
  std::string rendered;
  std::vector<TokenId> tokens;
  std::vector<TokenId> generation_tokens;
  std::size_t history_bytes{0};
  std::size_t history_tokens{0};
  Clock::time_point created;
  std::uint64_t ticket{0};
};

void QwenIncrementalTokenizer::RecordExternalFull(std::string_view reason,
                                                  std::size_t tokenized) const {
  std::size_t count = 0;
  {
    const std::lock_guard lock(mutex_);
    count = ++full_[std::string(reason)];
    tokenized_ += tokenized;
  }
  if (log_ && (count == 1 || count % 64 == 0))
    log_("incremental_tokenize full reason=" + std::string(reason) +
         " count=" + std::to_string(count));
}

void QwenIncrementalTokenizer::AddTokenized(std::size_t count) const {
  const std::lock_guard lock(mutex_);
  tokenized_ += count;
}

void QwenIncrementalTokenizer::Publish(
    std::shared_ptr<const Entry> entry) const {
  const std::lock_guard lock(mutex_);
  std::size_t slot = 0;
  for (; slot < entries_.size(); ++slot)
    if (!entries_[slot] || entries_[slot]->client == entry->client) {
      // Only a newer retained history for this client supersedes its work.
      // Independent clients can publish out of order, including to free slots.
      if (entries_[slot] && entry->ticket < entries_[slot]->ticket) {
        ++publication_skips_;
        return;
      }
      break;
    }
  if (slot == entries_.size()) {
    slot = std::min_element(used_.begin(), used_.end()) - used_.begin();
    ++evictions_;
  }
  used_[slot] = entry->ticket;
  entries_[slot] = std::move(entry);
}

IncrementalPrompt QwenIncrementalTokenizer::Prepare(
    const QwenTokenizer& tokenizer, std::string_view client,
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::uint32_t max_context,
    std::string_view request_guard, Clock::time_point now) const {
  if (client.empty())
    client = "anonymous";
  std::shared_ptr<const Entry> previous;
  std::uint64_t ticket = 0;
  {
    const std::lock_guard lock(mutex_);
    ticket = ++serial_;
    for (std::size_t slot = 0; slot < entries_.size(); ++slot)
      if (entries_[slot] && entries_[slot]->client == client) {
        previous = entries_[slot];
        used_[slot] = ticket;
        break;
      }
  }
  std::size_t input_bytes = client.size();
  auto guard = Eligibility(messages, options, input_bytes);
  if (!request_guard.empty())
    guard = request_guard;
  for (const auto& tool : tools)
    input_bytes += sizeof(tool) + tool.name.size() + tool.description.size() +
                   tool.parameters_json.size() + tool.definition_json.size();
  if (guard.empty() && input_bytes > kMaxEntryBytes / 8)
    guard = "memory_bound";
  if (guard.empty() && !EstimateFits(messages, tools, options))
    guard = "render_bound";
  if (guard.empty() && !SafeSpecials(tokenizer))
    guard = "unsafe_boundary";
  const auto key = PromptKey(messages, tools, options, max_context);
  auto reason = guard;
  if (reason.empty()) {
    if (!previous)
      reason = "no_entry";
    else if (now - previous->created > kLifetime)
      reason = "expired";
    else if (previous->tokenizer != &tokenizer)
      reason = "tokenizer_changed";
    else if (previous->messages.front().role != messages.front().role ||
             (messages.front().role == ChatRole::kSystem &&
              !SameMessage(previous->messages.front(), messages.front())))
      reason = "system_changed";
    else if (previous->key != key || !SameOptions(previous->options, options) ||
             previous->max_context != max_context ||
             !SameTools(previous->tools, tools))
      reason = "options_changed";
    else {
      std::size_t common = 0;
      while (common < std::min(previous->messages.size(), messages.size()) &&
             SameMessage(previous->messages[common], messages[common]))
        ++common;
      if (common < previous->messages.size())
        reason = common == 0 ? "first_message" : "history_changed";
      else if (common == messages.size())
        reason = "not_append";
      else if (previous->messages.back().role == ChatRole::kTool &&
               messages[common].role == ChatRole::kTool)
        reason = "tool_group_changed";
    }
  }

  std::string rendered;
  std::vector<ContentSpan> spans;
  std::size_t stable_bytes = 0;
  IncrementalPrompt result;
  if (reason.empty()) {
    std::string error;
    auto tail = QwenChatTemplate::RenderTail(
        messages.subspan(previous->messages.size()), options, &error, &spans);
    const auto history =
        std::string_view(previous->rendered).substr(0, previous->history_bytes);
    const std::string ending = std::string(kImEnd) + "\n";
    if (!tail || !history.ends_with(ending) || !tail->starts_with(kImStart)) {
      reason = "unsafe_boundary";
    } else {
      auto tokens = QwenChatTemplate::EncodeRendered(tokenizer, *tail, 0,
                                                     tail->size(), spans, {});
      AddTokenized(tokens.size());
      const auto& generation = previous->generation_tokens;
      if (tokens.size() < generation.size() ||
          !std::equal(generation.rbegin(), generation.rend(),
                      tokens.rbegin())) {
        reason = "unsafe_boundary";
      } else if (previous->history_tokens + tokens.size() > max_context) {
        reason = "context_limit";
      } else {
        result.tokenized_tokens = tokens.size();
        result.reused_tokens = previous->history_tokens;
        result.tokens.reserve(result.reused_tokens + tokens.size());
        result.tokens.insert(result.tokens.end(), previous->tokens.begin(),
                             previous->tokens.begin() + result.reused_tokens);
        result.tokens.insert(result.tokens.end(), tokens.begin(), tokens.end());
        result.cache_prefix_tokens = result.tokens.size() - generation.size();
        rendered.reserve(history.size() + tail->size());
        rendered.append(history).append(*tail);
        stable_bytes = rendered.size() -
                       (options.add_generation_prompt
                            ? GenerationPrompt(options.enable_thinking).size()
                            : 0);
        if (rendered.size() > options.max_output_bytes) {
          reason = "render_bound";
        } else {
          result.incremental = true;
          result.reason = "hit";
          const std::lock_guard lock(mutex_);
          ++hits_;
          reused_ += result.reused_tokens;
        }
      }
    }
  }
  if (!result.incremental) {
    RecordExternalFull(reason);
    result = {};
    result.reason = reason;
    if (reason == "images") {
      // Preserve the existing text-only preparation's image error exactly.
      auto prompt = models::qwen::vision::Prepare(tokenizer, messages, tools,
                                                  options, {}, max_context);
      result.tokens = std::move(prompt.tokens);
      result.cache_prefix_tokens = prompt.stable_prefix_tokens;
      return result;
    }
    std::string error;
    auto full = QwenChatTemplate::Render(messages, tools, options, &error,
                                         nullptr, &stable_bytes, &spans);
    if (!full)
      throw std::invalid_argument(error);
    rendered = std::move(*full);
    result.tokens = QwenChatTemplate::EncodeRendered(
        tokenizer, rendered, 0, rendered.size(), spans, {});
    AddTokenized(result.tokens.size());
    if (result.tokens.size() > max_context)
      throw std::length_error(
          "prompt exceeds the " + std::to_string(max_context) +
          "-token context; increase --context or shorten the conversation");
    auto suffix = QwenChatTemplate::EncodeRendered(
        tokenizer, rendered, stable_bytes, rendered.size(), spans, {});
    AddTokenized(suffix.size());
    if (suffix.size() > result.tokens.size() ||
        !std::equal(suffix.rbegin(), suffix.rend(), result.tokens.rbegin()))
      throw std::logic_error("Qwen stable prefix is not a token boundary");
    result.cache_prefix_tokens = result.tokens.size() - suffix.size();
    result.tokenized_tokens = result.tokens.size() + suffix.size();
  }

  const auto generation_bytes = options.add_generation_prompt
                                    ? GenerationPrompt(options.enable_thinking)
                                    : std::string_view{};
  const auto retained_bytes =
      input_bytes + rendered.size() + result.tokens.size() * sizeof(TokenId);
  // Unsafe or oversized requests never become reusable entries. Retain the
  // previous immutable input; later requests still have to match it exactly.
  if (guard.empty() && retained_bytes <= kMaxEntryBytes &&
      rendered.ends_with(generation_bytes) &&
      stable_bytes == rendered.size() - generation_bytes.size()) {
    auto entry = std::make_shared<Entry>();
    entry->client = client;
    entry->key = key;
    entry->tokenizer = &tokenizer;
    entry->messages.assign(messages.begin(), messages.end());
    entry->tools.assign(tools.begin(), tools.end());
    entry->options = options;
    entry->max_context = max_context;
    entry->history_bytes = stable_bytes;
    entry->history_tokens = result.cache_prefix_tokens;
    entry->generation_tokens.assign(
        result.tokens.begin() + result.cache_prefix_tokens,
        result.tokens.end());
    entry->rendered = std::move(rendered);
    entry->tokens = result.tokens;
    entry->created = now;
    entry->ticket = ticket;
    Publish(std::move(entry));
  }
  return result;
}

json::Value QwenIncrementalTokenizer::Statistics() const {
  const std::lock_guard lock(mutex_);
  auto stats = json::Value::object();
  auto full = json::Value::object();
  std::size_t full_count = 0;
  for (const auto& [reason, count] : full_) {
    full[reason] = count;
    full_count += count;
  }
  stats["full_path_count"] = full_count;
  stats["full_path_reasons"] = std::move(full);
  stats["incremental_hits"] = hits_;
  stats["tokens_reused"] = reused_;
  stats["tokens_tokenized"] = tokenized_;
  stats["entries"] =
      std::count_if(entries_.begin(), entries_.end(),
                    [](const auto& entry) { return bool(entry); });
  stats["max_entries"] = kMaxEntries;
  stats["max_entry_bytes"] = kMaxEntryBytes;
  stats["lifetime_seconds"] = std::chrono::seconds(kLifetime).count();
  stats["evictions"] = evictions_;
  stats["publication_skips"] = publication_skips_;
  return stats;
}
}  // namespace gufo::tokenization
