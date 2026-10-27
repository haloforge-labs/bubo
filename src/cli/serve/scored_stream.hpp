#ifndef GUFO_SERVER_SCORED_STREAM_HPP_
#define GUFO_SERVER_SCORED_STREAM_HPP_

#include "src/cli/serve/debug_response.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/logprobs.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {

// Scored requests alone use this transport. A callback is one published token,
// including empty UTF-8 fragments, and carries its immutable pre-draw score.
inline HttpResponse ScoredOpenAiStream(
    std::shared_ptr<TextGenerationBackend::GenerationRequest> generation,
    TextGenerationBackend& backend, std::string id, long long created,
    bool chat, bool include_usage,
    std::function<json::Value(const TextGenerationBackend::Result&)> usage,
    std::function<void(json::Value&, const TextGenerationBackend::Result&)>
        decorate_usage = {}) {
  auto log = std::make_shared<HttpResponse::StreamLog>();
  return {
      .status = 200,
      .reason = "OK",
      .body = {},
      .headers = {{"Content-Type", "text/event-stream"},
                  {"Cache-Control", "no-cache"}},
      .streaming_body =
          [generation, &backend, id = std::move(id), created, chat,
           include_usage, usage = std::move(usage),
           decorate_usage = std::move(decorate_usage),
           log](const HttpResponse::BodyWriter& writer) {
            bool started = false;
            std::size_t count = 0, offset = 0;
            core::Utf8Decoder decoder;
            const LogprobTokenText decode =
                [&backend](sampling::TokenId token) {
                  return backend.token_piece(token);
                };
            const auto emit = [&](std::string_view text,
                                  const sampling::TokenLogprob* score,
                                  std::string_view finish,
                                  const TextGenerationBackend::Result* result =
                                      nullptr,
                                  bool usage_only = false) {
              auto chunk = json::Value::object();
              chunk["id"] = id;
              chunk["created"] = created;
              chunk["model"] = backend.model_id();
              chunk["object"] =
                  chat ? "chat.completion.chunk" : "text_completion";
              chunk["choices"] = json::Value::array();
              if (!usage_only) {
                auto choice = json::Value::object();
                choice["index"] = 0;
                choice["finish_reason"] =
                    finish.empty() ? json::Value()
                                   : json::Value(std::string(finish));
                if (chat) {
                  choice["delta"] = json::Value::object();
                  if (!started)
                    choice["delta"]["role"] = "assistant";
                  choice["delta"]["content"] = std::string(text);
                } else {
                  choice["text"] = std::string(text);
                }
                choice["logprobs"] = json::Value();
                if (score) {
                  const auto records =
                      std::span<const sampling::TokenLogprob>(score, 1);
                  choice["logprobs"] = chat ? ChatLogprobs(records, decode)
                                            : LegacyLogprobs(records, decode);
                  if (!chat) {
                    // Not `["text_offset"][0]`: json::Value::operator[] takes a std::string, so a literal 0 is a null const char* (crash in strlen).
                    choice["logprobs"]["text_offset"] = json::Value::array();
                    choice["logprobs"]["text_offset"].push_back(offset);
                  }
                }
                chunk["choices"].push_back(std::move(choice));
              }
              if (result) {
                if (usage_only) {
                  chunk["usage"] = usage(*result);
                  if (decorate_usage)
                    decorate_usage(chunk, *result);
                } else {
                  chunk["timings"] = GenerationTimings(*result);
                  AddDebugTimeline(chunk, *result);
                }
              }
              started = true;
              return writer("data: " + chunk.dump() + "\n\n");
            };
            try {
              const auto result = generation->WaitScored(
                  [&](std::string_view piece,
                      const sampling::TokenLogprob& score) {
                    if (piece != decode(score.token))
                      throw std::logic_error(
                          "stream token text and logprob IDs differ");
                    const auto text = decoder.Push(piece, false);
                    if (!emit(text, &score, {}))
                      return false;
                    ++count;
                    for (const unsigned char c : text)
                      if ((c & 0xC0) != 0x80)
                        ++offset;
                    return true;
                  });
              log->details = GenerationLogDetails(result);
              RecordServerMetrics(result);
              if (result.cancelled)
                return;
              ValidateGeneratedLogprobs(result);
              if (count != result.tokens.size())
                throw std::logic_error(
                    "stream token and logprob counts differ");
              if (!emit(decoder.Push({}, true), nullptr,
                        result.finish_reason ==
                                TextGenerationBackend::FinishReason::kLength
                            ? "length"
                            : "stop",
                        &result))
                return;
              if (include_usage && !emit({}, nullptr, {}, &result, true))
                return;
              (void)writer("data: [DONE]\n\n");
            } catch (const std::exception& error) {
              generation->Cancel();
              if (!started)
                throw;
              log->error_code = "generation_failed";
              auto event = json::Value::object();
              event["error"]["message"] = error.what();
              event["error"]["type"] = "server_error";
              event["error"]["code"] = "generation_failed";
              (void)emit({}, nullptr, "error");
              log->error_event_sent = writer("data: " + event.dump() + "\n\n");
              (void)writer("data: [DONE]\n\n");
            }
          },
      .stream_log = std::move(log),
      .defer_stream_headers = true};
}

}  // namespace gufo::server
#endif
