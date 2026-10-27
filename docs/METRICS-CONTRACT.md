# Status and Prometheus audit

Version 1 uses the existing `/metrics` endpoint. Both readers use the
`detail` counters in `src/cli/serve/generation_metrics.hpp`. The export is in
`LlamaMetrics` in `src/cli/serve/http_server.cpp`; typed status is in
`src/cli/serve/status_metrics.hpp`. No new endpoint is added.

| Typed status | Prometheus name | Same source |
| --- | --- | --- |
| prefill_tokens_total | llamacpp:prompt_tokens_total | TotalPromptTokens |
| decode_tokens_total | llamacpp:tokens_predicted_total | TotalGenTokens |
| cached_tokens_total | llamacpp:prompt_tokens_cached_total | TotalCachedPromptTokens |
| prefill_seconds_total | llamacpp:prompt_seconds_total | TotalPromptSeconds |
| decode_seconds_total | llamacpp:tokens_predicted_seconds_total | TotalGenSeconds |
| max_sequence_tokens | llamacpp:n_tokens_max | MaxSequenceTokens |
| draft_rounds_total | llamacpp:spec_decode_num_drafts_total | TotalDraftRounds |
| draft_tokens_total | llamacpp:spec_decode_num_draft_tokens_total | TotalDraftTokens |
| draft_accepted_tokens_total | llamacpp:spec_decode_num_accepted_tokens_total | TotalDraftAcceptedTokens |
| device_lost_total | gufo_device_lost_total | DeviceLostTotal |
| sessions | llamacpp:requests_processing | RequestsProcessing |
| queue | llamacpp:requests_deferred | RequestsDeferred |

The following values shown by typed status have no matching `/metrics`
series in version 1:

- cache_disk_hits_total, cache_memory_hits_total, cache_misses_total,
  completed_successful_requests_total and cache_hit_ratio.
- Live aggregate wall throughput. Prometheus can derive window rates from
  the shared token counters; its existing speed gauges are per-request
  observations, not the CLI watch interval.
- Nullable last-completed speeds and age. The legacy speed gauges retain
  the last positive speed, so they do not carry the same null/age semantics.
- The 50-row request ring: request id, time, slot, prompt/completion/cache
  token counts, TTFT, prefill/decode rates, cache kind, finish reason and
  outcome. Per-request values are not given high-cardinality metric labels.
- Per-slot context, state, current request id and active tokens used. `/slots`
  exposes their underlying session counters. The existing aggregate
  kv_cache_usage_ratio includes in-flight prompt and generated tokens; it
  excludes retained cache and differs from processed-plus-cached tokens used.
- Process RSS, host-available memory and all planning memory components,
  ranges, budgets, units and evidence labels.

These are explicit export gaps. This job does not add series or claim that
Grafana already shows all control-panel values. GPU power is omitted here;
the owning hardware job must supply and name its ADLX source before display.
