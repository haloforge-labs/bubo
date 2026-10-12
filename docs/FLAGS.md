# Release switches

Generated from `src/cli/serve/release_switches.json`.

Profile defaults apply to Qwen3.8 Flash Next.
Other model families keep their existing defaults except for the exact seed and NPU profile policy.
Explicit flags and real environment values take precedence.
Empty environment values count as unset and produce a warning.

## Options and defaults

| Name | What it does | Default | When to change | Platform and hardware |
| --- | --- | --- | --- | --- |
| `--allow-unmeasured-memory` | Allow subagent context above the measured token limit. | Unset | Keep it off unless the required context exceeds the measured memory limit. | any; generic; any |
| `--api-key` | Require an API key from clients. | Unset | Set a key when clients must authenticate to the server. | any; generic; any |
| `--cache-disk` | Set the disk prompt cache directory. | Off | Opt in for fast restarts; model bytes are verified once, then a stamp skips later hashes. | any; generic; any |
| `--cache-disk-bytes` | Limit disk prompt cache storage. | 8,589,934,592 | Keep 8 GiB unless the disk cache must retain more prompt snapshots. | any; generic; any |
| `--cache-disk-staging-bytes` | Limit RAM used to stage disk prompt snapshots. | 0 = auto (<= disk budget, <= 1/8 available RAM) | Keep automatic sizing unless disk persistence needs a different RAM limit. | any; generic; any |
| `--cache-ram-bytes` | Set the RAM snapshot cache budget. | 6442450944 | Keep the profile limit unless available host RAM requires a smaller cache. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--context` | Set the context token limit for each session. | 262144 | Keep the profile context unless the task requires a different token limit. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--draft-head-model` | Select the prepared MTP draft head. | Unset | Select the prepared MTP draft head. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--draft-tokens` | Set the maximum speculative draft length. | 7 | Keep seven tokens unless speculative draft length tests require another value. | any; generic; qwen3.8-flash-next |
| `--exact` | Select the exact profile. | Unset | Enable it when output must repeat with the same build, prompt, and seed. | any; generic; any |
| `--frequency-penalty` | Penalize tokens by their generated occurrence count. | 0.0 (off) | Keep zero unless the task needs fewer frequently repeated tokens. | any; generic; any |
| `--host` | Set the server listen address. | 127.0.0.1 | Keep the local address unless remote clients must connect to the server. | any; generic; any |
| `--log-level` | Set the amount of server log detail. | info | Keep info unless diagnosis requires debug messages or normal operation needs fewer messages. | any; generic; any |
| `--log-progress` | Log prompt processing progress. | off | Keep it off unless prompt processing progress must appear in the log. | any; generic; any |
| `--max-buffered-output-bytes` | Limit buffered output for each request. | 65536 (64 KiB) | Keep 64 KiB unless slow clients require more buffered output per request. | any; generic; any |
| `--max-buffered-output-total` | Limit buffered output across all requests. | 262144 (256 KiB) | Keep 256 KiB unless concurrent slow clients require more total buffered output. | any; generic; any |
| `--max-connections` | Limit simultaneous HTTP connections. | 16 | Keep 16 unless the client workload requires more simultaneous HTTP connections. | any; generic; any |
| `--max-output-bytes` | Limit output text bytes for each request. | 1048576 (1 MiB) | Keep 1 MiB unless a request must return more output text. | any; generic; any |
| `--max-pending` | Limit queued requests. | 16 | Keep 16 unless the workload requires a larger request queue. | any; generic; any |
| `--max-pending-per-client` | Limit queued requests from each client. | Same as --max-pending | Keep the queue limit unless one client must have a smaller request allowance. | any; generic; any |
| `--max-request-bytes` | Set the maximum HTTP request body size. | 67108864 | Keep 64 MiB unless larger request bodies are required. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--max-tokens` | Limit new tokens generated per request. | -1 | Keep unlimited generation unless responses need a fixed new token limit. | any; generic; any |
| `--min-draft-tokens` | Set the minimum speculative draft length. | 1 | Keep one token unless speculative draft tests require a larger minimum. | any; generic; qwen3.8-flash-next |
| `--min-keep` | Retain a minimum number of sampling candidates. | 0 | Keep zero unless probability filters must retain a minimum number of candidates. | any; generic; any |
| `--min-p` | Filter tokens by probability relative to the best token. | 0.0 | Keep zero unless the task needs minimum probability filtering. | any; generic; any |
| `--mmproj` | Set the vision sidecar file. | Unset | Set the matching vision sidecar when the model must process images. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-draft-vocab` | Select the MTP draft head vocabulary. | auto with MTP; otherwise unused | Keep the profile vocabulary unless draft vocabulary tests require another setting. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-model` | Set the MTP sidecar file. | Unset | Set the matching MTP sidecar when speculative decoding is required. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-policy` | Select the MTP draft length controller. | survival with MTP; otherwise unused | Keep survival unless draft acceptance tests require the length policy. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--no-behaviour` | Disable the eight behaviour defaults. | Unset | Keep policies enabled unless plain sampling is required outside the exact profile. | any; generic; any |
| `--no-cache` | Disable RAM and disk prompt caches for all requests. | Off; exact enables it | Enable it when every request must prefill its full prompt for reproducible output. | any; generic; qwen3.8-flash-next |
| `--no-cache-disk` | Disable the disk snapshot cache. | Unset | Enable it when the supplied disk cache directory must be ignored. | any; generic; any |
| `--no-draft-head` | Suppress the manifest draft output head. | Unset | Enable it when MTP must run without the supplied draft head. | any; generic; any |
| `--no-mtp` | Disable MTP defaults and the speculative backend. | Unset | Enable it when decoding must run without MTP. | any; generic; any |
| `--no-prompt-lookup` | Disable prompt lookup proposals. | Unset | Keep prompt lookup enabled with MTP unless lookup comparisons require it off. | any; generic; any |
| `--no-twin` | Suppress the manifest dense twin. | Unset | Enable it when decoding must run without the supplied dense twin. | any; generic; any |
| `--no-vision` | Suppress the manifest vision sidecar. | Unset | Enable it when the server must omit the vision sidecar. | any; generic; any |
| `--npu-prefill` | Use NPU prefill when available. | on when usable | Default and subagents request ON; exact and stock force OFF; startup failures keep the GPU path. | any; generic; any |
| `--port` | Set the server listen port. | 8080 | Keep 8080 unless another service uses that port or clients require a different port. | any; generic; any |
| `--prefill-chunk` | Limit prompt tokens between active decode rounds. | 512 | Keep 512 unless active sessions need shorter pauses during another request prefill. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--presence-penalty` | Penalize tokens already generated in the response. | Qwen3.8: 0.0 thinking; 1.5 non-thinking unless GUFO_PRODUCTION_SAMPLING=1 | Keep the model preset unless the task needs a different penalty for previously generated tokens. | any; generic; any |
| `--preserve-thinking` | Select which thinking history to retain. | auto | Keep auto unless the client requires a different policy for retained thinking history. | any; generic; qwen3.8-flash-next |
| `--print-effective-settings` | Print resolved settings as JSON and exit. | Unset | Enable it when resolved settings must be inspected without loading a model. | any; generic; any |
| `--profile` | Select the release settings profile. | default | Keep default for normal use, select subagents for concurrency, or select exact for reproducibility. | any; generic; any |
| `--prompt-lookup` | Propose tokens from an earlier conversation match. | On with MTP; otherwise off | Keep it enabled with MTP unless prompt lookup comparisons require it off. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--reasoning-effort` | Set the default reasoning effort. | medium | Keep medium unless the task requires a different reasoning effort. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--repeat-last-n` | Set the recent token window for repetition penalties. | 64 | Keep 64 unless repetition penalties need a different recent token window. | any; generic; any |
| `--repeat-penalty` | Penalize recently used tokens. | 1.0 (off) | Keep one unless the task needs a penalty for recent repeated tokens. | any; generic; any |
| `--request-timeout-ms` | Limit request time in milliseconds. | 0 (off) | Keep zero unless requests must stop after a fixed time limit. | any; generic; any |
| `--seed` | Set the random sampling seed. | -1 (random); exact uses 1 | Use a nonnegative seed for reproducibility; exact uses one unless another nonnegative seed is supplied. | any; generic; any |
| `--served-model-name` | Set the model identifier exposed by the API. | Unset (uses model metadata name) | Set a name when clients require a model identifier different from the model file name. | any; generic; any |
| `--sessions` | Set the number of preallocated request sessions. | 1 | Keep one for exact output or select subagents when four requests must run concurrently. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--speculative` | Select the speculative draft backend. | mtp with an MTP sidecar; otherwise off | Keep MTP with its sidecar or select off when plain autoregressive decoding is required. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--temperature` | Set the randomness of token sampling. | preset (Qwen3.8: 1.0 thinking, 0.7 non-thinking) | Keep the model preset unless the task needs different randomness or greedy decoding with zero. | any; generic; any |
| `--think` | Set the default reasoning mode. | on | Keep on unless the task does not need reasoning. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--top-k` | Limit sampling to the highest probability candidates. | preset 20 (Qwen3.8) | Keep the model preset unless the task needs a different sampling candidate limit. | any; generic; any |
| `--top-p` | Filter sampling candidates by cumulative probability. | preset (0.95 thinking; 0.8 non-thinking unless GUFO_PRODUCTION_SAMPLING=1) | Keep the model preset unless the task needs a different cumulative probability limit. | any; generic; any |
| `--twin-model` | Select the prepared dense twin. | Unset | Select the prepared dense twin. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--verify-model` | Hash all eight manifest files plus distinct overrides before loading (about 2-4 minutes on SSD). | Unset | Use full verification to detect payload corruption before loading. | any; strix-halo-tuned; qwen3.8-flash-next |
| `BUBO_MANIFEST` | Select bubo-package.json; explicit --manifest takes precedence. | Unset | Set a package manifest when automatic discovery cannot find it. | any; strix-halo-tuned; any |
| `BUBO_MMPROJ` | Select the vision projector; explicit --mmproj takes precedence. | Unset | Select the vision projector; explicit --mmproj takes precedence. | any; strix-halo-tuned; any |
| `BUBO_MTP_MODEL` | Select the MTP file; explicit --mtp-model takes precedence. | Unset | Select the MTP file; explicit --mtp-model takes precedence. | any; strix-halo-tuned; any |
| `GUFO_CACHE_HOST_FLOOR_MB` | Reserve available host RAM during snapshot admission. | 0 (off) | Keep zero unless snapshot admission must reserve a minimum amount of available host RAM. | any; generic; any |
| `GUFO_DENSE_TWIN_BATCH` | Enable dense twin decode with multiple sessions. | off | Keep the profile setting unless batched twin comparisons require a different setting. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_MAX_N` | Limit the dense twin batch width. | 8 | Keep eight unless batched twin tests outside exact require a different batch limit. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_MODEL` | Select the four bit dense weights used for decode and verify. | Unset | Set the matching dense twin when faster decoding outside exact is required. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_DISK_DEFER` | Defer disk writes until idle. | Unset; 1 with --cache-disk | Keep the disk bundle setting unless disk write deferral tests require another setting. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_EVICT_DOMINATED` | Skip snapshots covered by a longer retained snapshot. | Unset; 1 with --cache-disk | Keep the disk bundle setting unless dominated snapshot eviction tests require another setting. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_EVICT_SKIP_TOKENS` | Skip snapshot writes close to an existing persisted prefix. | Unset; 8192 with --cache-disk | Keep 8192 with disk caching unless eviction tests require a different minimum prompt length. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_INPASS` | Enable disk snapshots during prefill passes. | 1 | Keep one unless disk cache tests require in-pass checkpoints to be disabled. | any; generic; any |
| `GUFO_DISK_PERSIST` | Persist disk snapshots on eviction after idle. | Unset; evict_idle with --cache-disk | Keep the disk bundle policy unless persistence tests require a different write policy. | any; generic; qwen3.8-flash-next |
| `GUFO_DRAFT_OUT_MODEL` | Select the four bit output head used for MTP drafts. | Unset | Set the matching draft head when faster MTP decoding outside exact is required. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_DRAFT_VOCAB_AUTO` | Set thresholds for automatic draft vocabulary selection. | 2,16,64 (2 of last 16 non-latin; back after 64 latin) | Keep the default thresholds unless automatic draft vocabulary tests require different thresholds. | any; generic; qwen3.8-flash-next |
| `GUFO_EOG_BOUNDARY` | Permit EOG tokens at allowed boundaries. | 1 | Keep it enabled unless end token boundary comparisons require it off. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_DEFER_P` | Set the deferred stop probability. | 0.9 | Keep the default unless early end token deferral tests require a different probability. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_MIN_PROB` | Set the minimum EOG probability required to stop. | 0.2 | Keep the default unless end token probability tests require another minimum. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_MIN_PROB_EXACT` | Compute EOG probability over the full vocabulary. | Off | Keep it off unless end token decisions require full vocabulary probabilities. | any; generic; qwen3.8-flash-next |
| `GUFO_EXPERT_V2K` | Run decode experts in the grouped expert kernel. | 0 | Keep it off unless grouped expert kernel tests require it; profile exact rejects 1. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_HOST` | Set the server address through the environment. | Unset | Leave it unset unless the server address must come from this environment variable. | any; generic; any |
| `GUFO_LENIENT` | Accept near miss MTP drafts inside thinking spans. | bounded with MTP; otherwise off | Keep bounded outside exact unless strict speculative acceptance is required. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LENIENT_LAMBDA` | Set the lenient speculative acceptance bound. | 0.5 | Keep the default unless lenient acceptance tests require a different bound. | any; generic; qwen3.8-flash-next |
| `GUFO_LENIENT_THINK_ONLY` | Limit lenient acceptance to thinking output. | on | Keep the thinking gate unless lenient acceptance tests must include non-thinking output. | any; generic; qwen3.8-flash-next |
| `GUFO_LOGIT_BIAS` | Enable request token probability adjustments. | off | Keep it off unless requests must change token probabilities with logit bias. | any; generic; any |
| `GUFO_LOOP_BREAKER` | Stop repeated answer and thinking loops. | 1 | Keep it enabled unless repeated text handling comparisons require it off. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LOOP_BREAKER_EOG_CLOSE` | Close a detected loop with an EOG token. | 1 | Keep it enabled unless loop termination tests must omit response closure. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LOOP_BREAKER_TRANSITION` | Select the loop breaker transition mode. | 2 | Keep two unless loop transition tests require another transition mode. | any; generic; any |
| `GUFO_LOOP_BREAKER_TRANSITION_TEXT` | Set the text used for loop breaker transitions. | built-in text | Keep the built-in text unless loop termination requires a different transition message. | any; generic; any |
| `GUFO_MAX_IMAGE_TOKENS` | Limit image tokens admitted for a request. | no cap | Keep it unset unless image token admission needs a fixed limit. | any; generic; qwen3.8-flash-next |
| `GUFO_MODEL_SUPPRESS_TOKENS` | Enable model token suppression. | off | Keep it off unless model token suppression is required. | any; generic; qwen3.8-flash-next |
| `GUFO_NPU_DIR` | Locate NPU artifacts for enabled prefill. | Executable folder/../tools/npu | Overrides the artifact directory beside the packaged bin folder; cannot enable NPU mode. | any; generic; any |
| `GUFO_NPU_GATE_SLICES` | Select zero through twelve paired query and gate slices. | 0 | Consumed only when NPU prefill is enabled. | any; generic; any |
| `GUFO_NPU_KEEPALIVE` | Enable periodic heartbeat with literal zero or one. | 1 | Consumed only when NPU prefill is enabled. | any; generic; any |
| `GUFO_NPU_KEEPALIVE_MS` | Set heartbeat interval to zero or one hundred through sixty thousand milliseconds. | 1000 | Consumed only when NPU prefill is enabled. | any; generic; any |
| `GUFO_NPU_SHEXP` | Enable shared expert staging with literal zero or one. | 1 | Consumed only when NPU prefill is enabled. | any; generic; any |
| `GUFO_NPU_SLICES` | Select one through twelve linear z slices for enabled prefill. | 6 | Consumed only when NPU prefill is enabled. | any; generic; any |
| `GUFO_NPU_TIMEOUT_MS` | Bound NPU completion to one through eight thousand milliseconds. | 8000 | Consumed only when NPU prefill is enabled. | any; generic; any |
| `GUFO_PORT` | Set the server port through the environment. | Unset | Leave it unset unless the server port must come from this environment variable. | any; generic; any |
| `GUFO_PREFILL_FORWARD_TOKENS` | Set the maximum tokens in each prefill forward pass. | 4096 | Keep 4096 unless prefill geometry tests outside exact require a different pass size. | any; generic; qwen3.8-flash-next |
| `GUFO_PRODUCTION_SAMPLING` | Enable the production sampling preset. | off | Keep it off unless production sampling policy tests require it on. | any; generic; qwen3.8-flash-next |
| `GUFO_REASONING_BUDGET` | Enable request reasoning token budgets. | off | Keep it off unless requests must enforce a reasoning token budget. | any; generic; qwen3.8-flash-next |
| `GUFO_ROUTED_LANEROW` | Enable the routed expert prefill kernel. | 1 | Keep the profile setting unless routed expert kernel comparisons require a different setting outside exact. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_SPEC_EOG_STOP` | Permit speculative EOG stops. | 1 | Keep it enabled unless speculative end token comparisons require it off. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_EOG_GUARD` | Block end of turn inside an unfinished thinking span. | 1 | Keep it enabled unless thinking termination comparisons require it off. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_GUARD_ESCALATE` | Close an unfinished thinking span after this many blocked end-of-turn tries. 0 turns it off. | 16 | Keep 16 unless thinking guard tests require a different end token streak length. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_BIAS` | Set a fixed thinking marker probability bias. | 0 | Keep zero unless a fixed thinking marker bias is required outside exact. | any; generic; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_BIAS_BY_EFFORT` | Map reasoning effort to think marker bias. | low:-2,medium:-1,xhigh:0 | Keep the default map unless reasoning effort tests require different thinking marker biases. | any; strix-halo-tuned; qwen3.8-flash-next |
| `HOST` | Set the server address through the environment fallback. | Unset | Leave it unset unless the server address must come from the environment fallback. | any; generic; any |
| `PORT` | Set the server port through the environment fallback. | Unset | Leave it unset unless the server port must come from the environment fallback. | any; generic; any |

## Profiles

| Profile | Effect |
| --- | --- |
| default | Use fast upstream practice for one session and medium reasoning effort. Output is not byte-exact. |
| subagents | Use four sessions with 131072 context tokens each and a 4 GiB RAM cache. |
| exact | Use reproducible plain autoregressive decoding with one session and no prompt cache. |
| stock | Keep upstream defaults. Inject no environment values or flag defaults. |

Subagents verifies up to 524288 total context tokens.
Use `--allow-unmeasured-memory` above this measured limit.
Four concurrent deep sessions can reach the host RAM guard near 50K tokens per session.
The default profile follows upstream gufo practice: fast output that is not byte-exact.
Exact is a reproducibility preset.
Output is byte-identical across runs of the same build and seed.
Output is identical to plain autoregressive decoding of that build with the same settings.
Output is not identical across builds or to other engines.
Exact disables speculation, behaviour policies, lenient acceptance, the dense twin, the draft head, and RAM and disk prompt caches.
Exact uses one session, forward size 4096, and seed 1 unless a seed is supplied.
Exact refuses values that enable disabled settings or change its session count or prefill geometry.
It runs at about half the decode speed, and every request prefills its full prompt.
Sidecar files (MTP, vision, dense twin, draft head) are set with their own options.
Reference: `bubo serve --profile stock --speculative off --no-cache --sessions 1 --seed 1 --think on --reasoning-effort medium`.
Use `--print-effective-settings` to inspect settings without loading tensors or starting a server.

## Diagnostics

| Name | What it does | Default | When to change | Platform and hardware |
| --- | --- | --- | --- | --- |
| `--debug-hooks` | Control debug hooks. | off | Inventory classification. | any; generic; any |
| `--help` | Control help. | Unset | unclassified | any; generic; any |
| `--manifest` | Control manifest. | Unset | unclassified | any; generic; any |
| `--max-score-tokens` | Control max score tokens. | 8192 | Inventory classification. | any; generic; any |
| `--model` | Control model. | required | Inventory classification. | any; generic; any |
| `--root` | Control root. | Unset | unclassified | any; generic; any |
| `--trace` | Write request and generation content to a JSONL file. | Unset | The trace file contains client content. | any; generic; any |
| `--ttl` | Control ttl. | Unset | unclassified | any; generic; any |
| `--verbose` | Control verbose. | Unset | unclassified | any; generic; any |
| `--voice` | Control voice. | Unset | unclassified | any; generic; any |
| `--voice-lang` | Control voice lang. | Unset | unclassified | any; generic; any |
| `--voice-text` | Control voice text. | Unset | unclassified | any; generic; any |
| `GPU_MAX_HW_QUEUES` | Set the hardware queue cap. | gufo sets | The engine manages this queue cap; a real environment value wins. | any; generic; any |
| `GUFO_A3B_DATA` | Control a3b data. | Unset | unclassified | any; generic; any |
| `GUFO_A3B_MODEL` | Control a3b model. | Unset | unclassified | any; generic; any |
| `GUFO_BATCH_DRAFT_HEAD` | Control batch draft head. | off | unclassified | any; generic; qwen3.8-flash-next |
| `GUFO_CACHE_REMOVAL_LOG` | Control cache removal log. | off | Inventory classification. | any; generic; any |
| `GUFO_DEBUG_HOOKS` | Control debug hooks. | off | Inventory classification. | any; generic; any |
| `GUFO_DENSE_TWIN_ALLOW_Q8` | Control dense twin allow q8. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_SHADOW` | Control dense twin shadow. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_TRACE` | Control dense twin trace. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_BOUNDARY_ALIGN` | Control disk boundary align. | as shown | Inventory classification. | any; generic; any |
| `GUFO_DISK_DEFER_MS` | Control disk defer ms. | as shown | Inventory classification. | any; generic; any |
| `GUFO_DISK_EVICT_BACKLOG_BYTES` | Control disk evict backlog bytes. | as shown | Inventory classification. | any; generic; any |
| `GUFO_DISK_IDLE_MS` | Control disk idle ms. | 30000 (0..600000) | Inventory classification. | any; generic; any |
| `GUFO_DISK_SLICE` | Control disk slice. | as shown | Inventory classification. | any; generic; any |
| `GUFO_DISPATCH_TELEMETRY` | Control dispatch telemetry. | off | Inventory classification. | any; generic; any |
| `GUFO_EVAL_API_KEY` | Control eval api key. | Unset | unclassified | any; generic; any |
| `GUFO_EVAL_ARTIFACT_ID` | Control eval artifact id. | Unset | unclassified | any; generic; any |
| `GUFO_EVAL_MODEL_REVISION` | Control eval model revision. | Unset | unclassified | any; generic; any |
| `GUFO_EVAL_SERVER_COMMIT` | Control eval server commit. | Unset | unclassified | any; generic; any |
| `GUFO_EVAL_SERVER_CONFIG` | Control eval server config. | Unset | unclassified | any; generic; any |
| `GUFO_EVAL_SERVER_LABEL` | Control eval server label. | Unset | unclassified | any; generic; any |
| `GUFO_HIPBLASLT_PLAN_CACHE` | Control hipblaslt plan cache. | - | Inventory classification. | any; generic; any |
| `GUFO_IQ4_WIDE_VARIANT` | Control iq4 wide variant. | 4 | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_IQ4_WMMA_WAVES` | Control iq4 wmma waves. | 8 | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_LOG_SYNC` | Write logs synchronously for crash diagnostics. | off | Use 1 for synchronous logs. A hard crash can lose about 100 ms of asynchronous logs. | any; generic; any |
| `GUFO_MEM_ACCOUNTING` | Control mem accounting. | off | Inventory classification. | any; generic; any |
| `GUFO_MOE_BENCH` | Control moe bench. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_MTP_TRACE` | Write one trace line for each decode cycle. | off | Trace output uses bounded asynchronous logging. Synchronous trace output can reduce decode speed. | any; generic; qwen3.8-flash-next |
| `GUFO_PLATFORM_TUNING` | Control Windows timing and memory placement switches. | All switches on on Windows | GUFO_PLATFORM_TUNING=none is a diagnostic that slows Windows decode. | windows; generic; any |
| `GUFO_PREFILL_DIAG` | Control prefill diag. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_REQUIRE_GFX1151` | Control require hip. | off | Inventory classification. | any; generic; any |
| `GUFO_REQUIRE_GPU` | Control require hip. | off | Inventory classification. | any; generic; any |
| `GUFO_REQUIRE_HIP` | Control require hip. | off | Inventory classification. | any; generic; any |
| `GUFO_SURVIVAL_FLOOR` | Control survival floor. | ON | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_LIST` | Control think marker list. | none | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_TRACE` | Control trace. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `HOME` | Control home. | - | Inventory classification. | any; generic; any |
| `LOCALAPPDATA` | Control localappdata. | - | Inventory classification. | any; generic; any |
| `PATH` | Select runtime DLL search paths. | Packaging controls the runtime path | Keep runtime DLLs beside the executable; packaging remains a later job. | windows; generic; any |
| `XDG_CACHE_HOME` | Control xdg cache home. | - | Inventory classification. | any; generic; any |

## Removed

Other model switches remain available for their model families.

| Name | What it does | Default | When to change | Platform and hardware |
| --- | --- | --- | --- | --- |
| `--dflash-model` | Control dflash model. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `--draft-policy` | Control draft policy. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `--dspark-model` | Control dspark model. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `A3B_ATTN_CHECK` | Control the a3b attn check setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_ATTN_PRECISE` | Control the a3b attn precise setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_DECODE_ATTN` | Control the a3b decode attn setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_DFLASH_PROFILE` | Control the a3b dflash profile setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_NO_RESUME` | Control the a3b no resume setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_PROBE_ONLY` | Control the a3b probe only setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_ATTN_CHECK` | Control the a3b wide attn check setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_BF16_BLAS` | Control the a3b wide bf16 blas setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_BLAS_F32` | Control the a3b wide blas f32 setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_DEBUG` | Control the a3b wide debug setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_F16_MAXCOLS` | Control the a3b wide f16 maxcols setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_F16_MINROWS` | Control the a3b wide f16 minrows setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_FN_ATTN` | Control the a3b wide fn attn setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_NO_DOWN64` | Control the a3b wide no down64 setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_NO_F16` | Control the a3b wide no f16 setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `A3B_WIDE_PROFILE` | Control the a3b wide profile setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `DS4_LOCK_FILE` | Control the ds4 lock file setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `GUFO_DEEPSEEK_DSPARK_TIMING` | Control deepseek dspark timing. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_DSPARK_TRACE` | Control deepseek dspark trace. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_DSPARK_WIDTH_TRACE` | Control deepseek dspark width trace. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE` | Control deepseek rocm decode stage profile. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE` | Control deepseek rocm graph prefill profile. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE` | Control deepseek rocm graph token profile. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE` | Control deepseek rocm indexer stage profile. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE` | Control deepseek rocm layer stage profile. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE` | Control deepseek rocm q stage profile. | Unset | Use this diagnostic with DeepSeek V4 Flash. | any; generic; deepseek-v4-flash |
| `GUFO_FFMPEG` | Control ffmpeg. | - | Use this switch with its model family. | any; generic; minimax-h3 |
| `GUFO_FFPROBE` | Control ffprobe. | - | Use this switch with its model family. | any; generic; minimax-h3 |
| `GUFO_GROUPED_DOWN` | Control grouped down. | Removed | This experiment has no source reader. | any; generic; qwen3.8-flash-next |
| `GUFO_PREFILL_STEP_LOG` | Control prefill step log. | off | This tree has no reader for this variable. | any; generic; any |
| `GUFO_PREFILL_TARGET_MS` | Control prefill target ms. | internal | This tree has no reader for this variable. | any; generic; any |
| `GUFO_QWEN27_DRAFTLOG` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_FLUSH` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_LOOKUP` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_LOOKUP_MODE` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_PROFILE` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_WEIGHT_UPLOAD_READERS` | Control weight upload readers. | 16 | This tree has no reader for this variable. | windows; generic; any |
| `QWEN3_ASR_ARTIFACTS` | Control the qwen3 asr artifacts setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `QWEN3_ASR_MODEL_ROOT` | Control the qwen3 asr model root setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `QWEN3_ASR_TEST_WAV` | Control the qwen3 asr test wav setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
| `TEMP` | Control the temp setting. | Unset | Other model or test configuration; no release default. | any; generic; any |
