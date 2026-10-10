# Release switches

Generated from `src/cli/serve/release_switches.json`.

Profile defaults apply to Qwen3.8 Flash Next.
Other model families keep their existing defaults.
Explicit flags and real environment values take precedence.
Empty environment values count as unset and produce a warning.

## Options and defaults

| Name | What it does | Default | When to change | Platform and hardware |
| --- | --- | --- | --- | --- |
| `--allow-unmeasured-memory` | Allow subagent context above the measured token limit. | Unset | Select the required release policy. | any; generic; any |
| `--api-key` | Control api key. | none | Inventory classification. | any; generic; any |
| `--cache-disk` | Control cache disk. | Off | Default staging limits persistence and restores beyond about 70K tokens. | any; generic; any |
| `--cache-disk-bytes` | Control cache disk bytes. | 8,589,934,592 | Inventory classification. | any; generic; any |
| `--cache-disk-staging-bytes` | Control cache disk staging bytes. | 0 = auto (<= disk budget, <= 1/8 available RAM) | Inventory classification. | any; generic; any |
| `--cache-ram-bytes` | Set the RAM snapshot cache budget. | 6442450944 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--context` | Set the context token limit for each session. | 262144 | Follow-up: drop this default if the model's native context equals 262144. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--draft-tokens` | Control draft tokens. | 7 | Inventory classification. | any; generic; qwen3.8-flash-next |
| `--exact` | Select the exact profile. | Unset | Select the required release policy. | any; generic; any |
| `--frequency-penalty` | Control frequency penalty. | 1.0 / 64 / 0.0 (off) | Inventory classification. | any; generic; any |
| `--host` | Control host. | 127.0.0.1 | Inventory classification. | any; generic; any |
| `--log-level` | Control log level. | info | Inventory classification. | any; generic; any |
| `--log-progress` | Control log progress. | off | Inventory classification. | any; generic; any |
| `--max-buffered-output-bytes` | Control max buffered output bytes. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--max-buffered-output-total` | Control max buffered output total. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--max-connections` | Control max connections. | 16 | Inventory classification. | any; generic; any |
| `--max-output-bytes` | Control max output bytes. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--max-pending` | Control max pending. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--max-pending-per-client` | Control max pending per client. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--max-request-bytes` | Set the maximum HTTP request body size. | 67108864 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--max-tokens` | Control max tokens. | -1 | Inventory classification. | any; generic; any |
| `--min-draft-tokens` | Control min draft tokens. | 1 | Inventory classification. | any; generic; qwen3.8-flash-next |
| `--min-keep` | Control min keep. | 0 | Inventory classification. | any; generic; any |
| `--min-p` | Control min p. | 0.0 | Inventory classification. | any; generic; any |
| `--mmproj` | Control mmproj. | Unset until the manifest job | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-draft-vocab` | Select the MTP draft head vocabulary. | auto | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-model` | Control mtp model. | Unset until the manifest job | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-policy` | Select the MTP draft length controller. | survival | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--no-behaviour` | Disable the eight behaviour defaults. | Unset | Select the required release policy. | any; generic; any |
| `--no-cache-disk` | Disable the disk snapshot cache. | Unset | Select the required release policy. | any; generic; any |
| `--no-draft-head` | Suppress the manifest draft output head. | Unset | Select the required release policy. | any; generic; any |
| `--no-mtp` | Disable MTP defaults and the speculative backend. | Unset | Select the required release policy. | any; generic; any |
| `--no-prompt-lookup` | Disable prompt lookup proposals. | Unset | Select the required release policy. | any; generic; any |
| `--no-twin` | Suppress the manifest dense twin. | Unset | Select the required release policy. | any; generic; any |
| `--no-vision` | Suppress the manifest vision sidecar. | Unset | Select the required release policy. | any; generic; any |
| `--port` | Control port. | 8080 | Inventory classification. | any; generic; any |
| `--prefill-chunk` | Limit prompt tokens between active decode rounds. | 512 | Larger chunks increase pauses for other active sessions. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--presence-penalty` | Control presence penalty. | preset (non-thinking 1.5 unless GUFO_PRODUCTION_SAMPLING=1) | Inventory classification. | any; generic; any |
| `--preserve-thinking` | Control preserve thinking. | auto | Inventory classification. | any; generic; qwen3.8-flash-next |
| `--print-effective-settings` | Print resolved settings as JSON and exit. | Unset | Select the required release policy. | any; generic; any |
| `--profile` | Select the release settings profile. | Unset | Select the required release policy. | any; generic; any |
| `--prompt-lookup` | Propose tokens from an earlier conversation match. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--reasoning-effort` | Set the default reasoning effort. | medium | Medium selects a think-marker bias of -1 in the production map. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--repeat-last-n` | Control repeat last n. | 1.0 / 64 / 0.0 (off) | Inventory classification. | any; generic; any |
| `--repeat-penalty` | Control repeat penalty. | 1.0 / 64 / 0.0 (off) | Inventory classification. | any; generic; any |
| `--request-timeout-ms` | Control request timeout ms. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--seed` | Control seed. | -1 random | Inventory classification. | any; generic; any |
| `--served-model-name` | Control served model name. | none | Inventory classification. | any; generic; any |
| `--sessions` | Set the number of preallocated request sessions. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--speculative` | Select the speculative draft backend. | mtp | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--temperature` | Control temperature. | preset (Qwen3.8: 1.0 thinking, 0.7 non-thinking) | Inventory classification. | any; generic; any |
| `--think` | Set the default reasoning mode. | on | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--top-k` | Control top k. | preset 20 (Qwen3.8) | Inventory classification. | any; generic; any |
| `--top-p` | Control top p. | preset (0.95 thinking; 0.8 non-thinking unless GUFO_PRODUCTION_SAMPLING=1) | Inventory classification. | any; generic; any |
| `GUFO_CACHE_HOST_FLOOR_MB` | Control cache host floor mb. | 0 (off) | Inventory classification. | any; generic; any |
| `GUFO_DENSE_TWIN_BATCH` | Enable dense twin decode with multiple sessions. | off | Subagents enables the measured batched twin path. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_MAX_N` | Control dense twin max n. | 8 | The built-in value needs no injection. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_MODEL` | Select the four bit dense weights used for decode and verify. | Unset until the manifest job | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_DISK_DEFER` | Defer disk writes until idle. | Unset; 1 with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_EVICT_DOMINATED` | Skip snapshots covered by a longer retained snapshot. | Unset; 1 with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_EVICT_SKIP_TOKENS` | Skip snapshot writes close to an existing persisted prefix. | Unset; 8192 with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_INPASS` | Control disk inpass. | 1 | The built-in value needs no injection. | any; generic; any |
| `GUFO_DISK_PERSIST` | Persist disk snapshots on eviction after idle. | Unset; evict_idle with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DRAFT_OUT_MODEL` | Select the four bit output head used for MTP drafts. | Unset until the manifest job | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_DRAFT_VOCAB_AUTO` | Control draft vocab auto. | 2,16,64 (2 of last 16 non-latin; back after 64 latin) | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_EOG_BOUNDARY` | Permit EOG tokens at allowed boundaries. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_DEFER_P` | Set the deferred stop probability. | 0.9 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_MIN_PROB` | Set the minimum EOG probability required to stop. | 0.2 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_MIN_PROB_EXACT` | Compute EOG probability over the full vocabulary. | Off | Use the full vocabulary EOG probability; production 0.2 uses the top-20 bound. | any; generic; qwen3.8-flash-next |
| `GUFO_HOST` | Control host. | - | Inventory classification. | any; generic; any |
| `GUFO_LENIENT` | Accept near miss MTP drafts inside thinking spans. | bounded | Bounded acceptance changes the sampling distribution. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LENIENT_LAMBDA` | Control lenient lambda. | 0.5 | The built-in value needs no injection. | any; generic; qwen3.8-flash-next |
| `GUFO_LENIENT_THINK_ONLY` | Control lenient think only. | on | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_LOGIT_BIAS` | Control logit bias. | off | Inventory classification. | any; generic; any |
| `GUFO_LOOP_BREAKER` | Stop repeated answer and thinking loops. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LOOP_BREAKER_EOG_CLOSE` | Close a detected loop with an EOG token. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LOOP_BREAKER_TRANSITION` | Control loop breaker transition. | 2 | The built-in value needs no injection. | any; generic; any |
| `GUFO_LOOP_BREAKER_TRANSITION_TEXT` | Control loop breaker transition text. | built-in text | Inventory classification. | any; generic; any |
| `GUFO_MAX_IMAGE_TOKENS` | Control max image tokens. | no cap | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_MODEL_SUPPRESS_TOKENS` | Control model suppress tokens. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_PORT` | Control port. | - | Inventory classification. | any; generic; any |
| `GUFO_PREFILL_FORWARD_TOKENS` | Set the maximum tokens in each prefill forward pass. | 4096 | Exact pins 4096; a real environment value wins and is logged. | any; generic; qwen3.8-flash-next |
| `GUFO_PRODUCTION_SAMPLING` | Control production sampling. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_REASONING_BUDGET` | Control reasoning budget. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_ROUTED_LANEROW` | Enable the routed expert prefill kernel. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_SPEC_EOG_STOP` | Permit speculative EOG stops. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_EOG_GUARD` | Block end of turn inside an unfinished thinking span. | 1 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_BIAS` | Control think marker bias. | 0 | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_BIAS_BY_EFFORT` | Map reasoning effort to think marker bias. | low:-2,medium:-1,xhigh:0 | Inventory classification. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_WEIGHT_UPLOAD_READERS` | Control weight upload readers. | 16 | Inventory classification. | windows; generic; any |
| `HOST` | Control host. | Unset | Use the environment fallback for the server address. | any; generic; any |
| `PORT` | Control port. | Unset | Use the environment fallback for the server address. | any; generic; any |

## Profiles

| Profile | Effect |
| --- | --- |
| default | Use production settings for one session. Use medium reasoning effort. |
| subagents | Use four sessions with 131072 context tokens each and a 4 GiB RAM cache. |
| exact | Disable lenient acceptance and the dense twin. Keep behaviour policies and the draft head. |
| stock | Keep upstream defaults. Inject no environment values or flag defaults. |

Subagents verifies up to 524288 total context tokens.
Use `--allow-unmeasured-memory` above this measured limit.
Four concurrent deep sessions can reach the host RAM guard near 50K tokens per session.
Exact pins seed 1, one session, and forward size 4096 when unset.
A real environment value can override a pin and produces a log.
Exact refuses an explicit dense twin or bounded lenient acceptance.
Exact does not promise text identity with plain autoregressive decoding.
Verify widths and unaligned warm restores can change floating point results.
Manifest artefacts remain unavailable until the manifest job.
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
| `GUFO_MEM_ACCOUNTING` | Control mem accounting. | off | Inventory classification. | any; generic; any |
| `GUFO_MOE_BENCH` | Control moe bench. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_MTP_TRACE` | Write one trace line for each decode cycle. | off | With stderr redirected by Start-Process, this trace costs about 20 % decode until asynchronous logging is available. | any; generic; qwen3.8-flash-next |
| `GUFO_PLATFORM_TUNING` | Control Windows timing and memory placement switches. | All switches on on Windows | GUFO_PLATFORM_TUNING=none is a diagnostic that slows Windows decode. | windows; generic; any |
| `GUFO_PREFILL_DIAG` | Control prefill diag. | off | Inventory classification. | any; generic; qwen3.8-flash-next |
| `GUFO_PREFILL_STEP_LOG` | Control prefill step log. | off | Inventory classification. | any; generic; any |
| `GUFO_PREFILL_TARGET_MS` | Control prefill target ms. | internal | Inventory classification. | any; generic; any |
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
| `GUFO_QWEN27_DRAFTLOG` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_FLUSH` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_LOOKUP` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_LOOKUP_MODE` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
| `GUFO_QWEN27_PROFILE` | Control qwen27 profile. | - | Use this switch with its model family. | any; generic; qwen3.8-27b |
