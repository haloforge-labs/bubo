# Switch audiences

Generated from `src/cli/serve/release_switches.json`.

The main panel shows `user` switches. The Advanced panel shows `advanced` switches. Its collapsed Debug section shows permitted `debug` switches with their purpose and default for troubleshooting.
GUFO_REASONING_BUDGET, GUFO_THINK_GUARD_ESCALATE, GUFO_LOOP_BREAKER_EOG_CLOSE and GUFO_LOOP_BREAKER_TRANSITION remain in the complete CLI diagnostics registry. The policy against thinking-token budgets, caps and force-close controls excludes them from GUI controls.
Changes to a running server take effect after restart when `restart_required` is true.

The decode twin head makes writing faster. Hardware controls take effect after restart.

## Add a switch

Add its complete row to `src/cli/serve/release_switches.json`, including `group` and `allowed`. Choose a group from Hardware and memory, Memory, Speed, Behaviour, Server, or Other. Add the switch name with a null value to `docs/switch-evidence.json` for the name-coverage contract. Run `python tools/release/gen_switches.py`, then `python tools/release/gen_switches.py --check`. The generator writes the C++ include, family include, flags docs, flags golden, listing golden, listing schema, GUI panel fixture, audience docs, and the payload of `tests/cli/contract/valid/switches.json`; the recorded event envelope stays intact. No Go or TypeScript edit is needed.

`allowed` owns enum values and numeric bounds. `allowed.suggested` records reader recommendations without restricting accepted values. Unsigned decimal, prefill prefix, effort-map, and platform-tuning grammars remain in the CLI reader; the recipe test checks every listed value and numeric boundary against C++. Secret rows must set `echo: false` and `gui_control: false`.

## CLI registry contract

`bubo serve --list-switches --json` emits a `switches` event whose payload is the object in `tests/cli/contract/switches.golden.json`: `schema_version: 1` and one complete catalog row per switch. The JSON schema is `tests/cli/contract/switches.schema.json`. The listing is safe before model load and server start. The renderer must preserve the catalog order.

`RenderReleaseDocs` must match `docs/FLAGS.md` and `tests/fixtures/release_flags_golden.md`, including the final engine-managed queue cap table retained for the brand gate.

`--loop-breaker on|off` maps to `GUFO_LOOP_BREAKER` values `1|0`. `--think-marker-bias` maps its finite number to `GUFO_THINK_MARKER_BIAS`. Each environment row lists the CLI flag in `suppressed_by`; an explicit CLI value takes precedence over the environment and profile default. Exact keeps both controls off or zero.

`GUFO_MAX_IMAGE_TOKENS` accepts positive integers. The disk cache byte caps use unsigned integers. The C++ value validator and the JSON schema must agree with these catalog rules.

Counts: user 35, advanced 13, debug 177.

| Switch | Group | Audience | Purpose |
| --- | --- | --- | --- |
| `--allow-unmeasured-memory` | Memory | debug | Allow subagent context above the measured token limit. |
| `--api-key` | Server | user | Require an API key from clients. |
| `--cache-disk` | Memory | user | Set the disk prompt cache directory. |
| `--cache-disk-bytes` | Memory | user | Limit disk prompt cache storage. |
| `--cache-disk-staging-bytes` | Memory | user | Limit RAM used to stage disk prompt snapshots. |
| `--cache-ram-bytes` | Memory | user | Set the RAM snapshot cache budget. |
| `--context` | Memory | user | Set the context token limit for each session. |
| `--debug-hooks` | Other | debug | Control debug hooks. |
| `--dflash-model` | Other | debug | Control dflash model. |
| `--draft-head-model` | Speed | advanced | Select the prepared MTP draft head. |
| `--draft-policy` | Speed | debug | Control draft policy. |
| `--draft-tokens` | Speed | advanced | Set the maximum speculative draft length. |
| `--dspark-model` | Other | debug | Control dspark model. |
| `--exact` | Other | debug | Select the exact profile. |
| `--frequency-penalty` | Behaviour | user | Penalize tokens by their generated occurrence count. |
| `--help` | Other | debug | Control help. |
| `--host` | Server | user | Set the server listen address. |
| `--log-level` | Server | debug | Set the amount of server log detail. |
| `--log-progress` | Server | debug | Log prompt processing progress. |
| `--loop-breaker` | Behaviour | user | Stop repeated answer and thinking loops. |
| `--manifest` | Other | debug | Control manifest. |
| `--max-buffered-output-bytes` | Server | debug | Limit buffered output for each request. |
| `--max-buffered-output-total` | Server | debug | Limit buffered output across all requests. |
| `--max-connections` | Server | debug | Limit simultaneous HTTP connections. |
| `--max-output-bytes` | Server | debug | Limit output text bytes for each request. |
| `--max-pending` | Server | debug | Limit queued requests. |
| `--max-pending-per-client` | Server | debug | Limit queued requests from each client. |
| `--max-request-bytes` | Server | debug | Set the maximum HTTP request body size. |
| `--max-score-tokens` | Server | debug | Control max score tokens. |
| `--max-tokens` | Behaviour | user | Limit new tokens generated per request. |
| `--min-draft-tokens` | Speed | advanced | Set the minimum speculative draft length. |
| `--min-keep` | Behaviour | user | Retain a minimum number of sampling candidates. |
| `--min-p` | Behaviour | user | Filter tokens by probability relative to the best token. |
| `--mmproj` | Memory | user | Set the vision file. |
| `--model` | Other | user | Control model. |
| `--mtp-draft-vocab` | Speed | advanced | Select the MTP draft head vocabulary. |
| `--mtp-model` | Speed | advanced | Set the MTP file. |
| `--mtp-policy` | Speed | advanced | Select the MTP draft length controller. |
| `--no-behaviour` | Behaviour | debug | Disable the eight behaviour defaults. |
| `--no-cache` | Memory | debug | Disable RAM and disk prompt caches for all requests. |
| `--no-cache-disk` | Memory | user | Disable the disk snapshot cache. |
| `--no-draft-head` | Speed | advanced | Suppress the manifest draft output head. |
| `--no-mtp` | Speed | advanced | Disable MTP defaults and the speculative backend. |
| `--no-prompt-lookup` | Speed | advanced | Disable prompt lookup proposals. |
| `--no-twin` | Hardware and memory | user | Suppress the manifest dense twin. |
| `--no-vision` | Memory | user | Turn off the vision file that the package names. |
| `--npu-prefill` | Hardware and memory | user | Use NPU prefill when available. |
| `--port` | Server | user | Set the server listen port. |
| `--prefill-chunk` | Speed | advanced | Limit prompt tokens between active decode rounds. |
| `--presence-penalty` | Behaviour | user | Penalize tokens already generated in the response. |
| `--preserve-thinking` | Behaviour | debug | Select which thinking history to retain. |
| `--print-effective-settings` | Other | debug | Print resolved settings as JSON and exit. |
| `--profile` | Other | user | Select the release settings profile. |
| `--prompt-lookup` | Speed | advanced | Propose tokens from an earlier conversation match. |
| `--reasoning-effort` | Behaviour | user | Set the default reasoning effort. |
| `--repeat-last-n` | Behaviour | user | Set the recent token window for repetition penalties. |
| `--repeat-penalty` | Behaviour | user | Penalize recently used tokens. |
| `--request-timeout-ms` | Server | debug | Limit request time in milliseconds. |
| `--root` | Other | debug | Control root. |
| `--seed` | Behaviour | user | Set the random sampling seed. |
| `--served-model-name` | Server | user | Set the model identifier exposed by the API. |
| `--sessions` | Memory | user | Set the number of preallocated request sessions. |
| `--speculative` | Speed | advanced | Select the speculative draft backend. |
| `--stall-timeout-ms` | Server | debug | Exit as device lost when generation makes no progress for this many milliseconds. |
| `--temperature` | Behaviour | user | Set the randomness of token sampling. |
| `--think` | Behaviour | user | Set the default reasoning mode. |
| `--think-marker-bias` | Behaviour | user | Set a fixed thinking marker probability bias. |
| `--top-k` | Behaviour | user | Limit sampling to the highest probability candidates. |
| `--top-p` | Behaviour | user | Filter sampling candidates by cumulative probability. |
| `--trace` | Other | debug | Write request and generation content to a JSONL file. |
| `--ttl` | Other | debug | Control ttl. |
| `--twin-model` | Hardware and memory | user | Select the prepared dense twin. |
| `--verbose` | Other | debug | Control verbose. |
| `--verify-model` | Other | advanced | Hash all eight manifest files plus distinct overrides before loading (about 2-4 minutes on SSD). |
| `--voice` | Other | debug | Control voice. |
| `--voice-lang` | Other | debug | Control voice lang. |
| `--voice-text` | Other | debug | Control voice text. |
| `A3B_ATTN_CHECK` | Other | debug | Control the a3b attn check setting. |
| `A3B_ATTN_PRECISE` | Other | debug | Control the a3b attn precise setting. |
| `A3B_DECODE_ATTN` | Other | debug | Control the a3b decode attn setting. |
| `A3B_DFLASH_PROFILE` | Other | debug | Control the a3b dflash profile setting. |
| `A3B_NO_RESUME` | Other | debug | Control the a3b no resume setting. |
| `A3B_PROBE_ONLY` | Other | debug | Control the a3b probe only setting. |
| `A3B_WIDE_ATTN_CHECK` | Other | debug | Control the a3b wide attn check setting. |
| `A3B_WIDE_BF16_BLAS` | Other | debug | Control the a3b wide bf16 blas setting. |
| `A3B_WIDE_BLAS_F32` | Other | debug | Control the a3b wide blas f32 setting. |
| `A3B_WIDE_DEBUG` | Other | debug | Control the a3b wide debug setting. |
| `A3B_WIDE_F16_MAXCOLS` | Other | debug | Control the a3b wide f16 maxcols setting. |
| `A3B_WIDE_F16_MINROWS` | Other | debug | Control the a3b wide f16 minrows setting. |
| `A3B_WIDE_FN_ATTN` | Other | debug | Control the a3b wide fn attn setting. |
| `A3B_WIDE_NO_DOWN64` | Other | debug | Control the a3b wide no down64 setting. |
| `A3B_WIDE_NO_F16` | Other | debug | Control the a3b wide no f16 setting. |
| `A3B_WIDE_PROFILE` | Other | debug | Control the a3b wide profile setting. |
| `BUBO_API_KEY` | Other | debug | Provide the server API key; --api-key takes precedence. |
| `BUBO_MANIFEST` | Other | debug | Select bubo-package.json; explicit --manifest takes precedence. |
| `BUBO_MMPROJ` | Memory | debug | Select the vision projector; explicit --mmproj takes precedence. |
| `BUBO_MTP_MODEL` | Speed | debug | Select the MTP file; explicit --mtp-model takes precedence. |
| `BUBO_RUNTIME_DIR` | Other | debug | Set the prepared runtime folder. |
| `DS4_LOCK_FILE` | Other | debug | Control the ds4 lock file setting. |
| `GPU_MAX_HW_QUEUES` | Other | debug | Set the hardware queue cap. |
| `GUFO_A3B_DATA` | Other | debug | Control a3b data. |
| `GUFO_A3B_MODEL` | Other | debug | Control a3b model. |
| `GUFO_BATCH_DRAFT_HEAD` | Other | debug | Control batch draft head. |
| `GUFO_CACHE_HOST_FLOOR_MB` | Memory | debug | Reserve available host RAM during snapshot admission. |
| `GUFO_CACHE_REMOVAL_LOG` | Memory | debug | Control cache removal log. |
| `GUFO_DEBUG_HOOKS` | Other | debug | Control debug hooks. |
| `GUFO_DEEPSEEK_DSPARK_TIMING` | Other | debug | Control deepseek dspark timing. |
| `GUFO_DEEPSEEK_DSPARK_TRACE` | Other | debug | Control deepseek dspark trace. |
| `GUFO_DEEPSEEK_DSPARK_WIDTH_TRACE` | Other | debug | Control deepseek dspark width trace. |
| `GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE` | Other | debug | Control deepseek rocm decode stage profile. |
| `GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE` | Other | debug | Control deepseek rocm graph prefill profile. |
| `GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE` | Other | debug | Control deepseek rocm graph token profile. |
| `GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE` | Other | debug | Control deepseek rocm indexer stage profile. |
| `GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE` | Other | debug | Control deepseek rocm layer stage profile. |
| `GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE` | Other | debug | Control deepseek rocm q stage profile. |
| `GUFO_DENSE_TWIN_ALLOW_Q8` | Other | debug | Control dense twin allow q8. |
| `GUFO_DENSE_TWIN_BATCH` | Hardware and memory | debug | Enable dense twin decode with multiple sessions. |
| `GUFO_DENSE_TWIN_MAX_N` | Hardware and memory | debug | Limit the dense twin batch width. |
| `GUFO_DENSE_TWIN_MODEL` | Hardware and memory | debug | Select the four bit dense weights used for decode and verify. |
| `GUFO_DENSE_TWIN_SHADOW` | Other | debug | Control dense twin shadow. |
| `GUFO_DENSE_TWIN_TRACE` | Other | debug | Control dense twin trace. |
| `GUFO_DISK_BOUNDARY_ALIGN` | Memory | debug | Control disk boundary align. |
| `GUFO_DISK_DEFER` | Memory | debug | Defer disk writes until idle. |
| `GUFO_DISK_DEFER_MS` | Memory | debug | Control disk defer ms. |
| `GUFO_DISK_EVICT_BACKLOG_BYTES` | Memory | debug | Control disk evict backlog bytes. |
| `GUFO_DISK_EVICT_DOMINATED` | Memory | debug | Skip snapshots covered by a longer retained snapshot. |
| `GUFO_DISK_EVICT_SKIP_TOKENS` | Memory | debug | Skip snapshot writes close to an existing persisted prefix. |
| `GUFO_DISK_IDLE_MS` | Memory | debug | Control disk idle ms. |
| `GUFO_DISK_INPASS` | Memory | debug | Enable disk snapshots during prefill passes. |
| `GUFO_DISK_PERSIST` | Memory | debug | Persist disk snapshots on eviction after idle. |
| `GUFO_DISK_SLICE` | Memory | debug | Control disk slice. |
| `GUFO_DISPATCH_TELEMETRY` | Other | debug | Control dispatch telemetry. |
| `GUFO_DRAFT_OUT_MODEL` | Speed | debug | Select the four bit output head used for MTP drafts. |
| `GUFO_DRAFT_VOCAB_AUTO` | Speed | debug | Set thresholds for automatic draft vocabulary selection. |
| `GUFO_EOG_BOUNDARY` | Behaviour | debug | Permit EOG tokens at allowed boundaries. |
| `GUFO_EOG_DEFER_P` | Behaviour | debug | Set the deferred stop probability. |
| `GUFO_EOG_MIN_PROB` | Behaviour | debug | Set the minimum EOG probability required to stop. |
| `GUFO_EOG_MIN_PROB_EXACT` | Behaviour | debug | Compute EOG probability over the full vocabulary. |
| `GUFO_EVAL_API_KEY` | Other | debug | Control eval api key. |
| `GUFO_EVAL_ARTIFACT_ID` | Other | debug | Control eval artifact id. |
| `GUFO_EVAL_MODEL_REVISION` | Other | debug | Control eval model revision. |
| `GUFO_EVAL_SERVER_COMMIT` | Other | debug | Control eval server commit. |
| `GUFO_EVAL_SERVER_CONFIG` | Other | debug | Control eval server config. |
| `GUFO_EVAL_SERVER_LABEL` | Other | debug | Control eval server label. |
| `GUFO_EXPERT_OVERLAY_KEEP` | Memory | debug | Name the routed-expert tensors that keep the base model's weights when an overlay is set. |
| `GUFO_EXPERT_OVERLAY_MODEL` | Memory | debug | Replace the routed experts of the model with those of another GGUF of the same architecture. |
| `GUFO_EXPERT_V2K` | Speed | debug | Grouped expert kernel (decode). |
| `GUFO_FFMPEG` | Other | debug | Control ffmpeg. |
| `GUFO_FFPROBE` | Other | debug | Control ffprobe. |
| `GUFO_GROUPED_DOWN` | Other | debug | Control grouped down. |
| `GUFO_HIPBLASLT_PLAN_CACHE` | Other | debug | Control hipblaslt plan cache. |
| `GUFO_HOST` | Server | debug | Set the server address through the environment. |
| `GUFO_INCREMENTAL_TOKENIZE` | Speed | debug | Reuse verified text render and token prefixes for exact conversation appends. Inspect incremental_tokenize counters in render timeline events. |
| `GUFO_IQ4_WIDE_VARIANT` | Other | debug | Control iq4 wide variant. |
| `GUFO_IQ4_WMMA_WAVES` | Other | debug | Control iq4 wmma waves. |
| `GUFO_IQ_ROUTE_MIN_ROWS` | Speed | debug | Set the minimum forward-pass rows for the IQ expert WMMA route. |
| `GUFO_LENIENT` | Speed | debug | Accept near miss MTP drafts inside thinking spans. |
| `GUFO_LENIENT_LAMBDA` | Speed | debug | Set the lenient speculative acceptance bound. |
| `GUFO_LENIENT_THINK_ONLY` | Speed | debug | Limit lenient acceptance to thinking output. |
| `GUFO_LOGIT_BIAS` | Behaviour | debug | Enable request token probability adjustments. |
| `GUFO_LOG_SYNC` | Other | debug | Write logs synchronously for crash diagnostics. |
| `GUFO_LOOP_BREAKER` | Behaviour | debug | Stop repeated answer and thinking loops. |
| `GUFO_LOOP_BREAKER_EOG_CLOSE` | Other | debug | Close a detected loop with an EOG token. |
| `GUFO_LOOP_BREAKER_TOOL_ARGS` | Behaviour | debug | Stop framing loops inside tool-call arguments. |
| `GUFO_LOOP_BREAKER_TRANSITION` | Other | debug | Select the loop breaker transition mode. |
| `GUFO_LOOP_BREAKER_TRANSITION_TEXT` | Other | debug | Set the text used for loop breaker transitions. |
| `GUFO_MASK_THREADS` | Speed | debug | Set worker threads for tool grammar mask cache misses. |
| `GUFO_MAX_IMAGE_TOKENS` | Memory | user | Limit image tokens admitted for a request. |
| `GUFO_MEM_ACCOUNTING` | Other | debug | Control mem accounting. |
| `GUFO_MODEL_SUPPRESS_TOKENS` | Behaviour | debug | Enable model token suppression. |
| `GUFO_MOE_BENCH` | Other | debug | Control moe bench. |
| `GUFO_MTP_TRACE` | Speed | debug | Write one trace line for each decode cycle. |
| `GUFO_NPU_CHIP_POWER` | Hardware and memory | debug | Log chip power during NPU prefill with literal zero or one. |
| `GUFO_NPU_DIR` | Hardware and memory | debug | Locate NPU artifacts for enabled prefill. |
| `GUFO_NPU_GATE_SLICES` | Hardware and memory | debug | Select zero through twelve paired query and gate slices. |
| `GUFO_NPU_GUARD` | Hardware and memory | debug | Hold NPU prefill during high power or memory pressure with on, off or log. |
| `GUFO_NPU_GUARD_TEST` | Hardware and memory | debug | Set the guard power threshold and hold time for tests as power_w and hold_s. |
| `GUFO_NPU_HAL_MONITOR` | Hardware and memory | debug | Log Windows NPU IOMMU fault events with literal zero or one. |
| `GUFO_NPU_KEEPALIVE` | Hardware and memory | debug | Enable periodic heartbeat with literal zero or one. |
| `GUFO_NPU_KEEPALIVE_MS` | Hardware and memory | debug | Set heartbeat interval to zero or one hundred through sixty thousand milliseconds. |
| `GUFO_NPU_MEM_TRACE` | Hardware and memory | debug | Log NPU buffer page residency and GPU memory with literal zero or one. |
| `GUFO_NPU_SHEXP` | Hardware and memory | debug | Enable shared expert staging with literal zero or one. |
| `GUFO_NPU_SLICES` | Hardware and memory | debug | Select one through twelve linear z slices for enabled prefill. |
| `GUFO_NPU_TEST_FAIL_CHUNK` | Hardware and memory | debug | Report one NPU prefill chunk of each request as invalid, to test the GPU retry. |
| `GUFO_NPU_TIMEOUT_MS` | Hardware and memory | debug | Bound NPU completion to one through eight thousand milliseconds. |
| `GUFO_NPU_WEIGHT_GUARD` | Hardware and memory | debug | Re-send a changed NPU weight buffer with literal zero or one. |
| `GUFO_PLATFORM_TUNING` | Other | debug | Control Windows timing and memory placement switches. |
| `GUFO_PORT` | Server | debug | Set the server port through the environment. |
| `GUFO_PREFILL_DIAG` | Other | debug | Control prefill diag. |
| `GUFO_PREFILL_FORWARD_TOKENS` | Speed | debug | Set the maximum tokens in each prefill forward pass. |
| `GUFO_PREFILL_STEP_LOG` | Other | debug | Control prefill step log. |
| `GUFO_PREFILL_TARGET_MS` | Other | debug | Control prefill target ms. |
| `GUFO_PRODUCTION_SAMPLING` | Other | debug | Enable the production sampling preset. |
| `GUFO_QWEN27_DRAFTLOG` | Other | debug | Control qwen27 profile. |
| `GUFO_QWEN27_FLUSH` | Other | debug | Control qwen27 profile. |
| `GUFO_QWEN27_LOOKUP` | Other | debug | Control qwen27 profile. |
| `GUFO_QWEN27_LOOKUP_MODE` | Other | debug | Control qwen27 profile. |
| `GUFO_QWEN27_PROFILE` | Other | debug | Control qwen27 profile. |
| `GUFO_REASONING_BUDGET` | Other | debug | Enable request reasoning token budgets. |
| `GUFO_REQUIRE_GFX1151` | Other | debug | Control require hip. |
| `GUFO_REQUIRE_GPU` | Other | debug | Control require hip. |
| `GUFO_REQUIRE_HIP` | Other | debug | Control require hip. |
| `GUFO_ROUTED_LANEROW` | Speed | debug | Enable the routed expert prefill kernel. |
| `GUFO_ROUTED_WMMA_IQ` | Speed | debug | Enable the F16 WMMA route for supported IQ and Q2 expert weights. |
| `GUFO_SPEC_EOG_STOP` | Speed | debug | Permit speculative EOG stops. |
| `GUFO_SURVIVAL_FLOOR` | Other | debug | Control survival floor. |
| `GUFO_THINK_EOG_GUARD` | Behaviour | debug | Block end of turn inside an unfinished thinking span. |
| `GUFO_THINK_GUARD_ESCALATE` | Behaviour | debug | Close an unfinished thinking span after this many blocked end-of-turn tries. 0 turns it off. |
| `GUFO_THINK_MARKER_BIAS` | Behaviour | debug | Set a fixed thinking marker probability bias. |
| `GUFO_THINK_MARKER_BIAS_BY_EFFORT` | Behaviour | user | Map reasoning effort to think marker bias. |
| `GUFO_THINK_MARKER_ENTROPY_GATE` | Behaviour | debug | Gate thinking marker bias at a raw top-k entropy threshold in nats. |
| `GUFO_THINK_MARKER_LIST` | Behaviour | debug | Control think marker list. |
| `GUFO_TRACE` | Other | debug | Control trace. |
| `GUFO_WEIGHT_UPLOAD_READERS` | Other | debug | Control weight upload readers. |
| `HOME` | Other | debug | Control home. |
| `HOST` | Server | debug | Set the server address through the environment fallback. |
| `LOCALAPPDATA` | Other | debug | Control localappdata. |
| `PATH` | Other | debug | Select runtime DLL search paths. |
| `PORT` | Server | debug | Set the server port through the environment fallback. |
| `QWEN3_ASR_ARTIFACTS` | Other | debug | Control the qwen3 asr artifacts setting. |
| `QWEN3_ASR_MODEL_ROOT` | Other | debug | Control the qwen3 asr model root setting. |
| `QWEN3_ASR_TEST_WAV` | Other | debug | Control the qwen3 asr test wav setting. |
| `TEMP` | Other | debug | Control the temp setting. |
| `TERM` | Other | debug | Identify the terminal for human status output. |
| `XDG_CACHE_HOME` | Memory | debug | Control xdg cache home. |
