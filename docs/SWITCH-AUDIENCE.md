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

Counts: user 38, advanced 18, debug 220.

| Switch | Group | Audience | Purpose |
| --- | --- | --- | --- |
| `--allow-unmeasured-memory` | Memory | debug | Allow subagent context above the measured token limit. |
| `--api-key` | Server | user | Require an API key from clients. |
| `--cache-disk` | Memory | user | Set the disk prompt cache directory. |
| `--cache-disk-bytes` | Memory | user | Limit disk prompt cache storage. |
| `--cache-disk-staging-bytes` | Memory | user | Limit RAM used to stage disk prompt snapshots. |
| `--cache-ram-bytes` | Memory | user | Set the RAM snapshot cache budget. |
| `--context` | Memory | user | Set the context token limit for each session. |
| `--debug-hooks` | Other | debug | None |
| `--dflash-model` | Other | debug | None |
| `--draft-head-model` | Speed | advanced | Select the prepared MTP draft head. |
| `--draft-policy` | Speed | debug | None |
| `--draft-tokens` | Speed | advanced | Set the maximum speculative draft length. |
| `--dspark-model` | Other | debug | None |
| `--exact` | Other | debug | Select the exact profile. |
| `--frequency-penalty` | Behaviour | user | Penalize tokens by their generated occurrence count. |
| `--help` | Other | debug | None |
| `--host` | Server | user | Set the server listen address. |
| `--kv-cache-type` | Memory | user | Use about half the cache memory, so more or longer conversations fit; replies can differ slightly from the default. |
| `--log-level` | Server | debug | Set the amount of server log detail. |
| `--log-progress` | Server | debug | Log prompt processing progress. |
| `--loop-breaker` | Behaviour | user | Stop repeated answer and thinking loops. |
| `--manifest` | Other | debug | None |
| `--max-buffered-output-bytes` | Server | debug | Limit buffered output for each request. |
| `--max-buffered-output-total` | Server | debug | Limit buffered output across all requests. |
| `--max-connections` | Server | debug | Limit simultaneous HTTP connections. |
| `--max-output-bytes` | Server | debug | Limit output text bytes for each request. |
| `--max-pending` | Server | debug | Limit queued requests. |
| `--max-pending-per-client` | Server | debug | Limit queued requests from each client. |
| `--max-request-bytes` | Server | debug | Set the maximum HTTP request body size. |
| `--max-score-tokens` | Server | debug | None |
| `--max-tokens` | Behaviour | user | Limit new tokens generated per request. |
| `--min-draft-tokens` | Speed | advanced | Set the minimum speculative draft length. |
| `--min-keep` | Behaviour | user | Retain a minimum number of sampling candidates. |
| `--min-p` | Behaviour | user | Filter tokens by probability relative to the best token. |
| `--mmproj` | Memory | user | Set the vision file. |
| `--model` | Other | user | Path of the model file to load. |
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
| `--prefill-mode` | Speed | user | Experimental. Reads prompts of 4,096+ new tokens up to 16 % faster with fewer experts. Decode uses all experts. |
| `--presence-penalty` | Behaviour | user | Penalize tokens already generated in the response. |
| `--preserve-thinking` | Behaviour | debug | Select which thinking history to retain. |
| `--print-effective-settings` | Other | debug | Print resolved settings as JSON and exit. |
| `--profile` | Other | user | Select the release settings profile. |
| `--prompt-lookup` | Speed | advanced | Propose tokens from an earlier conversation match. |
| `--reasoning-effort` | Behaviour | user | Set the default reasoning effort. |
| `--refusal-direction` | Behaviour | user | Turn the direction projection on or off. |
| `--refusal-direction-file` | Behaviour | advanced | Choose the direction file used when the projection is on. |
| `--refusal-direction-layer-end` | Behaviour | advanced | Set the last layer for stream mode, inclusive. Ignored in sites mode. |
| `--refusal-direction-layer-start` | Behaviour | advanced | Set the first layer for stream mode, inclusive. Ignored in sites mode. |
| `--refusal-direction-mode` | Behaviour | advanced | Choose whether to project selected outputs or the residual stream. |
| `--refusal-direction-strength` | Behaviour | advanced | Set the projection strength. 1 removes the direction; 0 has no effect. |
| `--repeat-last-n` | Behaviour | user | Set the recent token window for repetition penalties. |
| `--repeat-penalty` | Behaviour | user | Penalize recently used tokens. |
| `--request-timeout-ms` | Server | debug | Limit request time in milliseconds. |
| `--root` | Other | debug | None |
| `--seed` | Behaviour | user | Set the random sampling seed. |
| `--served-model-name` | Server | user | Set the model identifier exposed by the API. |
| `--sessions` | Memory | user | Set the number of preallocated request sessions. |
| `--speculative` | Speed | advanced | Select the speculative draft backend. |
| `--stall-timeout-ms` | Server | debug | Exit as device lost when generation makes no progress for this many milliseconds. |
| `--temperature` | Behaviour | user | Set the randomness of token sampling. |
| `--think` | Behaviour | user | Set the default reasoning mode. |
| `--think-marker-bias` | Behaviour | user | Uses one rethinking penalty at every reasoning effort, replacing the per-effort values. Negative numbers lower the chance of "wait" restarts. |
| `--top-k` | Behaviour | user | Limit sampling to the highest probability candidates. |
| `--top-p` | Behaviour | user | Filter sampling candidates by cumulative probability. |
| `--trace` | Other | debug | Write request and generation content to a JSONL file. |
| `--ttl` | Other | debug | None |
| `--twin-model` | Hardware and memory | user | Select the prepared dense twin. |
| `--verbose` | Other | debug | None |
| `--verify-model` | Other | advanced | Hash all eight manifest files plus distinct overrides before loading (about 2-4 minutes on SSD). |
| `--voice` | Other | debug | None |
| `--voice-lang` | Other | debug | None |
| `--voice-text` | Other | debug | None |
| `A3B_ATTN_CHECK` | Other | debug | None |
| `A3B_ATTN_PRECISE` | Other | debug | None |
| `A3B_DECODE_ATTN` | Other | debug | None |
| `A3B_DFLASH_PROFILE` | Other | debug | None |
| `A3B_NO_RESUME` | Other | debug | None |
| `A3B_PROBE_ONLY` | Other | debug | None |
| `A3B_WIDE_ATTN_CHECK` | Other | debug | None |
| `A3B_WIDE_BF16_BLAS` | Other | debug | None |
| `A3B_WIDE_BLAS_F32` | Other | debug | None |
| `A3B_WIDE_DEBUG` | Other | debug | None |
| `A3B_WIDE_F16_MAXCOLS` | Other | debug | None |
| `A3B_WIDE_F16_MINROWS` | Other | debug | None |
| `A3B_WIDE_FN_ATTN` | Other | debug | None |
| `A3B_WIDE_NO_DOWN64` | Other | debug | None |
| `A3B_WIDE_NO_F16` | Other | debug | None |
| `A3B_WIDE_PROFILE` | Other | debug | None |
| `BUBO_API_KEY` | Other | debug | Provide the server API key; --api-key takes precedence. |
| `BUBO_MANIFEST` | Other | debug | Select bubo-package.json; explicit --manifest takes precedence. |
| `BUBO_MMPROJ` | Memory | debug | Select the vision projector; explicit --mmproj takes precedence. |
| `BUBO_MTP_MODEL` | Speed | debug | Select the MTP file; explicit --mtp-model takes precedence. |
| `BUBO_RUNTIME_DIR` | Other | debug | Set the prepared runtime folder. |
| `DS4_LOCK_FILE` | Other | debug | None |
| `GPU_MAX_HW_QUEUES` | Other | debug | Set the hardware queue cap. |
| `GUFO_A3B_DATA` | Other | debug | None |
| `GUFO_A3B_MODEL` | Other | debug | None |
| `GUFO_BATCH_DRAFT_HEAD` | Other | debug | None |
| `GUFO_CACHE_HOST_FLOOR_MB` | Memory | debug | Reserve available host RAM during snapshot admission. |
| `GUFO_CACHE_REMOVAL_LOG` | Memory | debug | None |
| `GUFO_DEBUG_HOOKS` | Other | debug | None |
| `GUFO_DECODE_IQ_ROUTING_PATH` | Other | debug | Export narrow-pass expert routing for diagnostic replay. |
| `GUFO_DEEPSEEK_DSPARK_TIMING` | Other | debug | None |
| `GUFO_DEEPSEEK_DSPARK_TRACE` | Other | debug | None |
| `GUFO_DEEPSEEK_DSPARK_WIDTH_TRACE` | Other | debug | None |
| `GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE` | Other | debug | None |
| `GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE` | Other | debug | None |
| `GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE` | Other | debug | None |
| `GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE` | Other | debug | None |
| `GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE` | Other | debug | None |
| `GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE` | Other | debug | None |
| `GUFO_DENSE_IQ_VARIANT` | Speed | debug | Select a diagnostic dense kernel for the process lifetime. |
| `GUFO_DENSE_TWIN_ALLOW_Q8` | Other | debug | None |
| `GUFO_DENSE_TWIN_BATCH` | Hardware and memory | debug | Enable dense twin decode with multiple sessions. |
| `GUFO_DENSE_TWIN_MAX_N` | Hardware and memory | debug | Limit the dense twin batch width. |
| `GUFO_DENSE_TWIN_MODEL` | Hardware and memory | debug | Select the four bit dense weights used for decode and verify. |
| `GUFO_DENSE_TWIN_SHADOW` | Other | debug | None |
| `GUFO_DENSE_TWIN_TRACE` | Other | debug | None |
| `GUFO_DENSE_W1` | Speed | debug | Enable the production dense vector kernel for single-row projections. |
| `GUFO_DEVICE_ACCEPT` | Speed | debug | Queue greedy speculative rollback before reading acceptance on the host. |
| `GUFO_DISK_BOUNDARY_ALIGN` | Memory | debug | None |
| `GUFO_DISK_DEFER` | Memory | debug | Defer disk writes until idle. |
| `GUFO_DISK_DEFER_MS` | Memory | debug | None |
| `GUFO_DISK_EVICT_BACKLOG_BYTES` | Memory | debug | None |
| `GUFO_DISK_EVICT_DOMINATED` | Memory | debug | Skip snapshots covered by a longer retained snapshot. |
| `GUFO_DISK_EVICT_SKIP_TOKENS` | Memory | debug | Skip snapshot writes close to an existing persisted prefix. |
| `GUFO_DISK_IDLE_MS` | Memory | debug | None |
| `GUFO_DISK_INPASS` | Memory | debug | Enable disk snapshots during prefill passes. |
| `GUFO_DISK_PERSIST` | Memory | debug | Persist disk snapshots on eviction after idle. |
| `GUFO_DISK_SLICE` | Memory | debug | None |
| `GUFO_DISPATCH_RESERVE_MIB` | Memory | debug | Set the device heap reserve in MiB. |
| `GUFO_DISPATCH_TELEMETRY` | Other | debug | None |
| `GUFO_DRAFT_CHAIN` | Speed | debug | Run a fixed greedy draft depth in one captured graph. |
| `GUFO_DRAFT_FALSIFY` | Other | debug | Replace draft proposals with deterministic incorrect tokens for verifier checks. |
| `GUFO_DRAFT_FALSIFY_CYCLES` | Other | debug | Limit the number of falsified draft cycles in each request. |
| `GUFO_DRAFT_OUT_MODEL` | Speed | debug | Select the four bit output head used for MTP drafts. |
| `GUFO_DRAFT_POLICY` | Speed | debug | Select the MTP draft length policy. |
| `GUFO_DRAFT_VOCAB_AUTO` | Speed | debug | Set thresholds for automatic draft vocabulary selection. |
| `GUFO_EOG_BOUNDARY` | Behaviour | debug | Permit EOG tokens at allowed boundaries. |
| `GUFO_EOG_DEFER_P` | Behaviour | debug | Set the deferred stop probability. |
| `GUFO_EOG_MIN_PROB` | Behaviour | debug | Set the minimum EOG probability required to stop. |
| `GUFO_EOG_MIN_PROB_EXACT` | Behaviour | debug | Compute EOG probability over the full vocabulary. |
| `GUFO_EVAL_API_KEY` | Other | debug | None |
| `GUFO_EVAL_ARTIFACT_ID` | Other | debug | None |
| `GUFO_EVAL_MODEL_REVISION` | Other | debug | None |
| `GUFO_EVAL_SERVER_COMMIT` | Other | debug | None |
| `GUFO_EVAL_SERVER_CONFIG` | Other | debug | None |
| `GUFO_EVAL_SERVER_LABEL` | Other | debug | None |
| `GUFO_EXPERT_OVERLAY_KEEP` | Memory | debug | Name the routed-expert tensors that keep the base model's weights when an overlay is set. |
| `GUFO_EXPERT_OVERLAY_MODEL` | Memory | debug | Replace the routed experts of the model with those of another GGUF of the same architecture. |
| `GUFO_EXPERT_V2K` | Speed | debug | Grouped expert kernel (decode). |
| `GUFO_FFMPEG` | Other | debug | None |
| `GUFO_FFPROBE` | Other | debug | None |
| `GUFO_GROUPED_DOWN` | Other | debug | None |
| `GUFO_HC_BF16_DENSE` | Speed | debug | Use the narrow BF16 hyper-connection projection kernel. |
| `GUFO_HC_INJECT_BF16_FUSED` | Speed | debug | Fuse BF16 hyper-connection inject weights into the scalar decode epilogue. |
| `GUFO_HC_LOAD_Q8` | Speed | debug | Narrow BF16 hyper-connection weights to Q8_0 in memory while loading. |
| `GUFO_HC_LOAD_Q8_POISON` | Speed | debug | Poison buffers the weight loader releases and keep device ones allocated to expose stale reads. |
| `GUFO_HC_LOAD_Q8_VERIFY` | Speed | debug | Compare converted and draft-block weights on the device with the file after loading. |
| `GUFO_HEAD_CHECK` | Other | debug | Compare shortlisted head results against the full output head. |
| `GUFO_HEAD_SHORTLIST` | Speed | debug | Opt in to experimental shortlist scoring and exact rescoring for eligible output head calls; off by default. |
| `GUFO_HIPBLASLT_PLAN_CACHE` | Other | debug | None |
| `GUFO_HOST` | Server | debug | Set the server address through the environment. |
| `GUFO_INCREMENTAL_TOKENIZE` | Speed | debug | Reuse verified text render and token prefixes for exact conversation appends. Inspect incremental_tokenize counters in render timeline events. |
| `GUFO_IQ4_WIDE_VARIANT` | Other | debug | None |
| `GUFO_IQ4_WMMA_WAVES` | Other | debug | None |
| `GUFO_IQ_ROUTE_MIN_ROWS` | Speed | debug | Set the minimum forward-pass rows for the IQ expert WMMA route. |
| `GUFO_LENIENT` | Speed | debug | Accept near miss MTP drafts inside thinking spans. |
| `GUFO_LENIENT_LAMBDA` | Speed | debug | Set the lenient speculative acceptance bound. |
| `GUFO_LENIENT_THINK_ONLY` | Speed | debug | Limit lenient acceptance to thinking output. |
| `GUFO_LOGIT_BIAS` | Behaviour | debug | Enable request token probability adjustments. |
| `GUFO_LOGIT_EVAL_DUMP` | Other | debug | Write head inputs and full logits during logit-eval. |
| `GUFO_LOG_SYNC` | Other | debug | Write logs synchronously for crash diagnostics. |
| `GUFO_LOOP_BREAKER` | Behaviour | debug | Stop repeated answer and thinking loops. |
| `GUFO_LOOP_BREAKER_EOG_CLOSE` | Other | debug | Close a detected loop with an EOG token. |
| `GUFO_LOOP_BREAKER_TOOL_ARGS` | Behaviour | debug | Stop framing loops inside tool-call arguments. |
| `GUFO_LOOP_BREAKER_TRANSITION` | Other | debug | Select the loop breaker transition mode. |
| `GUFO_LOOP_BREAKER_TRANSITION_TEXT` | Other | debug | Set the text used for loop breaker transitions. |
| `GUFO_MASK_THREADS` | Speed | debug | Set worker threads for tool grammar mask cache misses. |
| `GUFO_MAX_IMAGE_TOKENS` | Memory | user | Limit image tokens admitted for a request. |
| `GUFO_MEM_ACCOUNTING` | Other | debug | None |
| `GUFO_MODEL_SUPPRESS_TOKENS` | Behaviour | debug | Enable model token suppression. |
| `GUFO_MOE_BENCH` | Other | debug | None |
| `GUFO_MOE_FUSION` | Speed | debug | Combine narrow expert output and the residual update in one launch. |
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
| `GUFO_PLATFORM_TUNING` | Other | debug | None |
| `GUFO_PLE_READAHEAD` | Speed | debug | Start PLE table reads when anchor and draft token IDs become available. |
| `GUFO_PORT` | Server | debug | Set the server port through the environment. |
| `GUFO_PREFILL_DIAG` | Other | debug | None |
| `GUFO_PREFILL_EXPERTS` | Speed | debug | Set how many routed experts each token keeps during prompt prefill. |
| `GUFO_PREFILL_EXPERTS_AUDIT` | Speed | debug | Select how much the prefill expert cut records. |
| `GUFO_PREFILL_EXPERTS_FOLD` | Speed | debug | Set the file of per-layer scalars for the fold prefill expert mode. |
| `GUFO_PREFILL_EXPERTS_KFILE` | Speed | debug | Set the file of per-layer expert counts for prefill. |
| `GUFO_PREFILL_EXPERTS_MASS` | Speed | debug | Set the routing mass that mass mode keeps in each prefill token. |
| `GUFO_PREFILL_EXPERTS_MODE` | Speed | debug | Select how the prefill expert cut rescales the kept experts. |
| `GUFO_PREFILL_EXPERTS_TAIL` | Speed | debug | Set how many final prompt tokens keep all experts in prefill. |
| `GUFO_PREFILL_FORWARD_TOKENS` | Speed | debug | Set the maximum tokens in each prefill forward pass. |
| `GUFO_PREFILL_NATIVE_GEMM` | Speed | debug | Run native GGUF dense layers through a blocked matrix GEMM in prefill: 1 uses int8, 2 (default) adds 16-bit K-quants. |
| `GUFO_PREFILL_STAGEB` | Other | debug | Write prefill expert samples to a directory for calibration. |
| `GUFO_PREFILL_STAGEB_LAYERS` | Other | debug | Select the layers that the prefill expert samples cover. |
| `GUFO_PREFILL_STAGEB_ROWS` | Other | debug | Set the token interval between prefill expert samples. |
| `GUFO_PREFILL_STAGEB_STRIDE` | Other | debug | Set the token interval between accumulated prefill expert statistics. |
| `GUFO_PREFILL_STEP_LOG` | Other | debug | None |
| `GUFO_PREFILL_TARGET_MS` | Other | debug | None |
| `GUFO_PREFILL_TELEMETRY` | Other | debug | Write prefill routing telemetry to a directory. |
| `GUFO_PREFILL_TELEMETRY_LAYERS` | Other | debug | Select the layers that the prefill telemetry covers. |
| `GUFO_PRODUCTION_SAMPLING` | Other | debug | Enable the production sampling preset. |
| `GUFO_QWEN27_DRAFTLOG` | Other | debug | None |
| `GUFO_QWEN27_FLUSH` | Other | debug | None |
| `GUFO_QWEN27_LOOKUP` | Other | debug | None |
| `GUFO_QWEN27_LOOKUP_MODE` | Other | debug | None |
| `GUFO_QWEN27_PROFILE` | Other | debug | None |
| `GUFO_REASONING_BUDGET` | Other | debug | Enable request reasoning token budgets. |
| `GUFO_REFUSAL_DIRECTION` | Behaviour | debug | Remove a configured refusal direction from selected model outputs. |
| `GUFO_REFUSAL_DIRECTION_ENABLED` | Behaviour | debug | Enable or disable the direction projection. |
| `GUFO_REFUSAL_DIRECTION_LAYER_END` | Behaviour | debug | Choose the last stream layer. |
| `GUFO_REFUSAL_DIRECTION_LAYER_START` | Behaviour | debug | Choose the first stream layer. |
| `GUFO_REFUSAL_DIRECTION_MODE` | Behaviour | debug | Choose the projection mode. |
| `GUFO_REFUSAL_DIRECTION_STRENGTH` | Behaviour | debug | Override the fitted direction strength. |
| `GUFO_REQUIRE_GFX1151` | Other | debug | None |
| `GUFO_REQUIRE_GPU` | Other | debug | None |
| `GUFO_REQUIRE_HIP` | Other | debug | None |
| `GUFO_ROUTED_LANEROW` | Speed | debug | Enable the routed expert prefill kernel. |
| `GUFO_ROUTED_WMMA_IQ` | Speed | debug | Enable the F16 WMMA route for supported IQ and Q2 expert weights. |
| `GUFO_SPEC_EOG_STOP` | Speed | debug | Permit speculative EOG stops. |
| `GUFO_SURVIVAL_FLOOR` | Other | debug | None |
| `GUFO_THINK_EOG_GUARD` | Behaviour | debug | Block end of turn inside an unfinished thinking span. |
| `GUFO_THINK_GUARD_ESCALATE` | Behaviour | debug | Close an unfinished thinking span after this many blocked end-of-turn tries. 0 turns it off. |
| `GUFO_THINK_MARKER_BIAS` | Behaviour | debug | Uses one rethinking penalty at every reasoning effort. Negative numbers lower the chance of "wait" and "let me rethink" restarts. |
| `GUFO_THINK_MARKER_BIAS_BY_EFFORT` | Behaviour | user | Lowers the chance of "wait" and "let me rethink" restarts while the model thinks. It does not cut thinking off. |
| `GUFO_THINK_MARKER_ENTROPY_GATE` | Behaviour | debug | Gate thinking marker bias at a raw top-k entropy threshold in nats. |
| `GUFO_THINK_MARKER_LIST` | Behaviour | debug | None |
| `GUFO_THINK_NUDGE` | Behaviour | debug | Insert an unfinished reasoning prefix when the model attempts to end an open thinking span. |
| `GUFO_THINK_NUDGE_MAX` | Behaviour | debug | Limit thinking stop nudge insertions per request. Zero disables insertions. |
| `GUFO_THINK_NUDGE_TEXT` | Behaviour | debug | Set the unfinished thinking stop nudge prefix, tokenized without special tokens. |
| `GUFO_TRACE` | Other | debug | None |
| `GUFO_TRACE_STAGES` | Other | debug | Trace decode stages inside verification graphs. |
| `GUFO_WEIGHT_UPLOAD_READERS` | Other | debug | None |
| `HOME` | Other | debug | None |
| `HOST` | Server | debug | Set the server address through the environment fallback. |
| `LOCALAPPDATA` | Other | debug | None |
| `PATH` | Other | debug | Select runtime DLL search paths. |
| `PORT` | Server | debug | Set the server port through the environment fallback. |
| `QWEN3_ASR_ARTIFACTS` | Other | debug | None |
| `QWEN3_ASR_MODEL_ROOT` | Other | debug | None |
| `QWEN3_ASR_TEST_WAV` | Other | debug | None |
| `TEMP` | Other | debug | None |
| `TERM` | Other | debug | Identify the terminal for human status output. |
| `XDG_CACHE_HOME` | Memory | debug | None |
