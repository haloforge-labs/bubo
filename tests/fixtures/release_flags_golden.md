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
| `--cache-ram-bytes` | Set the RAM snapshot cache budget. | 6442450944 | Keep the default. Change it only to fit the available host RAM. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--context` | Set the context token limit for each session. | 262144 | Follow-up: drop this default if the model's native context equals 262144. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--draft-tokens` | Control draft tokens. | 7 | Keep the default. Change it only to test a different speculative draft length. | any; generic; qwen3.8-flash-next |
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
| `--max-request-bytes` | Set the maximum HTTP request body size. | 67108864 | Keep the default. Increase it only for larger request bodies. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--max-tokens` | Control max tokens. | -1 | Inventory classification. | any; generic; any |
| `--min-draft-tokens` | Control min draft tokens. | 1 | Keep the default. Change it only to test the minimum speculative draft length. | any; generic; qwen3.8-flash-next |
| `--min-keep` | Control min keep. | 0 | Inventory classification. | any; generic; any |
| `--min-p` | Control min p. | 0.0 | Inventory classification. | any; generic; any |
| `--mmproj` | Control mmproj. | Unset until the manifest job | Set the matching vision sidecar when the model must process images. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-draft-vocab` | Select the MTP draft head vocabulary. | auto | Keep auto for one session and latin for subagents. Change it only for draft vocabulary tests. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-model` | Control mtp model. | Unset until the manifest job | Set the matching MTP sidecar to enable speculative decoding. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--mtp-policy` | Select the MTP draft length controller. | survival | Keep survival. Change it only for draft acceptance tests. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--no-behaviour` | Disable the eight behaviour defaults. | Unset | Select the required release policy. | any; generic; any |
| `--no-cache` | Disable RAM and disk prompt caches for all requests. | Off; exact enables it | Enable it for cold prefill and reproducible output. | any; generic; qwen3.8-flash-next |
| `--no-cache-disk` | Disable the disk snapshot cache. | Unset | Select the required release policy. | any; generic; any |
| `--no-draft-head` | Suppress the manifest draft output head. | Unset | Select the required release policy. | any; generic; any |
| `--no-mtp` | Disable MTP defaults and the speculative backend. | Unset | Select the required release policy. | any; generic; any |
| `--no-prompt-lookup` | Disable prompt lookup proposals. | Unset | Select the required release policy. | any; generic; any |
| `--no-twin` | Suppress the manifest dense twin. | Unset | Select the required release policy. | any; generic; any |
| `--no-vision` | Suppress the manifest vision sidecar. | Unset | Select the required release policy. | any; generic; any |
| `--port` | Control port. | 8080 | Inventory classification. | any; generic; any |
| `--prefill-chunk` | Limit prompt tokens between active decode rounds. | 512 | Larger chunks increase pauses for other active sessions. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--presence-penalty` | Control presence penalty. | preset (non-thinking 1.5 unless GUFO_PRODUCTION_SAMPLING=1) | Inventory classification. | any; generic; any |
| `--preserve-thinking` | Control preserve thinking. | auto | Keep auto. Change it only when the client needs different thinking history retention. | any; generic; qwen3.8-flash-next |
| `--print-effective-settings` | Print resolved settings as JSON and exit. | Unset | Select the required release policy. | any; generic; any |
| `--profile` | Select the release settings profile. | Unset | Select the required release policy. | any; generic; any |
| `--prompt-lookup` | Propose tokens from an earlier conversation match. | 1 | Keep it enabled with MTP. Disable it only for prompt lookup comparisons. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--reasoning-effort` | Set the default reasoning effort. | medium | Medium selects a think-marker bias of -1 in the production map. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--repeat-last-n` | Control repeat last n. | 1.0 / 64 / 0.0 (off) | Inventory classification. | any; generic; any |
| `--repeat-penalty` | Control repeat penalty. | 1.0 / 64 / 0.0 (off) | Inventory classification. | any; generic; any |
| `--request-timeout-ms` | Control request timeout ms. | 16 / =max-pending / 0 / 1 MiB / 64 KiB / 256 KiB | Inventory classification. | any; generic; any |
| `--seed` | Control seed. | -1 random | Inventory classification. | any; generic; any |
| `--served-model-name` | Control served model name. | none | Inventory classification. | any; generic; any |
| `--sessions` | Set the number of preallocated request sessions. | 1 | Keep one session. Use subagents when four requests must run concurrently. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--speculative` | Select the speculative draft backend. | mtp | Keep MTP when its sidecar is available. Use off for plain autoregressive decoding. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--temperature` | Control temperature. | preset (Qwen3.8: 1.0 thinking, 0.7 non-thinking) | Inventory classification. | any; generic; any |
| `--think` | Set the default reasoning mode. | on | Keep on. Change it only when the task does not need reasoning. | any; strix-halo-tuned; qwen3.8-flash-next |
| `--top-k` | Control top k. | preset 20 (Qwen3.8) | Inventory classification. | any; generic; any |
| `--top-p` | Control top p. | preset (0.95 thinking; 0.8 non-thinking unless GUFO_PRODUCTION_SAMPLING=1) | Inventory classification. | any; generic; any |
| `GUFO_CACHE_HOST_FLOOR_MB` | Control cache host floor mb. | 0 (off) | Inventory classification. | any; generic; any |
| `GUFO_DENSE_TWIN_BATCH` | Enable dense twin decode with multiple sessions. | off | Subagents enables the measured batched twin path. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_MAX_N` | Control dense twin max n. | 8 | Keep the default. Change it only for batched twin tests outside the exact profile. | any; generic; qwen3.8-flash-next |
| `GUFO_DENSE_TWIN_MODEL` | Select the four bit dense weights used for decode and verify. | Unset until the manifest job | Set the matching dense twin for faster decoding outside the exact profile. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_DISK_DEFER` | Defer disk writes until idle. | Unset; 1 with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_EVICT_DOMINATED` | Skip snapshots covered by a longer retained snapshot. | Unset; 1 with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_EVICT_SKIP_TOKENS` | Skip snapshot writes close to an existing persisted prefix. | Unset; 8192 with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DISK_INPASS` | Control disk inpass. | 1 | The built-in value needs no injection. | any; generic; any |
| `GUFO_DISK_PERSIST` | Persist disk snapshots on eviction after idle. | Unset; evict_idle with --cache-disk | Use the gated disk cache bundle. | any; generic; qwen3.8-flash-next |
| `GUFO_DRAFT_OUT_MODEL` | Select the four bit output head used for MTP drafts. | Unset until the manifest job | Set the matching draft head for faster MTP decoding outside the exact profile. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_DRAFT_VOCAB_AUTO` | Control draft vocab auto. | 2,16,64 (2 of last 16 non-latin; back after 64 latin) | Keep the default. Change it only to test automatic draft vocabulary thresholds. | any; generic; qwen3.8-flash-next |
| `GUFO_EOG_BOUNDARY` | Permit EOG tokens at allowed boundaries. | 1 | Keep it enabled. Disable it only to compare end token boundary handling. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_DEFER_P` | Set the deferred stop probability. | 0.9 | Keep the default. Change it only to test early end token deferral. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_MIN_PROB` | Set the minimum EOG probability required to stop. | 0.2 | Keep the default. Change it only to test the minimum end token probability. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_EOG_MIN_PROB_EXACT` | Compute EOG probability over the full vocabulary. | Off | Use the full vocabulary EOG probability; production 0.2 uses the top-20 bound. | any; generic; qwen3.8-flash-next |
| `GUFO_HOST` | Control host. | - | Inventory classification. | any; generic; any |
| `GUFO_LENIENT` | Accept near miss MTP drafts inside thinking spans. | bounded | Bounded acceptance changes the sampling distribution. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LENIENT_LAMBDA` | Control lenient lambda. | 0.5 | The built-in value needs no injection. | any; generic; qwen3.8-flash-next |
| `GUFO_LENIENT_THINK_ONLY` | Control lenient think only. | on | Keep the default. Disable it only to test lenient acceptance outside thinking. | any; generic; qwen3.8-flash-next |
| `GUFO_LOGIT_BIAS` | Control logit bias. | off | Inventory classification. | any; generic; any |
| `GUFO_LOOP_BREAKER` | Stop repeated answer and thinking loops. | 1 | Keep it enabled. Disable it only to compare repeated text handling. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LOOP_BREAKER_EOG_CLOSE` | Close a detected loop with an EOG token. | 1 | Keep it enabled. Disable it only to test loop termination without closing the response. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_LOOP_BREAKER_TRANSITION` | Control loop breaker transition. | 2 | The built-in value needs no injection. | any; generic; any |
| `GUFO_LOOP_BREAKER_TRANSITION_TEXT` | Control loop breaker transition text. | built-in text | Inventory classification. | any; generic; any |
| `GUFO_MAX_IMAGE_TOKENS` | Control max image tokens. | no cap | Keep the default. Set a limit only to bound image token admission. | any; generic; qwen3.8-flash-next |
| `GUFO_MODEL_SUPPRESS_TOKENS` | Control model suppress tokens. | off | Keep it disabled. Enable it only when model token suppression is required. | any; generic; qwen3.8-flash-next |
| `GUFO_PORT` | Control port. | - | Inventory classification. | any; generic; any |
| `GUFO_PREFILL_FORWARD_TOKENS` | Set the maximum tokens in each prefill forward pass. | 4096 | Keep 4096 in exact. Change it only outside exact for prefill geometry tests. | any; generic; qwen3.8-flash-next |
| `GUFO_PRODUCTION_SAMPLING` | Control production sampling. | off | Keep it disabled. Enable it only for production sampling policy tests. | any; generic; qwen3.8-flash-next |
| `GUFO_REASONING_BUDGET` | Control reasoning budget. | off | Keep it disabled. Enable it only when requests must enforce a reasoning token budget. | any; generic; qwen3.8-flash-next |
| `GUFO_ROUTED_LANEROW` | Enable the routed expert prefill kernel. | 1 | Keep it enabled. Disable it only for routed expert kernel comparisons. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_SPEC_EOG_STOP` | Permit speculative EOG stops. | 1 | Keep it enabled. Disable it only to compare speculative end token handling. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_EOG_GUARD` | Block end of turn inside an unfinished thinking span. | 1 | Keep it enabled. Disable it only to compare thinking termination. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_GUARD_ESCALATE` | Close an unfinished thinking span after this many blocked end-of-turn tries. 0 turns it off. | 16 | Keep 16. Change it only to test the end token streak before thinking guard escalation. | any; strix-halo-tuned; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_BIAS` | Control think marker bias. | 0 | Keep it unset. Set it only for a fixed thinking marker bias. | any; generic; qwen3.8-flash-next |
| `GUFO_THINK_MARKER_BIAS_BY_EFFORT` | Map reasoning effort to think marker bias. | low:-2,medium:-1,xhigh:0 | Keep the default map. Change it only to tune thinking marker bias for a reasoning effort. | any; strix-halo-tuned; qwen3.8-flash-next |
| `HOST` | Control host. | Unset | Use the environment fallback for the server address. | any; generic; any |
| `PORT` | Control port. | Unset | Use the environment fallback for the server address. | any; generic; any |

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
Output is identical to plain autoregressive decoding of that build.
Output is not identical across builds or to other engines.
Exact disables speculation, behaviour policies, lenient acceptance, the dense twin, the draft head, and RAM and disk prompt caches.
Exact uses one session, forward size 4096, and seed 1 unless a seed is supplied.
Exact refuses values that enable disabled settings or change its session count or prefill geometry.
It runs at about half the decode speed, and every request prefills its full prompt.
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
