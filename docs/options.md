# Bubo options

Use the `default` profile for normal use on a 70 W Strix Halo laptop.
The defaults below apply to Qwen3.8 Flash Next.
Bubo is tuned for Qwen3.8 Flash Next. The other models run with upstream gufo settings.

The tables list current user-facing release options.
Diagnostic and internal switches are omitted.
Explicit flags and real environment values override profile defaults.
Exact refuses incompatible overrides.
Use the [profiles page](profiles.md) for profile-specific values.

## Model files

| Option | What it does | Default and optimal setting | When to change |
| --- | --- | --- | --- |
| `--model PATH` (`-m`) | Select the main GGUF model. | Required. Use the first `UD-Q4_K_XL` shard. | Select another supported model when the task needs it. |

The loader finds the other shards in the same folder.
Supply the matching sidecar files for the optimal decode setup.
The current source accepts MTP and vision paths as flags.
After `bubo prepare`, the model manifest supplies the twin, draft head, MTP predictor and vision projector.
It also accepts twin and draft-head paths through environment variables.
The `--no-twin` and `--no-draft-head` flags suppress defaults.
They do not clear real environment paths.
Remove those environment paths when you need to omit their files.
Exact refuses them while they are set.

## Command-line options

These are options for `bubo serve llm`.
Flag rows without a value take no argument.
Use `bubo serve llm --help` for command-specific help.

| Option | What it does | Default | When to change |
| --- | --- | --- | --- |
| `--allow-unmeasured-memory` | Allow subagent context above the measured token limit. | Unset | Keep it off unless the necessary context exceeds the measured memory limit. |
| `--api-key VALUE` | Make clients supply an API key. | Unset | Set a key when clients must authenticate to the server. |
| `--cache-disk VALUE` | Set the disk prompt cache directory. | Off | Set a directory when prompt snapshots must persist across server restarts. |
| `--cache-disk-bytes VALUE` | Limit disk prompt cache storage. | 8589934592 (8 GiB) | Keep 8 GiB unless the disk cache must retain more prompt snapshots. |
| `--cache-disk-staging-bytes VALUE` | Limit RAM used to stage disk prompt snapshots. | 0 (automatic). At most the disk budget and one eighth of available RAM. | Keep the automatic limit unless disk persistence needs a different RAM limit. |
| `--cache-ram-bytes VALUE` | Set the RAM snapshot cache budget. | 6442450944 (6 GiB). Zero means automatic. | Keep 6 GiB. Use the subagents profile for 4 GiB. Reduce the budget if host RAM is limited. |
| `--context VALUE` | Set the context token limit for each session. | 262144 | Keep the profile context unless the task needs a different token limit. |
| `--draft-tokens VALUE` | Set the maximum speculative draft length. | 7 | Keep seven tokens unless speculative draft length tests need another value. |
| `--exact` | Select the exact profile. | Unset | Use it when output must repeat with the same build, prompt, and seed. |
| `--frequency-penalty VALUE` | Penalise tokens by their generated occurrence count. | 0.0 (off) | Keep zero unless the task needs fewer frequently repeated tokens. |
| `--host VALUE` | Set the server listen address. | 127.0.0.1 | Keep the local address unless remote clients must connect to the server. |
| `--log-level VALUE` | Set the amount of server log detail. | info | Keep info unless diagnosis needs debug messages or normal operation needs fewer messages. |
| `--log-progress` | Log prompt processing progress. | off | Keep it off unless prompt processing progress must appear in the log. |
| `--max-buffered-output-bytes VALUE` | Limit buffered output for each request. | 65536 (64 KiB) | Keep 64 KiB unless slow clients need more buffered output per request. |
| `--max-buffered-output-total VALUE` | Limit buffered output across all requests. | 262144 (256 KiB) | Keep 256 KiB unless concurrent slow clients need more total buffered output. |
| `--max-connections VALUE` | Limit simultaneous HTTP connections. | 16 | Keep 16 unless the client workload needs more simultaneous HTTP connections. |
| `--max-output-bytes VALUE` | Limit output text bytes for each request. | 1048576 (1 MiB) | Keep 1 MiB unless a request must return more output text. |
| `--max-pending VALUE` | Limit queued requests. | 16 | Keep 16 unless the workload needs a larger request queue. |
| `--max-pending-per-client VALUE` | Limit queued requests from each client. | Same as --max-pending | Keep the queue limit unless one client must have a smaller request allowance. |
| `--max-request-bytes VALUE` | Set the maximum HTTP request body size. | 67108864 | Keep 64 MiB unless larger request bodies are necessary. |
| `--max-tokens VALUE` | Limit new tokens generated per request. | -1 | Keep unlimited generation unless responses need a fixed new token limit. |
| `--min-draft-tokens VALUE` | Set the minimum speculative draft length. | 1 | Keep one token unless speculative draft tests need a larger minimum. |
| `--min-keep VALUE` | Retain a minimum number of sampling candidates. | 0 | Keep zero unless probability filters must retain a minimum number of candidates. |
| `--min-p VALUE` | Filter tokens by probability relative to the best token. | 0.0 | Keep zero unless the task needs minimum probability filtering. |
| `--mmproj VALUE` | Set the vision sidecar file. | Unset | Set the matching vision sidecar when the model must process images. |
| `--mtp-draft-vocab auto\|full\|latin` | Select the MTP draft head vocabulary. | Auto with MTP. Otherwise unused. | Keep the profile vocabulary unless draft vocabulary tests need another setting. |
| `--mtp-model VALUE` | Set the MTP sidecar file. | Unset | Set the matching MTP sidecar when speculative decoding is necessary. |
| `--mtp-policy survival\|length` | Select the MTP draft length controller. | Survival with MTP. Otherwise unused. | Keep survival unless draft acceptance tests need the length policy. |
| `--no-behavior` | Turn off the nine behavior defaults. | Off. Exact enables it. | Keep policies enabled unless plain sampling is necessary outside the exact profile. |
| `--no-cache` | Turn off RAM and disk prompt caches for all requests. | Off. Exact enables it. | Keep caches on for normal use. Use exact when every request must process its full prompt. |
| `--no-cache-disk` | Turn off the disk snapshot cache. | Unset | Use it when the supplied disk cache directory must be ignored. |
| `--no-draft-head` | Omit a default Q4_K draft head. Explicit environment paths still apply. | Off. Exact refuses a set draft-head path. | Use it to suppress a default file. Remove explicit environment paths separately. |
| `--no-mtp` | Turn off MTP defaults and speculative decode. | Unset | Use it when decoding must run without MTP. |
| `--no-prompt-lookup` | Turn off prompt lookup proposals. | Unset | Keep prompt lookup enabled with MTP unless lookup comparisons need it off. |
| `--no-twin` | Omit a default dense twin. Explicit environment paths still apply. | Off. Exact refuses a set twin path. | Use it to suppress a default file. Remove explicit environment paths separately. |
| `--no-vision` | Omit the vision sidecar. | Unset | Use it when the server must omit the vision sidecar. |
| `--npu-prefill off\|on` | Use the Ryzen AI NPU for prompt processing when it is available. | On when usable. Exact and stock force it off. | Use `off` if the NPU driver gives problems. See [NPU prefill](npu.md). |
| `--port VALUE` | Set the server listen port. | 8080 | Keep 8080 unless another service uses that port or clients need a different port. |
| `--prefill-chunk VALUE` | Limit prompt tokens between active decode rounds. | 512 | Keep 512 unless active sessions need shorter pauses during another request prefill. |
| `--presence-penalty VALUE` | Penalise tokens already generated in the response. | Qwen3.8: 0.0 thinking. 1.5 non-thinking unless GUFO_PRODUCTION_SAMPLING=1 | Keep the model preset unless the task needs a different penalty for previously generated tokens. |
| `--preserve-thinking auto\|on\|off` | Select which thinking history to retain. | auto | Keep auto unless the client needs a different policy for retained thinking history. |
| `--print-effective-settings` | Print resolved settings as JSON and exit. | Unset | Use it when resolved settings must be inspected without loading a model. |
| `--profile default\|stock\|exact\|subagents` | Select the release settings profile. | default | Keep default for normal use. Use stock, exact, or subagents for their documented tasks. |
| `--prompt-lookup` | Propose tokens from an earlier conversation match. | On with MTP. Otherwise off. | Keep it enabled with MTP unless prompt lookup comparisons need it off. |
| `--reasoning-effort auto\|minimal\|low\|medium\|high\|xhigh\|max` | Set the default reasoning effort. | medium | Keep medium. Use low for shorter reasoning or xhigh for difficult tasks. |
| `--repeat-last-n VALUE` | Set the recent token window for repetition penalties. | 64 | Keep 64 unless repetition penalties need a different recent token window. |
| `--repeat-penalty VALUE` | Penalise recently used tokens. | 1.0 (off) | Keep one unless the task needs a penalty for recent repeated tokens. |
| `--request-timeout-ms VALUE` | Limit request time in milliseconds. | 0 (off) | Keep zero unless requests must stop after a fixed time limit. |
| `--seed VALUE` | Set the random sampling seed. | -1 (random). Exact uses 1. | Use a nonnegative seed for reproducibility. Exact uses 1 unless you supply another nonnegative seed. |
| `--served-model-name VALUE` | Set the model identifier exposed by the API. | Unset (uses model metadata name) | Set a name when the client needs a fixed model identifier. |
| `--sessions VALUE` | Set the number of preallocated request sessions. | 1 | Keep one for exact output or select subagents when four requests must run concurrently. |
| `--speculative mtp\|off\|dflash2\|dspark` | Select the speculative draft backend. | MTP with a predictor. Otherwise off. | Keep MTP with the matching predictor. Use off for plain autoregressive decode. Other backends serve other model families. |
| `--temperature VALUE` | Set the randomness of token sampling. | preset (Qwen3.8: 1.0 thinking, 0.7 non-thinking) | Keep the model preset unless the task needs different randomness or greedy decoding with zero. |
| `--think on\|off\|auto` | Set the default reasoning mode. | on | Keep on. Use off when the task does not need reasoning. |
| `--top-k VALUE` | Limit sampling to the highest probability candidates. | preset 20 (Qwen3.8) | Keep the model preset unless the task needs a different sampling candidate limit. |
| `--top-p VALUE` | Filter sampling candidates by cumulative probability. | preset (0.95 thinking. 0.8 non-thinking unless GUFO_PRODUCTION_SAMPLING=1) | Keep the model preset unless the task needs a different cumulative probability limit. |

## Environment options

Keep profile values for normal use.
These names use the `GUFO_*` prefix where shown.
The `stock` profile does not clear existing environment values.
Set a value in PowerShell before the server command.

```powershell
$env:GUFO_DENSE_TWIN_MODEL = "D:\Models\Qwen3.8-Flash-Next\iq4-nofullattn-nohc.gguf"
```

| Option | What it does | Default | When to change |
| --- | --- | --- | --- |
| `GUFO_CACHE_HOST_FLOOR_MB` | Reserve available host RAM during snapshot admission. | 0 (off) | Keep zero unless snapshot admission must reserve a minimum amount of available host RAM. |
| `GUFO_DENSE_TWIN_BATCH` | Enable dense twin decode with multiple sessions. | off | Keep the profile setting unless batched twin comparisons need a different setting. |
| `GUFO_DENSE_TWIN_MAX_N` | Limit the dense twin batch width. | 8 | Keep eight unless batched twin tests outside exact need a different batch limit. |
| `GUFO_DENSE_TWIN_MODEL` | Select the four bit dense weights used for decode and verify. | No file set. Optimal: matching dense twin outside exact. | Set the matching dense twin when faster decoding outside exact is necessary. |
| `GUFO_DISK_DEFER` | Defer disk writes until idle. | Unset. 1 with --cache-disk | Keep the disk bundle setting unless disk write deferral tests need another setting. |
| `GUFO_DISK_EVICT_DOMINATED` | Skip snapshots covered by a longer retained snapshot. | Unset. 1 with --cache-disk | Keep the disk bundle setting unless dominated snapshot eviction tests need another setting. |
| `GUFO_DISK_EVICT_SKIP_TOKENS` | Skip snapshot writes close to an existing persisted prefix. | Unset. 8192 with --cache-disk | Keep 8192 with disk caching unless eviction tests need a different minimum prompt length. |
| `GUFO_DISK_INPASS` | Enable disk snapshots during prefill passes. | 1 | Keep one unless disk cache tests need in-pass checkpoints to be disabled. |
| `GUFO_DISK_PERSIST` | Persist disk snapshots on eviction after idle. | Unset. evict_idle with --cache-disk | Keep the disk bundle policy unless persistence tests need a different write policy. |
| `GUFO_DRAFT_OUT_MODEL` | Select the four bit output head used for MTP drafts. | No file set. Optimal: matching Q4_K draft head with MTP outside exact. | Set the matching draft head when faster MTP decoding outside exact is necessary. |
| `GUFO_DRAFT_VOCAB_AUTO` | Set thresholds for automatic draft vocabulary selection. | 2,16,64 (2 of last 16 non-latin. back after 64 latin) | Keep the default thresholds unless automatic draft vocabulary tests need different thresholds. |
| `GUFO_EOG_BOUNDARY` | Permit EOG tokens at allowed boundaries. | 1 | Keep it enabled unless end token boundary comparisons need it off. |
| `GUFO_EOG_DEFER_P` | Set the deferred stop probability. | 0.9 | Keep the default unless early end token deferral tests need a different probability. |
| `GUFO_EOG_MIN_PROB` | Set the minimum EOG probability necessary to stop. | 0.2 | Keep the default unless end token probability tests need another minimum. |
| `GUFO_EOG_MIN_PROB_EXACT` | Compute EOG probability over the full vocabulary. | Off | Keep it off unless end token decisions need full vocabulary probabilities. |
| `GUFO_HOST` | Set the server address through the environment. | Unset | Leave it unset unless the server address must come from this environment variable. |
| `GUFO_LENIENT` | Accept near miss MTP drafts inside thinking spans. | Bounded with MTP. Otherwise off. | Keep bounded outside exact unless strict speculative acceptance is necessary. |
| `GUFO_LENIENT_LAMBDA` | Set the lenient speculative acceptance bound. | 0.5 | Keep the default unless lenient acceptance tests need a different bound. |
| `GUFO_LENIENT_THINK_ONLY` | Limit lenient acceptance to thinking output. | on | Keep the thinking gate unless lenient acceptance tests must include non-thinking output. |
| `GUFO_LOGIT_BIAS` | Enable request token probability adjustments. | off | Keep it off unless requests must change token probabilities with logit bias. |
| `GUFO_LOOP_BREAKER` | Stop repeated answer and thinking loops. | 1 | Keep it enabled unless repeated text handling comparisons need it off. |
| `GUFO_LOOP_BREAKER_EOG_CLOSE` | Close a detected loop with an EOG token. | 1 | Keep it enabled unless loop termination tests must omit response closure. |
| `GUFO_LOOP_BREAKER_TRANSITION` | Select the loop breaker transition mode. | 2 | Keep two unless loop transition tests need another transition mode. |
| `GUFO_LOOP_BREAKER_TRANSITION_TEXT` | Set the text used for loop breaker transitions. | built-in text | Keep the built-in text unless loop termination needs a different transition message. |
| `GUFO_MAX_IMAGE_TOKENS` | Limit image tokens admitted for a request. | no cap | Keep it unset unless image token admission needs a fixed limit. |
| `GUFO_MODEL_SUPPRESS_TOKENS` | Enable model token suppression. | off | Keep it off unless model token suppression is necessary. |
| `GUFO_PORT` | Set the server port through the environment. | Unset | Leave it unset unless the server port must come from this environment variable. |
| `GUFO_EXPERT_V2K` | Run decode experts in the grouped expert kernel (adapted from Strata, MIT). | 0 (off). Exact rejects 1. | Set 1 to try the optional kernel. |
| `GUFO_NPU_DIR` | Folder with the NPU kernel files. | `tools\npu` in the package. | Change it only if you moved the kernel files. |
| `GUFO_NPU_GATE_SLICES` | Number of NPU slices for the attention q/gate rows. | 0 (off). 1 to 12 turn it on. | Keep 0 for normal use. |
| `GUFO_NPU_SLICES` | Number of NPU slices for the linear-attention rows. | 6 (1 to 12). | Keep the default unless NPU tests need another value. |
| `GUFO_NPU_SHEXP` | Use the NPU for the shared expert projections. | 1 (0 or 1). | Keep the default. |
| `GUFO_NPU_TIMEOUT_MS` | Time limit for one NPU job. | 8000 (1 to 8000). | Keep the default. |
| `GUFO_NPU_KEEPALIVE` / `GUFO_NPU_KEEPALIVE_MS` | Keep the NPU awake between jobs. | 1 / 1000 ms (0 or 100 to 60000). | Keep the default. |
| `GUFO_PREFILL_FORWARD_TOKENS` | Set the maximum tokens in each prefill forward pass. | 4096 | Keep 4096 unless prefill geometry tests outside exact need a different pass size. |
| `GUFO_PRODUCTION_SAMPLING` | Enable the production sampling preset. | off | Keep it off unless production sampling policy tests need it on. |
| `GUFO_REASONING_BUDGET` | Enable request reasoning token budgets. | off | Keep it off unless requests must enforce a reasoning token budget. |
| `GUFO_ROUTED_LANEROW` | Enable the routed expert prefill kernel. | 1 | Keep the profile setting unless routed expert kernel comparisons need a different setting outside exact. |
| `GUFO_SPEC_EOG_STOP` | Permit speculative EOG stops. | 1 | Keep it enabled unless speculative end token comparisons need it off. |
| `GUFO_THINK_EOG_GUARD` | Block end of turn inside an unfinished thinking span. | 1 | Keep it enabled unless thinking termination comparisons need it off. |
| `GUFO_THINK_GUARD_ESCALATE` | Close thinking after this many blocked end-of-turn attempts. Zero turns it off. | 16 | Keep 16 unless thinking guard tests need a different end token streak length. |
| `GUFO_THINK_MARKER_BIAS` | Set a fixed thinking marker probability bias. | 0 | Keep zero unless a fixed thinking marker bias is necessary outside exact. |
| `GUFO_THINK_MARKER_BIAS_BY_EFFORT` | Map reasoning effort to think marker bias. | low:-2,medium:-1,xhigh:0 | Keep the default map unless reasoning effort tests need different thinking marker biases. |
| `HOST` | Set the server address through the environment fallback. | Unset | Leave it unset unless the server address must come from the environment fallback. |
| `PORT` | Set the server port through the environment fallback. | Unset | Leave it unset unless the server port must come from the environment fallback. |

## Default behavior

The default profile enables the behavior policies.
They control response boundaries, unfinished thinking, and repeat loops.
Adaptive thinking uses the map `low:-2,medium:-1,xhigh:0`.
Medium is the default effort.
Bounded lenient acceptance applies to thinking spans with MTP.
The twin and draft head need matching model files.
The default output is not byte-exact.

## Disk prompt cache

Keep disk prompt cache off for normal use.
Disk snapshots write to the SSD.
Automatic RAM staging can limit snapshots near 70K tokens.
The limit depends on the model and available RAM.
Set `--cache-disk` only when prompts must persist across restarts.
To turn it on, start Bubo with `start.cmd -CacheDisk <folder>`, or add `--cache-disk <folder>` to the server command. The first start with the cache checks the model files once and saves a stamp. The 109K-token test used `--cache-disk-staging-bytes` 5 GB.
Measured on a 109K-token session: restore after a restart takes 3.5 s instead of 88.8 s.
SSD writes are about 3 GB for each long saved session and about 5 GB for a 90-request agent replay.
Decode speed cost is about 2.5 %.
The resolver supplies the disk policies when that folder option is set.
The default disk budget is 8 GiB.

## Preparation with bubo prepare

```text
bubo prepare [all|runtime|model] [OPTIONS]
```

The default step is `all`. The command needs no Python and no administrator access.

| Option | What it does | Default |
| --- | --- | --- |
| `--output ROOT` | Parent folder for the model files. | `%LOCALAPPDATA%\Bubo` |
| `--yes` | Accept the displayed download plan. The command still prints the plan. | Ask |
| `--verify` | Force a full SHA-256 check of the selected files. | Quick fingerprint |
| `--offline` | Use local files only. A missing file fails before any network use. | Off |
| `--resume` | Resume partial downloads that the command started. | On |
| `--no-resume` | Start partial downloads again. | Off |
| `--mtp-model PATH` | Copy an existing MTP file that matches the pinned hash. | Download |
| `--mmproj PATH` | Copy an existing vision projector that matches the pinned hash. | Download |
| `--cache-dir PATH` | Folder for the runtime archives. | `ROOT\cache\rocm-10.0.0-gfx1151` |
| `--runtime-dir PATH` | Folder for the AMD runtime. | Beside `bubo.exe` |
| `--help` | Print help. No file is written and no network is used. | |

The runtime step rejects the model import options. The model step rejects `--runtime-dir`.
Set `HTTPS_PROXY` (and `NO_PROXY` if needed) when your network needs a proxy.

| Exit code | Meaning |
| ---: | --- |
| 0 | Success, help, or nothing to do. |
| 2 | Syntax, schema, header or profile error. |
| 3 | You declined the plan, or there was no answer, or you canceled. |
| 4 | Offline and a file is missing, or a source is not pinned. |
| 5 | Network or TLS failure. |
| 6 | Size or hash check failed. |
| 7 | Storage or access error, or a conflicting file already exists. |
| 8 | Another prepare run is active. |
| 9 | Archive or internal error. |

The model step writes the eight model files, `licenses\LICENSE`, `licenses\NOTICE`, `licenses\PROVENANCE.md`, the folder `.bubo` and, last, `bubo-package.json`.
The manifest names the profiles `default`, `subagents`, `stock` and `exact`.
The manifest path is chosen in this order: `--manifest PATH`, then the `BUBO_MANIFEST` variable, then the file beside the selected model. A flag overrides an environment value, which overrides the manifest, which overrides the profile.

### Model checks

The server checks file sizes, GGUF headers and tensor tables, and the manifest entries. It never hashes model payloads, including the first start.
`--verify-model` needs a valid package manifest. It hashes all eight model files (about 2 to 4 minutes on an SSD, an estimate) and the files you select with explicit paths.
`--cache-disk` needs a prepared manifest. The first start with the cache checks the model files once and saves a stamp.

## Chat API limits

Two requests have fixed conditions. Both are designed behavior.

- **Logprobs.** A chat request can return logprobs only when thinking is off, the request is plain text (no tools and no images), and it has no stop sequences. Otherwise the server returns HTTP 400 with the message `chat logprobs require thinking off, plain text, and no stop sequences`.
- **Prompt inspection.** The `/debug/prompt` endpoint accepts plain text only. A request with tools or images returns the message `prompt inspection requires plain text without tools`.

## Memory

Use the 128 GB laptop setup with 96 GB GPU memory for Qwen3.8 Flash Next.
Use a fixed 128 GB page file.
Reduce context if memory is limited.
More sessions and images need more memory.
Keep `--allow-unmeasured-memory` off for normal use.

See [install](install.md), [quickstart](quickstart.md), and [profiles](profiles.md).
