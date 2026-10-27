# Halowin Bubo v0.1.1

Halowin Bubo v0.1.1 adds the Bubo app, eight more models, a memory guard and setup checks. It is faster than v0.1.0 on the same laptop.
It targets Strix Halo laptops with Ryzen AI Max and Radeon 8060S graphics at approximately 70 W.
The base is gufo v0.10.0 with selected upstream fixes.

Sampled output for a fixed seed can differ from v0.1.0.

## Speed

Same laptop, same model file, approximately 70 W. Bubo v0.1.1 against Bubo v0.1.0:

| Work | v0.1.1 | v0.1.0 | Change |
| --- | ---: | ---: | ---: |
| Mixed work (agent, code, prose) | 66.0 | 59.5 | +11 % |
| Code edit, rename | 107.5 | 97.3 | +10 % |
| Code edit, comment change | 97.4 | 90.8 | +7 % |
| C++ code | 74.8 | 69.7 | +7 % |
| Cold 2K prompt | 1,328 | 1,227 | +8 % |
| Prose | 56.4 | 52.9 | +6 % |
| Cold 32K prompt | 1,384 | 1,318 | +5 % |

Python code and reading new text in an agent session are level with 0.1.0. Speeds are tokens per second. Conditions and every row: [Benchmarks](https://haloforgelabs.com/benchmarks).

## New

- **The Bubo app.** `start.cmd` opens a desktop app. The app downloads the engine files and a model after you approve them, starts the server, and shows the connection address, the requests and the log. It starts the same `bubo serve` server as the command line. It can update Bubo with one click. See [Use the app](https://haloforgelabs.com/docs/app/).
- **Lower-memory Flash Next.** The app lists a lower-memory build of Qwen3.8 Flash Next (Models, then Discover, then Download). It is a download of about 90 GB. At the default context of 200,000 tokens, Bubo estimates about 66 GiB of GPU memory for it instead of about 91 GiB, which is about 25 GiB less. It reads prompts at about 0.88 to 0.92 of the speed of the standard build. It writes code and agent text at about 0.75 to 0.9 of that speed, and prose at about the same speed. These are ranges from one run. On a 35-question reasoning check (GPQA Diamond, AIME 2026, HMMT 2026, IMO 2026) it answers 29 correctly, the same as the standard build and Bubo 0.1.0 within the noise of a set this small. Its dense tensors come from the Huihui model and its expert tensors from ISTA-DASLab's GSQ-RCO quantization of Qwen3.8 Flash Next. See [Licences](https://haloforgelabs.com/licences).
- **Shorter wait for the first token.** On the next request of a conversation, the first token arrives about 40 ms sooner at 32K tokens of context (527 ms instead of 567 ms, 7 % sooner) and about 113 ms sooner at 100K (515 ms instead of 627 ms, 18 % sooner). The wait in the queue falls from about 52 ms to 7 ms at 32K, and from about 130 ms to 8 ms at 100K. The output is identical.
- **More models.** The app downloads Qwen3.8 27B, Qwen3.6 27B, Cyber-Tiel Coder 35B-A3B, Gemma 4 26B-A4B and 31B (each in two builds: Unsloth QAT and Huihui abliterated QAT) and DiffusionGemma 26B E38 (abliterated, 4-bit). These are 16 to 23 GB each. All of them accept images. See [Models supported](https://haloforgelabs.com/models).
- **DiffusionGemma accepts images.** Bubo downloads the vision file for DiffusionGemma with the model. In a test on the release package, a request with one image returned an answer that described the image correctly.
- **96 GB machines.** Use the lower-memory build of Qwen3.8 Flash Next. Set Variable Graphics Memory (AMD Software: Adrenalin Edition > Performance > Tuning) to the largest value offered (72 GB if available, at least 64 GB), then restart Windows. At 72 GB, Bubo uses the default 200,000-token context for one session. At 64 GB it does too when about 16 GB of system memory is free, and picks a smaller context when less is free. These values are calculated: Bubo has not yet run on a 96 GB machine.
- **Memory check before Start.** A tight setup starts only after you tick the override for it. A setup that does not fit never starts.
- **Live progress.** The Requests page of the app lists each running request with its phase (Reading or Writing), its progress and its speed in tokens per second, and the Server page shows a one-line summary. `bubo status --watch` shows the same in a terminal.
- **Smaller updates.** An update keeps the files that did not change.
- **Memory guard.** Bubo keeps free memory above a floor of 4,096 MiB. A request that needs new memory when free memory is below the floor is refused with HTTP 503 (`insufficient_memory`, `Retry-After: 5`) or HTTP 507 (`insufficient_memory_permanent`). Under Windows low-memory pressure, Bubo drops cache snapshots first. It stops a running request only as the last step (HTTP 503, `memory_pressure`). `GUFO_CACHE_HOST_FLOOR_MB=0` turns the guard off. See [System setup](https://haloforgelabs.com/docs/system-setup/#memory-guard).
- **Memory estimates.** `bubo models estimate --model PATH` prints a memory estimate and a recommended context. `bubo models list` shows the prepared packages and the models in your folders.
- **Setup checks.** `bubo diagnose` shows `gpu_reservation`, `page_file` and `power_mode`. `bubo serve` writes one WARN line at the start when the GPU memory reservation or the page file differs from the tested setup. A warning never stops the server.
- **Cache details in the response.** `usage.gufo` has `cache_snapshots_skipped` and `cache_snapshot_skip_reason` when the cache refuses a snapshot copy. `cache_miss_reason` names the cause when the cache removed or refused a checkpoint. See [options](docs/options.md#ram-snapshot-cache).
- **MTP for Gemma 4.** A Gemma 4 package that has an MTP file uses MTP by default. `--no-mtp` turns it off. On the Huihui abliterated QAT builds, medium-length replies are about 1.1x faster on Gemma 4 26B-A4B (about 67 tokens/s instead of 60) and about 1.9x faster on Gemma 4 31B (about 21 instead of 11), with the draft model on and the same laptop.
- **Gemma 4 images.** Gemma 4 26B and 31B accept images through the vision file.
- **Repair.** `bubo prepare` replaces a damaged model file in the model folder after you approve the download. The plan lists the file as `replaces a file that did not match`. If you answer no, nothing changes (exit code 3). A file that you supplied, or a file outside the model folder, is not replaced.
- **Stall watchdog.** If generation stops making progress, the server exits as `device_lost` (exit status 75). The app starts the server again once.

## Changed

- **NPU prefill is off by default.** After the prompt-reading speed-ups, it adds less than 1 %. Turn it on with `--npu-prefill on`. If you turn it on, the server uses it only with AMD XRT runtime 32.0.20102.3931. With any other runtime, the server writes a warning, reports the NPU as faulted, and uses the GPU. Decode does not change. See [NPU prefill](docs/npu.md).
- **Default RAM cache is 8 GiB** (8,589,934,592 bytes) in the `default` profile. In v0.1.0 it was 6 GiB. The `subagents` profile stays at 4 GiB. An explicit `--cache-ram-bytes` value always wins.
- **The RAM floor of the snapshot cache** is 4,096 MiB by default and counts the host memory that a snapshot really uses. In v0.1.0 it was off.
- **Model name.** `GET /v1/models` lists the model's own id and `bubo`, and both names work. For another name, set `--served-model-name`.
- **Default context.** For Qwen3.8 Flash Next the default context is 200,000 tokens (v0.1.0: 262,144). The largest is still 262,144 (`--context 262144`).
- **Image limit.** On Qwen3.8 Flash Next the default limit is 65,536 image tokens for each request. All images in the prompt count. The limit reserves 640 MiB for each session. In v0.1.0 the reservation followed the context size. Change the limit with `GUFO_MAX_IMAGE_TOKENS`.
- **`--help` works before `bubo prepare runtime`.** The help text of every command is part of `bubo.exe`.
- **Package layout.** `bubo.exe` is in the `bin` folder. `start.cmd` is in the package folder. The engine files go to `%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151` by default. Change the folder with `--runtime-dir` or `BUBO_RUNTIME_DIR`. See [install](docs/install.md).
- **Disk cache options** are no longer in `start.cmd`. Use `bubo serve llm --cache-disk` (see the [quickstart](docs/quickstart.md#fast-restarts-with-the-disk-cache)).

## Fixed

- The `routed-lanerow fallback` INFO line is written once for each format and shape, not once for each layer and pass.
- The `device_lost` log line says `remedy=restart_process exit_status=75`. In the app, the server starts again once, after 5 seconds.
- `usage.completion_tokens_details.reasoning_tokens` is in Chat Completions responses, for non-streaming and streaming requests.
- API: a `max_tokens` of `-1` or `null` keeps the server default. CORS preflight answers with the headers that the page asks for. Errors on `/v1/messages` and `/v1/messages/count_tokens` use the Anthropic error body. Responses streams send a keepalive event.
- Streamed tool calls: when the model sent a tool call with an empty parameter value, the stream could end with an error ("generation failed") and no final event. The call now arrives with the empty value. Requests without streaming were not affected.
- `/v1/responses`: when a tool has no `strict` field, or `strict` is `null`, Bubo uses the tool schema as the client sent it. Optional properties stay optional. `strict: true` and `strict: false` work as before. `/v1/chat/completions` did not change. (upstream #643)
- The command line accepts empty Windows environment values.
- The model scan follows OneDrive folders. It refuses symbolic links and junctions only.
- The docs listed `/v1/messages/count_tokens` as not implemented. It works and returns `input_tokens`.

## Known issues

- Bubo is not code-signed. Windows can ask you to confirm. See [Windows security prompts](docs/install.md#windows-security-prompts).
- The repeat guard stops repeated text in answers only when the repeating unit is 16 tokens or shorter (thinking and tool arguments: up to 128 tokens). A longer repeating passage in an answer can run until the length limit.
- Deep sessions and images need more memory. Reduce `--context` if memory is limited.
- The server writes some whole numbers in exponent form, for example the context size as `2e+05` in `/v1/models`. This is valid JSON. A client that needs an integer type there can fail.

The full list: [Known issues](https://haloforgelabs.com/docs/known-issues/).

## Credits and licenses

Thanks to the [gufo contributors](https://github.com/gufo-org/gufo) and [pixmaate/gufo](https://github.com/pixmaate/gufo).
Their code retains the MIT license.
AMD runtime components retain their separate licenses.
Model files retain the licenses of their authors.

See [install](docs/install.md), [quickstart](docs/quickstart.md), [options](docs/options.md), and [profiles](docs/profiles.md).

---

# Halowin Bubo v0.1.0

The v0.1.0 notes are in the release history of this repository: [v0.1.0](https://github.com/haloforge-labs/bubo/releases/tag/v0.1.0).

