# Halowin Bubo

**Local AI for Windows on AMD Strix Halo laptops.**

**Supported hardware:** Bubo is built and tested for AMD Strix Halo with 128 GB of memory on Windows, mainly laptops. Smaller models need less memory. It can work on other machines, with no promise.

**96 GB machines:** use the lower-memory model. Set Variable Graphics Memory (AMD Software: Adrenalin Edition > Performance > Tuning) to the largest value offered (72 GB if available, at least 64 GB), then restart Windows. At 72 GB, Bubo uses the default 200,000-token context for one session. At 64 GB it does too when about 16 GB of system memory is free, and picks a smaller context when less is free. These values are calculated: Bubo has not yet run on a 96 GB machine.

Halowin Bubo is a native Windows fork of [gufo](https://github.com/gufo-org/gufo), from HaloForge Labs.
It uses AMD HIP and optimal defaults for Ryzen AI Max laptops with Radeon 8060S graphics at approximately 70 W.
The command is `bubo`. The executable is `bubo.exe`.

Gufo provides the engine. The Windows layer comes from [pixmaate/gufo, branch `windows-port`](https://github.com/pixmaate/gufo/tree/windows-port).
The base is gufo v0.10.0 with selected upstream fixes.

## Performance at 70 W

Tokens per second, higher is better. Same laptop, same model file (Huihui Qwen3.8 Flash Next, UD-Q4_K_XL), same day.

| Workload | Bubo 0.1.1 | Bubo 0.1.0 |
| --- | ---: | ---: |
| Prose | 56.4 | 52.9 |
| Python code | 68.7 | 65.4 |
| C++ code | 74.8 | 69.7 |
| Long agent session, writing | 68.5 | 62.4 |
| Long agent session, reading new text | 946 | 930 |
| Code edit, rename | 107.5 | 97.3 |
| Reading a 32K prompt | 1,384 | 1,318 |

The laptop is an ASUS ProArt PX13 with Radeon 8060S graphics in Performance mode at approximately 70 W. Sampled rows use temperature 1.0 with thinking on. Edit rows are greedy. Comparisons with other engines and every row: [Benchmarks](https://haloforgelabs.com/benchmarks).

## Requirements

Use this setup for Qwen3.8 Flash Next. Smaller models need less memory: see [Models supported](https://haloforgelabs.com/models).

| Item | Optimal setting |
| --- | --- |
| Operating system | Windows 11 x64. Tested on build 26200. |
| Microsoft runtime | Microsoft Visual C++ 2015-2022 Redistributable (x64), from https://aka.ms/vs/17/release/vc_redist.x64.exe. Most PCs already have it. Windows stops `bubo.exe` at start if the runtime is missing. The start script checks for it. |
| Hardware | AMD Strix Halo with Ryzen AI Max and Radeon 8060S graphics (`gfx1151`). |
| System RAM | 128 GB for Qwen3.8 Flash Next. |
| GPU memory | 96 GB in the BIOS or the laptop control app. |
| Page file | Fixed 128 GB on the system drive. Initial and maximum size: 131072 MB. |
| Power mode | Performance mode at approximately 70 W. |
| Graphics driver | AMD 32.0.31041.1004 (2026-08-16) or newer. |
| Disk | About 113 GiB free for the default model. |
| Model | Qwen3.8 Flash Next, GGUF, `UD-Q4_K_XL`. The app and `bubo prepare` download all files. |

The default context is 200,000 tokens for one session. The largest is 262,144 tokens (`--context 262144`).
More context, images, and more sessions need more memory.
A page file does not replace physical RAM for full-speed operation.

## Install

1. Download `bubo-v0.1.1-windows-x64-gfx1151.zip` from the [Bubo release page](https://github.com/haloforge-labs/bubo/releases/tag/v0.1.1). The file is 66,241,059 bytes. Its SHA-256 value is `eeffccd3dc4c322b968a332ce378ffd99521eb377db262ac195111d847d5f2ee`. `bubo --version` prints `bubo 0.1.1 (0cb4bd87), engine gufo 0.10.0`.
2. Extract the complete package into a folder.
3. To use the app, double-click `start.cmd` in the package folder. See [Use the app](https://haloforgelabs.com/docs/app/).
4. To use the command line, open PowerShell in the `bin` folder of the package and run `.\bubo.exe prepare`.

The Bubo package contains no AMD runtime files or model weights.
The app and the `bubo prepare` command download them. They need no Python and no administrator access.
It has two steps, `runtime` and `model`. With no step name it runs both and asks once.

| Step | What it downloads | Size |
| --- | --- | ---: |
| `runtime` | Three official AMD ROCm 10.0.0 wheels for `gfx1151` from `stable.repo.amd.com` | 1,014,229,359 bytes (1.0 GB) |
| `model` | Eight pinned model files: four `UD-Q4_K_XL` shards, MTP predictor, vision projector, dense twin and Q4_K draft head | 116,617,136,608 bytes (117 GB) |

Before a download, the command shows each address, size and SHA-256 value, the AMD license information and the destination, and asks for approval (`--yes` accepts the plan).
It checks the hash of every file and writes `bubo-package.json` last.
If a download stops, run the command again. It resumes.
If your network needs a proxy, set `HTTPS_PROXY`.
The default folder for the model is `%LOCALAPPDATA%\Bubo`. Use `--output ROOT` to change it. The engine files go to `%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151` by default. Use `--runtime-dir` or `BUBO_RUNTIME_DIR` to change that folder.
The twin and the draft head come from the HaloForge Labs Hugging Face repository.
See the [install guide](docs/install.md) for all options.

## First run

The app starts the server for you. See [Use the app](https://haloforgelabs.com/docs/app/).

To use the command line, run `bubo prepare` first. Then start a text server with your first model shard.
The server finds the other shards and the sidecar files through `bubo-package.json`.

```powershell
.\bubo.exe serve llm --model "$env:LOCALAPPDATA\Bubo\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
```

This command uses the `default` profile. Requests use the model's id, which `/v1/models` shows (for example `huihui-qwen38-flash-next-ud-q4-k-xl-v1`). The name `bubo` also works, so older setups keep working. Use `--served-model-name` to set another name.
The manifest supplies the MTP predictor, the dense twin, the draft head and the vision projector.
Use the [quickstart](docs/quickstart.md) for a first API request.

The local API address is `http://127.0.0.1:8080/v1`.
Keep the local address for normal laptop use.

## Lower memory options

- Pick a smaller model. The app has eight models of 16 to 23 GB.
- Use a shorter context: `--context N`.
- Use a smaller RAM cache: `--cache-ram-bytes N`. The default is 8 GiB.
- Leave out the dense twin (`--no-twin`) or the image projector (`--no-vision`).
- Run `bubo models estimate --model PATH` for a memory estimate and a recommended context.
- The memory guard keeps 4,096 MiB free. A request that needs more is refused with HTTP 503 or 507.

## Fast restarts with the disk cache

The disk prompt cache is off by default. When you turn it on, Bubo keeps the computed prompt state of long conversations on the disk. After a server restart, or after the RAM cache drops a conversation, the conversation continues without processing the whole prompt again. The cache writes to your SSD.

See the [quickstart](docs/quickstart.md#fast-restarts-with-the-disk-cache) for the steps. To turn it on, add the folder for the cache and a staging size to the server command:

```powershell
.\bubo.exe serve llm --served-model-name bubo --cache-disk "D:\BuboCache" --cache-disk-staging-bytes 5368709120
```

- Bubo saves a snapshot about 30 s after the conversation goes idle.
- The disk budget is 8.6 GB by default. Change it with `--cache-disk-bytes`. A long session can need more.
- Add `--cache-disk-staging-bytes 5368709120` (5.4 GB) for long sessions.
- Turn the cache off again with `--no-cache-disk`. The `exact` profile turns all caches off and ignores the disk cache.

## Profiles

| Profile | Use | Main settings |
| --- | --- | --- |
| `default` | Normal use. This is the optimal profile. | One session, 200,000 context tokens, 8.6 GB RAM cache, medium effort. |
| `stock` | Plain gufo settings. | No Bubo profile defaults. Existing flags and environment values still apply. |
| `exact` | Reproducible-output mode for Qwen3.8 Flash Next. | One session, seed 1, fixed prefill size, no speculation or prompt cache. |
| `subagents` | Concurrent agent requests. | Four sessions, 131072 context tokens each, 4.3 GB RAM cache. |

Select a profile with `--profile NAME`.
See [profiles](docs/profiles.md) for the full scope and memory limits.

## Options

These defaults apply to Qwen3.8 Flash Next.

| Option | Default | When to change |
| --- | --- | --- |
| `--profile` | `default` | Select `stock`, `exact`, or `subagents` for the uses above. |
| `--reasoning-effort` | `medium` | Use `low` for shorter reasoning or `xhigh` for difficult tasks. |
| `--think` | `on` | Use `off` when the task does not need reasoning. |
| `--context` | `200000` | Reduce it if memory is limited. The largest value is `262144`. |
| `--sessions` | `1` | Use the `subagents` profile for concurrent requests. |
| `--mtp-model` | From the manifest | Supply another matching MTP sidecar for speculative decode. |
| `--mmproj` | From the manifest | Supply another BF16 vision sidecar for images. `--no-vision` leaves it out. |
| `--cache-ram-bytes` | `8589934592` (8.6 GB) | Reduce the budget if host RAM is limited. Zero means automatic sizing. |
| `--npu-prefill` | Off | Use `on` to prefill long prompts on the NPU. See [NPU prefill](docs/npu.md). |
| `--served-model-name` | The model's id | Set the model name that clients send. `bubo` always works as well. |
| `--no-mtp` | Unset | A Gemma 4 package that has an MTP file uses MTP by default. Use `--no-mtp` to turn it off. |
| `--cache-disk` | Off | Set a folder only when prompts must persist across restarts. |
| `--port` | `8080` | Change it if another service uses the port. |
| `--print-effective-settings` | Off | Print the resolved settings before you load the model. |

Full model SHA-256 examination is opt-in: `bubo prepare --verify` or `bubo serve llm --verify-model`.
See [all options](docs/options.md) for the current options.

## Supported models

The app downloads these models after you approve the files. The weights are not part of the Bubo package.

| Model | Download | Images |
| --- | ---: | :--: |
| Qwen3.8 Flash Next (Huihui and original) | 116.6 GB | Yes |
| Qwen3.8 27B | 19.6 GB | Yes |
| Qwen3.6 27B | 18.5 GB | Yes |
| Cyber-Tiel Coder 35B-A3B | 23.4 GB | Yes |
| Gemma 4 26B-A4B (Unsloth QAT and Huihui abliterated QAT) | 15.7 and 18.8 GB | Yes |
| Gemma 4 31B (Unsloth QAT and Huihui abliterated QAT) | 18.8 and 20.8 GB | Yes |
| DiffusionGemma 26B E38 (abliterated, 4-bit) | 20.1 GB | Yes |

Bubo is tuned for Qwen3.8 Flash Next.
Bubo also contains the code of the other model families of [gufo](https://github.com/gufo-org/gufo): DeepSeek V4 Flash, Qwen3-ASR, Qwen3-TTS, Qwen-Image and MiniMax H3. These are not tested on Windows, and the Bubo app does not download them. They run with upstream gufo settings, if at all.
See [Models supported](https://haloforgelabs.com/models).

## Known issues

- Bubo is not code-signed. Windows can ask you to confirm. See [Windows security prompts](docs/install.md#windows-security-prompts).
- The repeat guard stops repeated text in answers only when the repeating unit is 16 tokens or shorter (thinking and tool arguments: up to 128 tokens). A longer repeating passage in an answer can run until the length limit.
- Deep sessions and images need more memory. Reduce `--context` if memory is limited.
- The server writes some whole numbers in exponent form, for example the context size as `2e+05` in `/v1/models`. This is valid JSON. A client that needs an integer type there can fail.
- Bubo refuses a request that needs new memory when free memory is below the floor (HTTP 503 or 507). See [Memory guard](https://haloforgelabs.com/docs/system-setup/#memory-guard).
- If the GPU stops responding, the server logs `event=device_lost remedy=restart_process exit_status=75` and exits with status 75. The Bubo app starts the server again once. Without the app, start the server again, or run it under a supervisor that restarts it on failure. See [Known issues](https://haloforgelabs.com/docs/known-issues/).
- Tracing reduces decode speed. Keep tracing off.
- With the prompt cache on (default profile), a restore from the cache is not bit-identical to a fresh prefill of the same prompt. The `exact` profile turns the cache off for this reason.
- Chat logprobs need thinking off, plain text (no tools or images) and no stop sequences. Otherwise the server returns 400. See [options](docs/options.md#chat-api-limits).
- The `exact` profile stops with a one-line error when an explicit flag or environment value would change an exact-profile value. Remove that setting, or do not combine it with `--profile exact`.

## Licenses and credits

Gufo and the Windows layer use the MIT license.
Credit goes to the gufo contributors and pixmaate/gufo `windows-port`.
HaloForge Labs provides the Bubo release defaults and Windows package.

AMD runtime components have separate licenses.
Read `THIRD-PARTY.txt` in the package, and the `licenses` folder in the package and in the runtime folder.
Keep these notices with the installed runtime.

Qwen licenses apply to model weights and the derived dense twin and draft head.
Read the [Qwen license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) and the [Huihui license](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/blob/main/LICENSE).
All licences: [Licences](https://haloforgelabs.com/licences).

## Links

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Options](docs/options.md)
- [Profiles](docs/profiles.md)
- [Release notes](RELEASE-NOTES.md)
- [NPU prefill](docs/npu.md)
- [Model files on Hugging Face](https://huggingface.co/HaloForgeLabs/bubo-qwen3.8-flash-next)
- [gufo](https://github.com/gufo-org/gufo)
- [Releases](https://github.com/haloforge-labs/bubo/releases)
- [HaloForge Labs](https://haloforgelabs.com)

