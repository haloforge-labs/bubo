# <img src="assets/owl.svg" width="40" alt="Owl (Egyptian hieroglyph G17)" align="absmiddle"> Halowin Bubo

**Local AI for Windows on AMD Strix Halo laptops.**

**Supported hardware:** Bubo is built and tested for AMD Strix Halo with 128 GB of memory on Windows, mainly laptops. It can work on other machines, but we make no promise.

Halowin Bubo is a native Windows fork of [gufo](https://github.com/gufo-org/gufo), from HaloForge Labs.
It uses AMD HIP and optimal defaults for Ryzen AI Max laptops with Radeon 8060S graphics at approximately 70 W.
The command is `bubo`. The executable is `bubo.exe`.

Gufo provides the engine. The Windows layer comes from [pixmaate/gufo, branch `windows-port`](https://github.com/pixmaate/gufo/tree/windows-port).
The base is gufo v0.10.0 plus main commit `97b8645f`.

## What makes Bubo fast

One chip. One package. Every layer tuned for it. All numbers are our own measurements on one 70 W Strix Halo laptop.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **Janus dense twin and Q4_K draft head.** A second, 4-bit copy of 252 dense weights serves decode and draft checking. Together with the draft head it makes decode 5.9 % faster at about 70 W laptop power. Prefill and `exact` mode use the 8-bit weights.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **MTP stack.** Draft length control, prompt lookup and bounded lenient acceptance in thinking. Decode at 128 tokens goes from 32.19 tok/s without MTP to 37.15 tok/s with MTP (release build v0.1.0, 2026-10-12), and up to 97.4 tok/s on a copy-heavy code edit, all at about 70 W laptop power.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **Loop and end-of-turn fixes.** In a test of 15 long answers, 14 stopped mid-answer without the fix and 1 with it.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **NPU prefill on Windows.** About 5 to 7 % faster prefill at 8K tokens and about 3.5 to 5.5 % at 16K at about 70 W laptop power. Decode does not change.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **Disk prompt cache (optional).** A 30K-token conversation continues after a restart in about 2 s instead of about 25 s. A 109K-token conversation: 3.5 s instead of 88.8 s.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **Exact mode.** In our proof, `exact` gave the same tokens as the stock reference on 24 fixed cases.

<img src="assets/owl.svg" width="18" alt="" align="absmiddle"> **Several users and real agents.** 102.2 tok/s in total for 8 users with plain decode, and 51.6 tok/s for a real coding agent, both at about 70 W laptop power.

Details and limits: [Under the hood](https://haloforgelabs.com/docs/under-the-hood/).

## Performance at 70 W

| Workload | Measured speed | Conditions |
| --- | --- | --- |
| Prompt processing, pp2048 | **1,364 tokens/s** | Without MTP. Median of 3. Approximately 2,040 prompt tokens. Release build v0.1.0, 2026-10-12. |
| Prompt processing, natural text, 2K / 8K / 32K | **1,240 / 1,292 / 1,287 tokens/s** | Research-notes text. Median of 2 to 3 rounds. Release build v0.1.0, 2026-10-12. |
| Decode, tg128 | **32.19 tokens/s without MTP, 37.15 with MTP** | Upstream's benchmark program. Greedy. Median of 3. Release build v0.1.0, 2026-10-12. |
| Real agent sessions | **51.6 tokens/s** | 90 requests. Context from 28K to 70K tokens. Temperature 1.0. Thinking on. 2026-10-10. |
| Copy-heavy code edit | **Up to 97.4 tokens/s** | Rename edit. 96.5 on the first request and 97.4 on the second, with the prompt cached. Release build v0.1.0, 2026-10-12. |

These results use an ASUS ProArt PX13 with Radeon 8060S graphics at approximately 70 W.
The workloads differ. The edit result includes text copied from the prompt.
Model choice, prompt content, and settings affect speed.

## Reasoning check against gufo (pixmaate’s Windows port)

Both engines answered the same 35 requests (GPQA Diamond, AIME 2026, HMMT 2026, IMO 2026) on the same laptop in Performance mode, with the same seeds.

| Set | Halowin Bubo correct | gufo (pixmaate’s Windows port) correct |
| --- | --- | --- |
| GPQA Diamond | 20/24 | 19/24 |
| AIME 2026 | 3/4 | 2/4 |
| HMMT 2026 | 3/4 | 4/4 |
| IMO 2026 (final answer) | 3/3 | 2/3 |
| Total | 29/35 | 27/35 |

Halowin Bubo gives the same or better accuracy and decodes about 24 % faster on these requests at about 70 W laptop power. The 95 % intervals overlap and each set is small. Read this as "no loss of accuracy". Details: the Benchmarks page on the site.


## Requirements

Use this setup for Qwen3.8 Flash Next.

| Item | Optimal setting |
| --- | --- |
| Operating system | Windows 11 x64. Tested on build 26200. |
| Microsoft runtime | Microsoft Visual C++ 2015-2022 Redistributable (x64), from https://aka.ms/vs/17/release/vc_redist.x64.exe. Most PCs already have it. Install it before you run `bubo.exe`: Windows stops `bubo.exe` at start if the runtime is missing. The start script checks for it. |
| Hardware | AMD Strix Halo with Ryzen AI Max and Radeon 8060S graphics (`gfx1151`). |
| System RAM | 128 GB for the documented large-model setup. |
| GPU memory | 96 GB in the BIOS or the laptop control app. |
| Page file | Fixed 128 GB on the system drive. Initial and maximum size: 131072 MB. |
| Power mode | Performance mode at approximately 70 W. |
| Graphics driver | AMD 32.0.31041.1004 (2026-08-16) or newer. |
| Model | Qwen3.8 Flash Next, GGUF, `UD-Q4_K_XL`. `bubo prepare` downloads all four shards. |

The default context is 262144 tokens for one session.
More context, images, and more sessions need more memory.
A page file does not replace physical RAM for full-speed operation.

## Install

1. Download `bubo-v0.1.0-windows-x64-gfx1151.zip` from the [Bubo release page](https://github.com/haloforge-labs/bubo/releases/tag/v0.1.0). The file is 44,096,931 bytes. Its SHA-256 value is `d1c92dacf68e8e379b8e02c8d68ba18ba102b2fd22784e25cb06b0757fac1306`. `bubo --version` prints `bubo 0.1.0 (3ff74dce), engine gufo 0.10.0`.
2. Extract the complete package into a folder.
3. Open PowerShell in that folder.
4. Run `.\bubo.exe prepare`.

The Bubo package contains no AMD runtime files or model weights.
The `bubo prepare` command downloads them. It needs no Python and no administrator access.
It has two steps, `runtime` and `model`. With no step name it runs both and asks once.

| Step | What it downloads | Size |
| --- | --- | ---: |
| `runtime` | Three official AMD ROCm 10.0.0 wheels for `gfx1151` from `stable.repo.amd.com` | 1,014,229,359 bytes (967.2 MiB) |
| `model` | Eight pinned model files: four `UD-Q4_K_XL` shards, MTP predictor, vision projector, dense twin and Q4_K draft head | 116,617,136,608 bytes (108.6 GiB) |

Before a download, the command shows each address, size and SHA-256 value, the AMD license information and the destination, and asks for approval (`--yes` accepts the plan).
It checks the hash of every file and writes `bubo-package.json` last.
The default checks find a wrong file, a wrong variant and a truncated file. Add `--verify` for a full SHA-256 check.
If a download stops, run the command again. It resumes.
If your network needs a proxy, set `HTTPS_PROXY`.
Plan about 113 GiB of free space (an estimate).
The default folder for the model is `%LOCALAPPDATA%\Bubo`. Use `--output ROOT` to change it.
The twin and the draft head come from the HaloForge Labs Hugging Face repository (commit `bb72204d`).
See the [install guide](docs/install.md) for all options.

## First run

After `bubo prepare`, start a text server with your first model shard.
The server finds the other shards and the sidecar files through `bubo-package.json`.

```powershell
.\bubo.exe serve llm --model "$env:LOCALAPPDATA\Bubo\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
```

This command uses the `default` profile.
The manifest supplies the MTP predictor, the dense twin, the draft head and the vision projector.
Use the [quickstart](docs/quickstart.md) for a first API request.

The local API address is `http://127.0.0.1:8080/v1`.
Keep the local address for normal laptop use.

## Fast restarts with the disk cache

The disk prompt cache is off by default. When you turn it on, Bubo keeps the computed prompt state of long conversations on the disk. After a server restart, or after the RAM cache drops a conversation, the conversation continues without processing the whole prompt again.

See the [quickstart](docs/quickstart.md#fast-restarts-with-the-disk-cache) for the same steps. Measured on a 70 W Strix Halo laptop with a 30K-token session:

| Measured | Value |
| --- | --- |
| Cold prompt processing, 30K tokens | About 25 s |
| Next turn, same server, from memory | 0.9 s |
| Continue after a server restart, with the disk cache | About 2 s (1.8 to 2.4 s in two runs; about 30,400 prompt tokens restored) |
| First start with the cache (one-time model check) | 74 s |
| Later starts with the cache | 29 to 32 s |
| Size of one saved snapshot | About 954 MB for 30K tokens (about 31 KB for each token) |

In an earlier test, a 109K-token session continued after a restart in 3.5 s instead of 88.8 s. A long saved session writes about 3 GB to the SSD. A 90-request agent replay wrote about 5.1 GB. Decode speed cost is about 2.5 %. The cache writes to your SSD. Use these numbers to decide if the cache is right for your disk.

To turn it on, start Bubo with the folder for the cache:

```powershell
start.cmd -CacheDisk D:\BuboCache
```

The first start with the cache reads the eight model files once and checks them. Bubo saves a stamp. Later starts use the stamp and do not read the files again. Without a prepared model package the server refuses the cache.

To use the server command directly, add `--cache-disk "D:\BuboCache"` to it. Run `start.cmd -DryRun` to print the command that `start.cmd` runs.

- Bubo saves a snapshot about 30 s after the conversation goes idle.
- The disk budget is 8 GiB by default. Change it with `--cache-disk-bytes`. A long session can need more.
- `start.cmd -CacheDisk <folder>` is all you need. It also sets a 5 GiB staging area. Change the staging size with `-CacheDiskStagingBytes`.
- If you run `bubo serve` directly, add `--cache-disk-staging-bytes 5368709120` for long sessions. Bubo stages each snapshot in RAM before it writes it. The automatic staging size follows the free RAM at start, and it can be too small for one snapshot. With about 7 GiB free, the area was 941 MB, smaller than one 30K-token snapshot (954 MB). Bubo then dropped the snapshots and the restart processed the whole prompt again (25 s).
- Turn the cache off again with `--no-cache-disk`. The `exact` profile turns all caches off and ignores the disk cache.

## Profiles

| Profile | Use | Main settings |
| --- | --- | --- |
| `default` | Normal use. This is the optimal profile. | One session, 262144 context tokens, 6 GiB RAM cache, medium effort. |
| `stock` | Plain gufo settings. | No Bubo profile defaults. Existing flags and environment values still apply. |
| `exact` | Reproducible-output mode for Qwen3.8 Flash Next. | One session, seed 1, fixed prefill size, no speculation or prompt cache. |
| `subagents` | Concurrent agent requests. | Four sessions, 131072 context tokens each, 4 GiB RAM cache. |

Select a profile with `--profile NAME`.
The `exact` profile pins one session, seed 1, a fixed prefill size, and turns off speculation and prompt caches, to make output repeatable.
Each exact request processes its full prompt.
See [profiles](docs/profiles.md) for the full scope, what was tested (one build, one machine) and memory limits.

## Options

These defaults apply to Qwen3.8 Flash Next.

| Option | Default | When to change |
| --- | --- | --- |
| `--profile` | `default` | Select `stock`, `exact`, or `subagents` for the uses above. |
| `--reasoning-effort` | `medium` | Use `low` for shorter reasoning or `xhigh` for difficult tasks. |
| `--think` | `on` | Use `off` when the task does not need reasoning. |
| `--context` | `262144` | Reduce it if memory is limited. |
| `--sessions` | `1` | Use the `subagents` profile for concurrent requests. |
| `--mtp-model` | From the manifest | Supply another matching MTP sidecar for speculative decode. |
| `--mmproj` | From the manifest | Supply another BF16 vision sidecar for images. `--no-vision` leaves it out. |
| `--cache-ram-bytes` | `6442450944` (6 GiB) | Reduce the budget if host RAM is limited. Zero means automatic sizing. |
| `--cache-disk` | Off | Set a folder only when prompts must persist across restarts. |
| `--port` | `8080` | Change it if another service uses the port. |
| `--print-effective-settings` | Off | Print the resolved settings before you load the model. |

Adaptive thinking uses the effort map `low:-2,medium:-1,xhigh:0` for thinking markers.
Behavior policies include the engine loop breaker.
Bounded lenient acceptance applies to thinking with MTP.
The dense twin and Q4_K draft head are the optimal decode files outside `exact`.
The default output is not byte-exact.

Full model SHA-256 examination is opt-in: `bubo prepare --verify` or `bubo serve llm --verify-model`.
See [all options](docs/options.md) for the current options and preparation status.

## Supported models

Bubo is tuned for Qwen3.8 Flash Next. The other models run with upstream gufo settings.

| Model | Files and inputs | Guide |
| --- | --- | --- |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF. Text. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/deepseek-v4-flash/README.md) |
| Qwen3.8 27B | Q4/Q8 GGUF. Text and images. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/qwen3.8-27b/README.md) |
| Qwen3.8 Flash-Next | Sharded Q4 GGUF. Text and images. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/qwen3.8-flash-next/README.md) |
| Qwen3.6 35B-A3B | Q8 GGUF (`qwen35moe`) and fine-tunes. Text. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/qwen3.6-35b-a3b/README.md) |
| Qwen3-ASR 1.7B | BF16 safetensors. Audio to text. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/qwen3-asr/README.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors. Text and reference audio to speech. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/qwen3-tts/README.md) |
| Qwen-Image-2.1 | BF16 safetensors. Image generation and edits. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/qwen-image-2.1/README.md) |
| MiniMax H3 FL2VA | Pinned safetensors. Text to audio and video. | [Usage](https://github.com/gufo-org/gufo/blob/97b8645f/docs/models/minimax-h3/README.md) |


## Known issues

- The answer loop breaker detects repeat periods up to 16 tokens. Longer answer repeats can continue.
- Disk prompt cache is off by default to limit SSD writes. Automatic staging can limit disk snapshots near 70K tokens.
- Deep sessions and images need more memory. Reduce context if memory is limited.
- Like many open-source Windows builds, Bubo v0.1.0 is not code-signed. SmartScreen can show a warning, and Smart App Control blocks unsigned programs. See [Windows security prompts](docs/install.md#windows-security-prompts).
- The disk prompt cache stages each snapshot in RAM before it writes it. With little free RAM at start (about 7 GiB), the staging area can be smaller than one long snapshot, and Bubo drops the snapshot. `start.cmd -CacheDisk` sets a 5 GiB staging area. If you run `bubo serve` directly, add `--cache-disk-staging-bytes 5368709120`. See [Fast restarts with the disk cache](docs/quickstart.md#fast-restarts-with-the-disk-cache).
- The NPU kernel file ships as the fork built it. The compiler toolchain that built it is not recorded. We plan a rebuild with the open-source Peano compiler in a later release.
- With the prompt cache on (default profile), a restore from the cache is not bit-identical to a fresh prefill of the same prompt. The `exact` profile turns the cache off for this reason.
- Chat logprobs need thinking off, plain text (no tools or images) and no stop sequences. Otherwise the server returns 400. The `/debug/prompt` endpoint accepts plain text without tools only. See [options](docs/options.md#chat-api-limits).
- The `exact` profile stops with a one-line error when an explicit flag or environment value would change an exact-profile value, for example `Error: profile=exact: explicit value re-enables GUFO_DENSE_TWIN_MAX_N`. Remove that setting, or do not combine it with `--profile exact`.


## License and credits

Bubo is open source under the MIT license, the same license as gufo. The code is copyright HaloForge Labs and the gufo contributors. Read [LICENSE](LICENSE) and [NOTICE](NOTICE).
Credit goes to the gufo contributors and pixmaate/gufo `windows-port`.
HaloForge Labs provides the Bubo release defaults and Windows package.

Third-party components keep their own licenses. Read `THIRD_PARTY_NOTICES.md`.
AMD runtime components have separate licenses. Read `NOTICE-windows.md` and the `licenses` folder in the package, and keep these notices with the installed runtime.

The Qwen Community License 1.0 applies to the model weights and to the derived dense twin and draft head, as the [Hugging Face model card](https://huggingface.co/HaloForgeLabs/bubo-qwen3.8-flash-next) states.
Read the [Qwen license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) and the [Huihui license](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/blob/main/LICENSE).

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
