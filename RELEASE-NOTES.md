# <img src="assets/owl.svg" width="32" alt="Owl (Egyptian hieroglyph G17)" align="absmiddle"> Halowin Bubo v0.1.0

First release from HaloForge Labs.

Halowin Bubo runs the gufo engine natively on Windows with AMD HIP.
It targets Strix Halo laptops with Ryzen AI Max and Radeon 8060S graphics at approximately 70 W.
The base is gufo v0.10.0 plus main commit `97b8645f`.
The Windows layer comes from [pixmaate/gufo `windows-port`](https://github.com/pixmaate/gufo/tree/windows-port).

## Included

- Optimal defaults for Qwen3.8 Flash Next.
- Four profiles: `default`, `stock`, `exact`, and `subagents`.
- An OpenAI-compatible local API.
- All gufo model families listed in the [README](README.md#supported-models).
- Fast restarts with the optional disk prompt cache: a 30K-token conversation continues in about 2 s after a restart instead of processing the whole prompt again (about 25 s). It is off by default (`start.cmd -CacheDisk <folder>`). A saved snapshot is about 954 MB for 30K tokens. See [the README section](README.md#fast-restarts-with-the-disk-cache).
- RAM prompt cache.
- NPU prefill (on by default when usable): prefill is about 5 to 7 % faster on an 8K-token prompt and about 3.5 to 5.5 % faster on a 16K-token prompt at 70 W. Decode does not change. Prompts under 4,096 tokens do not use it. It needs the AMD NPU driver and XRT, and it is off in `exact` and `stock`. See [NPU prefill](docs/npu.md).
- Optional grouped decode expert kernels, adapted from Strata (MIT): off by default. Turn them on with `GUFO_EXPERT_V2K=1`. They are not available in `exact`.
- Resolved settings output through `--print-effective-settings`.
- Windows notices and component licenses.

## Default settings

| Item | Default |
| --- | --- |
| Profile | `default` |
| Sessions | `1` |
| Context | `262144` tokens |
| Reasoning | Thinking on, medium effort |
| Adaptive thinking | Marker bias map `low:-2,medium:-1,xhigh:0` |
| Behavior | Policies on, with the engine loop breaker |
| Speculative decode | MTP when the matching sidecar is supplied |
| MTP control | `survival`, draft vocabulary `auto`, prompt lookup on |
| Lenient acceptance | `bounded`, thinking spans with MTP |
| Dense twin and draft head | Use the matching derived files outside `exact` |
| RAM cache | `6442450944` bytes (6 GiB) |
| Disk cache | Off |
| Local API | `127.0.0.1:8080` |

The default output is not byte-exact.
Use `--profile exact` for reproducibility with the same build, machine, files, prompt, seed, and request settings.
Exact disables speculation, behavior policies, the dense twin, the draft head, and prompt caches.
It uses one session and fixed prefill geometry.
Each request processes its full prompt.

Exact-profile result: in exact mode, Bubo gives the same tokens as the stock reference. We tested this on the release source tree (the full exact proof ran on build `163762d9`; the release revision `3ff74dce` differs from that tree only in the Windows launcher script and its test). It is not a promise for all builds:
- `--exact` matched the stock-flag reference on 24 fixed cases (short, 8K and 33K natural, thinking, tool, stop, 4095/4096/4097 edges, image, long decode, non-ASCII) at temperature 0 and 1, seed 1. There were five exact passes and one retained pass, and all 24 cases matched in each. The comparison covered text, reasoning, tool calls, token ids and finish reasons.
- Logprobs were compared only where the product supports them (thinking-off cases). The other cases were compared by text and token ids.
- The prompt and continuation cache is off in `--exact`. Cached restores are not bit-identical to a fresh prefill (3 of 3 measured cases differed) and are outside this guarantee.
- One machine (Windows, AMD Radeon 8060S, ROCm path). Other hardware, seeds and prompts are not covered. No speed claim.
- The `exact` and `stock` profiles refuse the NPU and Strata switches and reject the tested production environment overrides (`GUFO_DENSE_TWIN_MAX_N`). Other user overrides were not tested.

Bubo is tuned for Qwen3.8 Flash Next. The other models run with upstream gufo settings.

## Runtime and model setup

The package contains no AMD files or model weights.
Run `bubo prepare`. It downloads three official ROCm 10.0.0 wheels for `gfx1151` (967.2 MiB) and eight pinned model files (108.6 GiB) after you approve a displayed plan.
It checks the SHA-256 hash of every file, resumes stopped downloads, honors `HTTPS_PROXY`, and writes `bubo-package.json` last.
The default checks find a wrong file, a wrong variant and a truncated file. `--verify` forces a full SHA-256 check.
The server finds the model files through the manifest. It never hashes model payloads at startup. `--verify-model` checks all eight files on request.

The dense twin and Q4_K draft head come from the HaloForge Labs Hugging Face repository (commit `bb72204d`).
Qwen licenses apply to these derived model files.

## Measured performance

The 70 W laptop measurements include 1,364 tokens/s for prompt processing (pp2048, release build v0.1.0, 2026-10-12) and 51.6 tokens/s across real agent requests (2026-10-10).
A copy-heavy rename edit reaches up to 97.4 tokens/s on its second request with the prompt cached (release build v0.1.0, 2026-10-12), also at about 70 W laptop power.
See the [README](README.md#performance-at-70-w) for the conditions.

## Reasoning check against gufo (pixmaate’s Windows port)

Pre-release build (B6.1), 2026-10-10, not v0.1.0. Same 35 requests, same seeds, same laptop in Performance mode.

| Set | Halowin Bubo correct | gufo (pixmaate’s Windows port) correct |
| --- | --- | --- |
| GPQA Diamond | 20/24 | 19/24 |
| AIME 2026 | 3/4 | 2/4 |
| HMMT 2026 | 3/4 | 4/4 |
| IMO 2026 (final answer) | 3/3 | 2/3 |
| Total | 29/35 | 27/35 |

Same or better accuracy, and about 24 % faster decode at about 70 W laptop power. Tokens used, on the 32 requests that both engines answered: Bubo generated 580,456 completion tokens and gufo (pixmaate’s Windows port) 525,293. The median ratio per request is 1.00. On the pre-release build, one Bubo request (AIME item 29) repeated text until the context limit (261,913 tokens); it is left out of the 32. The 95 % intervals overlap.

## Fixed

- In rare cases, the thinking phase could not end and repeated text until the context limit.


## Known issues

- The answer loop breaker detects repeat periods up to 16 tokens. Longer answer repeats can continue.
- Disk prompt cache is off by default to limit SSD writes. Automatic staging can limit snapshots near 70K tokens.
- Deep sessions and images need more memory. Use the documented memory setup or reduce context.


## Credits and licenses

Thanks to the [gufo contributors](https://github.com/gufo-org/gufo) and [pixmaate/gufo](https://github.com/pixmaate/gufo).
Their code retains the MIT license.
AMD runtime components retain their separate licenses.
Model files retain the [Qwen license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

See [install](docs/install.md), [quickstart](docs/quickstart.md), [options](docs/options.md), and [profiles](docs/profiles.md).
