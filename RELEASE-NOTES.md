# Halowin Bubo v0.1.2

Halowin Bubo v0.1.2 runs the ISTA-DASLab GSQ-RCO files IQ3_S, IQ3_XXS, IQ2_XS and the Flash Next Coder as published, reads prompts of 4,096 or more new tokens faster in an optional fast mode, can store the context cache in half the memory, and adds in-engine abliteration, off by default.
It targets Strix Halo laptops with Ryzen AI Max and Radeon 8060S graphics at approximately 70 W.
The base is gufo v0.26.0.

Output for a fixed seed can differ from v0.1.1.

## What is new

- **Run it in a lot less memory: ISTA quant support.** Bubo reads the published ISTA-DASLab GSQ-RCO GGUF files IQ3_S, IQ3_XXS and IQ2_XS of Qwen3.8 Flash Next without conversion, including the IQ formats of the expert tensors. On ISTA's own benchmarks (mean of AIME25, GPQA-Diamond and LiveCodeBench v6), IQ3_S scores 93.26 against 93.12 for the full-precision BF16 model, and IQ2_XS scores 89.16, 4.0 points (4 %) lower. The main files are 55 GB and 39 GB; with the 28.8 GB embedding file the downloads are about 84 GB and 68 GB. Bubo also runs the Flash Next Coder (GSQ-RCO, 256 experts). The app lists the Coder. With the MTP predictor the Coder writes about 69 tokens/s instead of about 35 without it (about twice as fast), and the text is identical. Its weights take about 29 GB of GPU memory (about 32 GB with the draft file, about 38 GB for the running server at a short context), and it reads a 22K-token prompt at 1,316 tokens/s. The download is about 61 GB (58.4 GB of model files and 2.8 GB for the MTP file), and it does not take image input. On a 22K-token prompt, the published IQ3_S, IQ3_XXS and IQ2_XS files read at 1,253, 1,260 and 1,213 tokens/s.
- **Fast prompt reading.** `bubo serve llm --prefill-mode fast`, or Prefill mode on the Server page in the app. About 16 % faster reading of prompts that have 4,096 or more new tokens (1,718 instead of 1,482 tokens/s on a 22K-token prompt) with no noticeable reasoning regression, by using a tuned number of experts while reading. New tokens are the prompt tokens that the cache does not restore. Shorter prompts read as in normal mode. Decode unaffected. The mode works on models with 512 experts. The Coder (256 experts) and the profile `exact` refuse it. The default is `normal`.
- **Smaller context cache.** `--kv-cache-type q8_0`, or "KV cache" with the choice "Q8 (half the memory)" on the Server page in the app, stores the context cache of Qwen3.8 Flash Next in 8-bit form. The KV cache is 47 % smaller (4.5 GiB to 2.4 GiB at 196,608 tokens), so each extra 64K session needs 1.1 GB instead of 1.9 GB and more or longer conversations fit. The default stays `f16`. Existing caches rebuild when the type changes. Flash Next only.
- **In-engine abliteration (off by default).** `--refusal-direction on`, or "In-engine abliteration (Qwen3.8 Flash Next)" on the Server page in the app, turns on the projection of a refusal direction out of the model's outputs. The direction file is in the `refusal` folder beside `bubo.exe`, and the zip includes it. It applies to Qwen3.8 Flash Next models that are not already abliterated. For other models, such as the Huihui builds and the Coder, Bubo refuses to start with it on and says why. A restart is needed.

## Speed

Bubo v0.1.2, standard Flash Next file, default settings, 70 W, tokens/s: prose 57.1, Python 66.3, C++ 77.5, six-prompt mix 65.6, long agent session writing 66.2, code edit (rename) 109.0, code edit (comment) 97.6, 32K cold prompt 1,415, 2K cold prompt 1,320. Method and every row: [Benchmarks](https://haloforgelabs.com/benchmarks/).

## Other changes

- More API: `POST /apply-template`; more fields in `GET /props`; repeated `--served-model-name`; IPv6 listen addresses. Tool calls: more than 128 tools work; integer arguments keep their written form; optional arguments stay; whitespace between parallel calls is not content. `/health` answers while the model loads.
- Thinking: when the model tries to end its thinking early, the server can add a short phrase so it keeps thinking, up to 2 times for each request. On in the `default` and `subagents` profiles, off in `exact`.
- Opt in: `--enable-yarn` lets `--context` go above 262,144 on Flash Next (without it Bubo refuses a larger context). `bubo serve embedding` serves EmbeddingGemma 2 on `/v1/embeddings`; Bubo does not ship the weights.
- Prompt reading on IQ and K-quant files is about 5 % faster by default.
- The repeat guard now also ends answers that repeat a passage of up to 128 tokens (it was 16).
- The memory check now counts the Q8 KV cache and the Coder exactly.
- `--cache-ram-bytes off` works: Bubo keeps no RAM checkpoint.
- Bubo checks a draft, projector or head file that you pick yourself from its headers before the model loads.
- A Q8 KV cache setting on a model that does not support it falls back to f16 instead of failing.
- Fixed: DiffusionGemma tool calls without the opening tool-call marker are now read as tool calls when they have the closing marker and a declared tool name (issue #11).
- Fixed: requests for token log probabilities (logprobs) could fail with a server error on Qwen3.8 Flash Next when MTP was on.
- `bubo diagnose` reports the revision and the compiler of the build.
- The lower-memory build writes 19–28 % faster than in v0.1.1, and reads a 32K-token prompt about 7 % faster (same laptop, 70 W).

## Known issues

- Bubo is not code-signed. Windows can ask you to confirm. See [Windows security prompts](docs/install.md#windows-security-prompts).
- If you use `--cache-disk`, the first start of v0.1.2 deletes the disk-cache files of v0.1.1 in that folder (`*.kvc` and temporary files, top level only), and the first requests after that run cold. Use a separate folder for each server.
- The option `--cache-disk-staging-bytes` is ignored on Qwen3.8 Flash Next, and the server logs a warning.
- Deep sessions and images need more memory. Reduce `--context` if memory is limited.

The full list: [Known issues](https://haloforgelabs.com/docs/known-issues/).

## Install

Download `bubo-v0.1.2-windows-x64-gfx1151.zip` (71,952,963 bytes, SHA-256 `153903f2a60820e15d126e6db8df7406aba325bbb77378e17d5bea1a87ca1a2d`). After you extract it, `bubo --version` prints `bubo 0.1.2 (53e4c3e4b), engine gufo 0.26.0`.

## Credits and licenses

Thanks to the [gufo contributors](https://github.com/gufo-org/gufo) and [pixmaate/gufo](https://github.com/pixmaate/gufo), and to ISTA-DASLab for the GSQ and RCO quantizations (arXiv:2604.18556, arXiv:2605.00649).
Their code retains the MIT license.
AMD runtime components retain their separate licenses.
Model files retain the licenses of their authors. The Flash Next Coder uses the Qwen Community License 1.0.
New third-party notices: libsndfile, libsoxr and their audio libraries (LGPL and others), LLVM OpenMP, the IBM Plex Sans font (SIL Open Font License 1.1).

---

The v0.1.1 notes are in the release history of this repository: [v0.1.1](https://github.com/haloforge-labs/bubo/releases/tag/v0.1.1).
