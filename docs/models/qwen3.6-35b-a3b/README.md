# Qwen3.6-35B-A3B (Windows port)

Hybrid Gated DeltaNet / gated-attention MoE (40 layers, 256 experts, top 8,
3B active), GGUF architecture `qwen35moe`. It runs on its own runtime in
`src/models/qwen36_a3b`, which reuses Flash-Next's kernels without changing
them, and is served by `gufo serve` like the other text models. Fine-tunes
with the same architecture (for example Tiel-Coder-35B-A3B) load the same way.

Built and measured on Windows (gfx1151, TheRock ROCm); the Linux build
compiles it but has not run it.

## Load and serve

```powershell
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
build\release\gufo.exe serve llm --model Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf `
  --context 131072 --think on --temperature 1.0 --top-p 0.95 --top-k 20 `
  --speculative mtp --draft-tokens 6 --mtp-policy survival `
  --mtp-draft-vocab latin --prompt-lookup
```

| Option | Effect |
| --- | --- |
| no `--speculative` | Target-only decoding. |
| `--speculative mtp` | Drafts with the MTP block inside the GGUF (no `--mtp-model`). |
| `--speculative dflash2 --dflash-model PATH` | A DFlash2 draft GGUF for this target (for example Tiel-Coder's). |
| `--draft-tokens N` | Most drafts per step (at most 7). |
| `--mtp-policy survival` | Stop a step's drafts once their estimated survival falls below 0.6 (MTP) or 0.2 (DFlash2). |
| `--mtp-draft-vocab latin` | The draft head covers Latin-text tokens only (Flash-Next's rule); drafts only. |
| `--prompt-lookup` | Drafts copied from the context after a 12-token match. |

One session (`--sessions 1`); no images; the disk cache is off for this
model. The prompt cache keeps the recurrent state (~130 MB per snapshot) and
reuses the KV rows in place.

## Exactness

Every verified row is sampled with the request's own sampler and a draft is
kept only when the sample equals it; on a mismatch the random state is
rewound, so the next step draws that token again. With the same seed, plain
decoding, MTP, MTP with prompt lookup and DFlash2 produce byte-identical text
(checked on three sampled tasks, greedy and top-k 0 requests). A cached turn
can differ from a cold one the way any change of prefill split does; an exact
retry of the same request replays byte-identical text from the snapshot.

## Speed (2026-09-29, Q8_K_XL, 96 GB carve-out)

Three 512-token tasks (story, code, reasoning), thinking on, temperature 1:

| Mode | t/s |
| --- | --- |
| AR | 53-54 |
| MTP 6, survival, Latin draft vocab | 76 / 80 / 93 |
| + prompt lookup | 76 / 80 / 92 |
| Tiel-Coder AR | 54-55 |
| Tiel-Coder DFlash2 7, survival, lookup | 73 / 76 / 99 |

Prefill runs about 1.5-1.8K tokens/s. `gufo-a3b` (target `gufo-a3b`) is the
bench and diagnostics driver (`--bench probe|depth|lookup|rows|prefill`).
