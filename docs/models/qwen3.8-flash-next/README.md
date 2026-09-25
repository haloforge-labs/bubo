# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

**UD-IQ4_XS** also loads (unqualified): its IQ3_S/IQ4_XS gate/up and IQ4_NL
down experts run on the vendored MMQ/MMVQ kernels, and its Q6_K embedding and
output head are re-encoded to Q8_0 at load (Q8_0's step is about a quarter of
Q6_K's, so the added rounding is small, but it is not bit-identical).
`qwen38_flash_next_iq_experts_ops_test` checks those expert paths against a
double-precision reference.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Sampled
requests use deterministic acceptance/cost control for seeded replay; all-greedy
C>1 batches may use measured cycle costs. Each request keeps private caches,
rollback and RNG. See [MTP qualification](QUALITY.md).

Three opt-in MTP options change only what the draft side proposes; verification
is unchanged, so sampled outputs keep the target distribution and greedy text is
byte-identical. They apply to single-session decoding (measured on Windows,
sampled, thinking on, 3 seeds per prompt, against the default):

| Option | What it does | Measured |
| --- | --- | --- |
| `--mtp-draft-vocab latin` | The draft head scores only special and ASCII/Latin-script text tokens (130K of 248K), reading half of the output head per draft step; ids map back on the GPU | +4.9% decode, identical texts and acceptance |
| `--mtp-policy survival` | Sampled drafting stops when the product of per-proposal acceptance estimates (from the draft head's top probability, by depth) falls below 0.40, instead of the per-cycle length controller | With the draft vocabulary: +8.7% over the default (prose +7%, code +7%, reasoning +13%) |
| `--prompt-lookup` | After each kept MTP draft, a match of 12+ tokens in the conversation proposes the tokens that followed it (point-mass proposals, exact under rejection sampling). Shorter matches propose nothing: an ungated 3-token rule lost 6-8% at temperature 1 | File-editing replies +10% sampled; log quoting +23% greedy; prose unchanged. Timings report `lookup_n` / `lookup_n_accepted` |

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
