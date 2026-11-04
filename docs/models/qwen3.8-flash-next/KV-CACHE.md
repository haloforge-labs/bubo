# Flash-Next KV cache formats

F16 remains the default. On the ROCm backend, `prompt`, `chat`, `bench` and
`serve llm` accept `--kv-cache-type q8_0` or `--kv-cache-type q8_0-rotated`.
Both modes are runtime opt-ins, not weight quantization settings. They apply
only to Flash-Next (`qwen4exp`); unsupported architectures and the CPU reference
path reject them rather than silently using F16.

```sh
gufo chat --model "$MODEL" --kv-cache-type q8_0
gufo serve llm --model "$MODEL" --kv-cache-type q8_0-rotated
```

These modes change the model's arithmetic and may change output tokens. Exact
codec/operator tests do not establish model-quality or end-to-end performance
parity with F16.

## Storage and arithmetic

The selected format covers ordinary trunk and MTP K/V. Indexer raw keys,
pooled keys, recurrent state and model weights retain their existing formats.
A q8 block holds one little-endian F16 scale followed immediately by 32 signed
bytes: 34 bytes rather than 64 bytes of F16. A 512-element K or V row therefore
occupies 544 bytes rather than 1024. Total session memory also includes the
unchanged state and scratch allocations; it does not shrink by that ratio.

For each independent block, insertion computes `d = max(abs(x)) / 127` in F32,
then `q = clamp(roundf(x / d), -127, 127)`. Division is IEEE rounded division;
`roundf` ties go away from zero. Only the stored scale is rounded to F16 RNE.
Codes are not recomputed from that rounded scale. An all-zero block or an F32
scale that underflows to zero produces zero scale/codes. F16 scale underflow
preserves codes and reconstructs zeros.

Attention reconstructs `F16_RNE(F32(stored_scale) * q)`, not an unrounded F32
product. Nonzero operands must match exactly; GPU operand comparisons treat
signed zeros as equivalent. The packed half load subtracts its integer bias
before multiplication, with an explicit optimization barrier: combining those
steps into an affine half FMA changes rounding.

The portable codec rejects nonfinite inputs, negative scales (including -0),
code -128 and reconstructions that overflow F16. Packed row validation runs
before converting ingress-buffer ownership. Disk startup checks matching
packed publications before catalog/index exposure, including runners without
optional durable artifact fingerprints. Bad semantic publications are retired;
resource or operational failures are not mislabeled as malformed data.

## Optional rotation

`q8_0-rotated` uses a normalized natural-order Sylvester Hadamard transform:
H256 independently for each K/Q head and four contiguous H64 transforms for
each V/output head. Normalization comes first (1/16 or 1/8), followed by
ascending-distance F32 butterflies. The supported geometry is D256, rotary64,
24 query heads and 2 KV heads.

K/Q rotation follows RMS normalization and RoPE; V rotation follows its F32
projection. Quantization follows rotation. The output inverse runs after all
attention reductions and before the sigmoid gate. Separate cache-only MTP
producers use the same ordering. Indexer inputs are not rotated. Batched MTP
insertion and pooled indexer production carry the configured YaRN parameters.
A floating-point transform/inverse round trip is not promised to be bit-exact.

Rotation is always compiled; there is no separate build option. Select it
only through `--kv-cache-type q8_0-rotated`. The runtime default remains F16.

## Transient plain-q8 prefill view

For eligible ordinary matrix attention calls with at least 256 rows, plain
`q8_0` caches always use a shared transient F16 view, with no extra switch or
opt-out. They are reconstructed into scratch and consumed by the existing WMMA
attention route. Reconstruction uses the stored q8 bytes and their exact half
rounding. Canonical storage, snapshots, ingress and the numerical epoch do not
change. This does not select F16 KV storage.

The view excludes rotated KV, MTP attention, narrow/decode-sized calls and
last-only calls. View scratch = **2 KiB × min(configured context, WMMA context
limit), shared per executor**, not per session: two F16 buffers at KV width 512.
The WMMA limit is 262144 at the supported geometry: 390.625 MiB at context
200000, at most 512 MiB. An unspecified executor context uses its native
capacity before applying the same cap; unsupported WMMA geometries allocate
no view. Canonical/session capacity is
unchanged; supported calls beyond the compact view use the packed reader,
with no new rejection. Buffers are allocated eagerly only for plain q8
executors with an eligible maximum batch. Admission observes memory after
these shared allocations. Ineligible calls retain direct packed loads.
Contexts outside the WMMA route keep the existing narrow attention fallback.

## Compatibility and tests

Payload versions are 19 for F16/plain q8 and 20 for rotated q8. Append layouts are
1, 2 and 3 respectively. Identities bind dtype, basis, codec, context, execution
policy and the `rocm-qfn-q8-wmma` numerical lane. Type/epoch mismatches
are rejected before restoration. No conversion of older snapshots is attempted.
The existing four-query sparse WMMA grouping is retained; equality
across different query groupings or prefill boundaries is not a codec contract.

CPU tests cover the codec, layouts, identity bytes, packed ownership/rollback
and CRC-valid malformed disk admission. Native operator targets include
`qwen38_flash_next_kv_q8_ops_test`, `qwen38_flash_next_kv_rotation_ops_test`
and `qwen38_flash_next_kv_q8_load_test`. The load fixture exhausts 6,268,920
scale/code pairs in the all-codes-finite scale range, including subnormal
operands and both 34-byte block parities.

Explicit model targets cover snapshot/continuation restoration and scaled YaRN
scalar/chunk/ragged-batch production. Separate-producer fixtures slice supported
stacked Q8 MTP weights into test-only row views; they do not enable additional
weight formats. CPU unit tests are not HIP compilation, native execution or
full-model quality evidence. See [testing](../../TESTING.md) and
[model quality](QUALITY.md) for the broader validation requirements, including
[historical q8 distribution references and their protocol limits](QUALITY.md#q8-kv-distribution-reference-october-27-2026).
