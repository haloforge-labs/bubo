# Attention q/gate and shared-expert prefill lanes

P8 reuses the immutable server capability, one private-BO adapter and P3
single-writer durable trace. No getenv enabler or new XRT layer is introduced.
All lane declarations/hooks/sources are absent from the OFF build graph.

Qualified trunk: 48 layers, full-attention interval 4, hidden 2560, original
linear geometry. Attention qualifies the 12 non-linear layers at 24 query
heads, 2 KV heads, head dim 256, rotary dim 64, complete Q8_0 query/gate rows
12288 and K/V 512 each. GATE_SLICES=0 disables only this lane; 1..12 stages two
paired query/gate heads per slice. GPU suffix retains pitch 13312, writes the
remaining query heads and every K/V row. Entire NPU prefix is fenced before
PrepareAttention reads any q/gate row. The conservative unfused GPU suffix
plus existing norm/RoPE preparation is source-compiled, unvalidated on GPU.

Shared lane qualifies up/gate Q8_0 [ff,2560], matching ff in 32..1024 in
multiples of 32 and down Q8_0 [2560,ff]. All 48 trunk layers are candidates;
predictor/twin/batched/decode/prepared/speculative paths are not authorized.
SHEXP=0 disables only this lane. Gate then up use two full fixed GEMMs with
zero-padded B rows; only ff columns are decoded from each. Input/result/B
storage is CPU-owned. Routed experts can overlap the NPU worker. The output
upload fences both separate GPU tensors before SwiGLUHalf, down projection
and the MoE epilogue. No post-submission GPU fallback/retry is allowed.

Both lanes require eager non-speculative whole 4096-token groups, 1..4 groups
within executor capacity. Last-only/trimmed final work falls back before
submission. Independent submitted/executed/healthy/result-valid states,
recovery, C size/codec/finite checks all precede any output upload. Input
uses the P7 mandatory invalid flag and completed HIP download; output uses
completed pitched copies. CPU worker captures B/A/service independently of
the caller. Caller waits 30 s; HIP events use parsed deadline; uncertain DMA
retains owners through the inherited recovery hold. Allocation/OS blocking
cannot be universally bounded by polling.

Additional persistent B: 270 MiB query (8 slices x 12 layers), 270 MiB shared
(two streams x 48), 540 MiB default; query at 12 slices is 405 MiB, total
675 MiB. Maximum query FP32 tail is 768 MiB, shared 128 MiB; future result
copy and pinned upload can retain three copies. Staging allocates about
34.5 MiB temporary stacked query bytes per layer. These are not included in
GPU-only resident-byte accounting. P10/G4 must budget/qualify real allocations.

CPU/fake tests cover shapes, layout, result states, deadlines, ownership,
trace tags, rejection and abstract consumer fences. Native HIP source builds
do not certify encoder execution, GPU suffix arithmetic, actual launch order,
DMA/coherence, logits/state, performance or overlap. G4 needs real device
checks and GPU hardware-queue timeline plus matching NPU QPC completions.
P9 still owns request poison/cancel/drain and approximate cache identities.
