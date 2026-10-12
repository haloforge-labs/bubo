# Attention and shared-expert prefill

Attention q/gate and shared-expert lanes reuse the persistent session and
private-buffer adapter. Eligible calls are eager, non-speculative prefill
with complete 4096-token groups within executor capacity.

Query/gate slices select paired heads from 24 query heads of dimension 256.
The GPU computes the remaining query heads and every K/V row. Shared gate/up
uses matching Q8_0 matrices with 32..1024 rows in multiples of 32, padded to
the fixed GEMM shape. Gate and up results are completed before SwiGLU.

Completed, finite output is required before any consumer. Rejection disables
subsequent NPU work; uncertain DMA retains resources for process lifetime.
CPU and fake-device tests cover shapes, layout, result validation and fences.

Lane geometry adapts the MIT mikealanni/gufo fork. See NOTICES.md.
