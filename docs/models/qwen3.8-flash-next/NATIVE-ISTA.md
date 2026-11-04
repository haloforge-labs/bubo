# Native ISTA loading

Audit: 2026-10-30T00:19:52-04:00 (`date`), base bf2301714.

A full pre-change role/type/pass audit and a small validation failure list
were recorded before the change.

Implementation order: IQ2_XS, IQ3_S, IQ3_XXS. Preserve published weight types
through binding and upload; dense dispatch reads each tensor's own type.
Existing Q8/F32 production arithmetic and optional twin guards must remain
unchanged. CPU/source work only; operator performs GPU validation.

The owner superseded the standalone header-test framework with one CPU
reference comparison, one real-weight byte comparison, one cold bench command,
and one same-prompt old/new server comparison. No verdict packs or timing gates.

Stage 1: dense native types share the existing MMVQ/MMQ adapters, BF16 stacks
retain their encoding, and activation scratch accounts for native quantized
matrices. Q6_K head re-encoding is removed. Twin identity is carried separately
from weight type. Legacy F32 production stacks retain their established
conversion; native BF16-router packages preserve F32 sidecar types as well.
Quantized gather and F16 PLE convolution remain for stage 2. Build configuration
is in progress; no execution claim is made at this stage.

Stage 2: native IQ3_S/IQ4_XS/Q6_K embedding gathers use block decoders adapted
from the local MIT `ggml/src/ggml-cuda/dequantize.cuh` source. The copyright is
retained in `native_gather.hpp`; `licenses/gsqhalo-MIT.txt` and
`licenses/strata-MIT.txt` remain present. F16 PLE convolution widens each weight
in registers. Draft subset copies use encoded row geometry. Native vector
inputs allocate their zero-id rows independently of optional twins. The common
IQ expert entry point uses the same negative-id guard for every IQ type.

Follow-up: device descriptors own tensor names after optional readers close.
Native decode cohorts retain 1..8-row vector slices; only explicit twin
tensors use the twin batch optimization. Native uploads reject any pending
weight conversion and log the retained storage types. Dense launch logs name
the actual Q8 route as well as the new native routes.

Stage 3: `qwen38_flash_next_native_cpu_test` accepts real shard-1 paths and an
optional `--mtp` sidecar. It runs the actual config and binder, reports role
routes from tensor descriptors, and compares three small real rows through the
new gather helpers against the existing reference dequantization and dot.
One aggregate byte comparison checks the same old/new output buffers. The
shared negative expert-id predicate is included in that CPU invocation.
No synthetic GGUF, emulator, or separate header framework is used. The test
is registered under the CPU label and skips without explicit model paths.

Stage 4: `qwen38_flash_next_native_bench` rotates original output-projection
weights over all available layers and prints vector/verification/matrix
timings. It discards bandwidth above 256 GB/s without speed verdict gates.
Operator build/stage, bench, matched old/new production prompt, and native
rung commands are in the job's `COMMANDS.md`. They are not executed by this
CPU/source job. Runtime staging hard-links unchanged baseline dependencies
and independently copies changed binaries; its manifest is informational.

CPU validation completed 2026-10-30T01:54:22-04:00 (`date`): real IQ2_XS,
IQ3_S, IQ3_XXS and production Huihui headers bind; the 512-expert MTP sidecar
matches. Three real 2,560-column rows through the new gather helpers match
reference dequantization and dot exactly; 30,720 output bytes match. The common
negative expert-id guard passes its CPU check. Runtime and native benchmark
compile successfully; no GPU or server qualification was run.

Behavior commit 47fd1bb8 is staged as a separate runtime. First operator
command and the four build/bench/server commands are in the job folder. The
final report is kept with the job records.
Full Q2_0 and Coder verification remain lowest priority and are not done.
