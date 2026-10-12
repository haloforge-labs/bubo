# Internal linear z contract (P7)

Only a successfully started, resolver-authorized ServerPrefill is injected
through server/backend/model options. Direct model callers have no capability.
Build OFF removes every new declaration, header and dispatch branch.

The qualified trunk has 48 layers, full-attention interval 4, K=2560,
QKV=10240 rows and z=6144 rows. The 36 linear layers independently own their
CPU BFP B streams. CPU Q8 staging precedes GPU upload; no borrowed GGUF,
HIP userptr, mapped BO or second XRT layer is used. The persistent P2/P3
adapter's six private BOs are allocated at startup and reused serially after
confirmed completion; staged weights are copied into its B slot per dispatch.

Slices 1..6 offload the final slices of z while preserving fused GPU
convolution. Values 7..12 offload all z and trailing QKV channels; they use a
plain GPU prefix projection and defer convolution until the NPU tail has been
fenced. The entire strict 1..12 range is supported. Other model/weight shapes
fall back before submission. Only eager, non-speculative prefill
of 4096, 8192, 12288 or 16384 tokens, within executor capacity, qualifies.
Decode, verification, batched prepared projections and partial groups stay GPU.

The HIP bridge zeros the mandatory encoder invalid flag, encodes/downloads each
4096-token A group into pinned owned storage, fences the event and checks the
flag before any NPU submission. Its worker owns independent A/B bytes and the
shared service capability until completion, including after a caller timeout.
There are no HIP host callbacks. GPU projection prefix/convolution and
alpha/beta work can overlap the serial CPU/XRT worker. The full output pitch
remains 16384; only the GPU grid's row count changes.

The first-consumer barrier is immediately before GatedDeltaNet. This is
conservative: it also precedes the GDN scan, rather than adding a callback
between scan and gated norm. Every slice must be submitted, executed, healthy,
result-valid, of the expected size and free of recovery uncertainty. All C
streams are decoded into an independent finite FP32 tail; no partial result
is uploaded. The pinned two-dimensional tail upload finishes before GDN is
allowed to read z. Early exits after input staging also fence queued GPU
producer work before unwinding; an uncertain fence retains all staging and the
owning model/server stack. External HIP uncertainty parks the NPU worker,
trace and independent watcher instead of marking the host normally finished. No accumulation or change to the remaining GPU dot products, scan, gated
norm or output projection arithmetic is introduced. The 7..12 route uses the
existing unfused GPU convolution after completing all raw QKV rows; those NPU
QKV values are approximate and require independent G4 qualification.

The lane waits at most 30 seconds for its worker. XRT calls retain P3's
independent phase watchdog and parsed completion timeout; consumer rejection
is serialized through that same worker and stops periodic/new dispatches.
This rejection can itself require the bounded P3 phase wait. No retry or
post-submit GPU fallback is permitted. Uncertain NPU or HIP DMA holds the
host alive with owned resources for operator recovery. Successful HIP event
polling uses the parsed timeout; a user-space timeout cannot prevent an OS hang.
Allocation, HIP event/copy API blocking behavior and actual device teardown
remain unqualified until G4.

Trace records use lane=linear-z and explicit layer/slice/group, with P3's
durable timestamps, sizes, ERT outcomes and optional output checksum. Consumer
rejection writes result_invalid and the first-error state through the existing
single-writer trace. It never writes directly from a competing HIP callback.

CPU/fake tests qualify selection, layouts, result flags, launch geometry,
first-consumer ordering and owned lifetime. HIP bridge/kernel compilation is
not device correctness, overlap, speed or firmware safety evidence. G4 must
validate A encoding/invalid flags, BFP results, tail copy canaries, event/copy
coherency and the integrated output. P9 must reconcile request poisoning,
cancellation, drain and approximate cache identities before qualification.

Owned B storage costs 607.5 MiB for 36 qualified layers at the default six
slices, 1215 MiB at twelve slices. At four groups/twelve slices the FP32 tail
is 768 MiB; the future/result copy and pinned upload can retain three copies.
These additional host allocations are not folded into the existing GPU-only
resident-byte accounting. Qualification must include them in the memory budget;
no performance or complete memory-accounting qualification is claimed here.
