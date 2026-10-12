# P5 staging and configuration contract

Consume `ParseKnobs` only with the resolver's immutable prefill authorization.
Registered environment/manifest values are data; none can authorize device
creation. Missing values select ON defaults; present empty, signed, trailing,
overflowing or out-of-range values fail before session creation. Binary switches
accept only the literal `0` or `1`. DIR is a nonempty UTF-8 path, including `0`.

Linear slices are 1–12 (default 6); query/gate slices are 0–12 (default 8),
covering two heads per 1024-row slice; shared expert is 0|1 (default 1).
Isolation is 0|1 (default 0), but both values retain the port's private BO
copies: no shared HIP/XRT pages. TIMEOUT_MS is 1–8000 (default 8000);
KEEPALIVE is 0|1 (default 1); KEEPALIVE_MS is 0 or 100–60000 (default 1000);
TRACE is 0|1 (default 0). KEEPALIVE=0 or KEEPALIVE_MS=0 disables periodic
heartbeats, not persistent startup, its bounded heartbeat or dispatch exposure.
Partial lane settings also do not exclude startup exposure. The fork's LAYERS,
STATS and WARMUP_MS experiments are not ported here.

`Q8Matrix` must describe complete CPU Q8_0 storage with 2560 columns, using
the canonical 34-byte blocks. Qualification checks byte lengths and overflow
before offsets/copies. Linear z uses the tail of a 16384-row matrix. Query/gate
uses a 13312-row matrix with 24 query heads, 2 K/V heads and head dimension
256; the last 1024 K/V rows remain outside the staged slices. Shared gate/up
must have matching nonzero row counts through 1024. Staging owns a copy,
preserves Q8_0 bits, rejects nonfinite scales and pads short shared matrices
with zero Q8_0 blocks. Public slice descriptors are revalidated before copying.

These are Q8_0 input bytes, not an XRT B BO. P6 supplies BFP encoding and layout;
P7/P8 supply layer selection, BO registration and completion/consumer fences.
`QualifiedPrefill` permits only complete 4096-token groups that fit the caller's
registered group capacity. Decode, short/remainder batches and incompatible
weights retain their existing GPU path in those integration packets. No GPU
packing, execution or model consumer is introduced by P5.

## Fabric clock qualification remains open

The production loader is lazy, guarded by the separate fabric authorization,
and uses the absolute System32 OpenCL loader with system-directory search.
It does not issue a vendor device query or `clSetDeviceClockModeAMD` setter:
the current inputs contain no qualified exact previous-mode query/ABI or
independent crash/host-death restoration implementation. Symbol availability
does not prove either. The flag returns a clear configuration error.

The injected clock policy requires both prerequisites before setting PEAK,
saves the exact previous mode, marks restoration pending before a possibly
partially failing setter, restores on normal unwind/explicit abort/crash-handler/
watcher-death callbacks, and retains pending state on a failed restore. These
paths are CPU mock proofs, not real crash recovery. Enable the installed-driver
setter only after an independently reviewed implementation can meet the
addendum's ALWAYS-restore requirement; do not substitute an assumed DEFAULT
mode or claim a same-process destructor covers process death or power loss.
