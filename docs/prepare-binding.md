# Prepared model binding

`bubo serve llm --model PATH` discovers only `bubo-package.json` beside the model.
`--model ROOT` also accepts the prepare output folder and selects its compiled
`models/huihui-qwen38-flash-next-ud-q4-k-xl-v1/bubo-package.json` package.
`--manifest PATH` (a canonical manifest or its containing folder) takes precedence
over `BUBO_MANIFEST`, which takes precedence over discovery. A broken selected
manifest fails; it never silently falls back.

Paths resolve in this order: positive flag, environment, manifest, profile,
built-in. Positive flags are `--twin-model`, `--draft-head-model`, `--mtp-model`,
and `--mmproj`. Environment selectors are `GUFO_DENSE_TWIN_MODEL`,
`GUFO_DRAFT_OUT_MODEL`, `BUBO_MTP_MODEL`, and `BUBO_MMPROJ`.
`--no-twin`, `--no-draft-head`, `--no-mtp`, and `--no-vision` suppress their
artefact. A positive flag combined with its suppression flag is rejected.
Stock requires both `--mtp-model PATH` and explicit `--speculative mtp` for MTP.
A draft head also requires enabled MTP; the table supplies that separate
explicit dependency when testing its binding. Exact permits no MTP, twin,
or draft head and rejects positive flags for them (exit 2). Vision is permitted
under every profile. With no manifest, projector auto-discovery remains a
fallback; `--no-vision` prevents it. A disabled typed selection never falls
back to inherited environment values in the engine.

The following ONE T10 table is the contract in
`src/cli/serve/artefact_binding_cases.hpp`; CPU tests assert every engine-option
selection and the exact rendered table. Each source is tested independently
with other required settings supplied separately.

| Profile | Artefact | Source | Binding |
| --- | --- | --- | --- |
| default | twin | flag | Bound |
| default | twin | env | Bound |
| default | twin | manifest | Bound |
| default | draft_head | flag | Bound |
| default | draft_head | env | Bound |
| default | draft_head | manifest | Bound |
| default | mtp | flag | Bound |
| default | mtp | env | Bound |
| default | mtp | manifest | Bound |
| default | mmproj | flag | Bound |
| default | mmproj | env | Bound |
| default | mmproj | manifest | Bound |
| subagents | twin | flag | Bound |
| subagents | twin | env | Bound |
| subagents | twin | manifest | Bound |
| subagents | draft_head | flag | Bound |
| subagents | draft_head | env | Bound |
| subagents | draft_head | manifest | Bound |
| subagents | mtp | flag | Bound |
| subagents | mtp | env | Bound |
| subagents | mtp | manifest | Bound |
| subagents | mmproj | flag | Bound |
| subagents | mmproj | env | Bound |
| subagents | mmproj | manifest | Bound |
| stock | twin | flag | Bound |
| stock | twin | env | Disabled |
| stock | twin | manifest | Disabled |
| stock | draft_head | flag | Bound |
| stock | draft_head | env | Disabled |
| stock | draft_head | manifest | Disabled |
| stock | mtp | flag | Bound |
| stock | mtp | env | Disabled |
| stock | mtp | manifest | Disabled |
| stock | mmproj | flag | Bound |
| stock | mmproj | env | Bound |
| stock | mmproj | manifest | Bound |
| exact | twin | flag | Rejected (exit 2) |
| exact | twin | env | Disabled |
| exact | twin | manifest | Disabled |
| exact | draft_head | flag | Rejected (exit 2) |
| exact | draft_head | env | Disabled |
| exact | draft_head | manifest | Disabled |
| exact | mtp | flag | Rejected (exit 2) |
| exact | mtp | env | Disabled |
| exact | mtp | manifest | Disabled |
| exact | mmproj | flag | Bound |
| exact | mmproj | env | Bound |
| exact | mmproj | manifest | Bound |

At ordinary startup, all eight named files receive size and GGUF header/tensor
checks and their manifest entries must match compiled pins. Main header/table
bytes also match J2's immutable header/table pins. Default startup reads no
tensor payload for hashing, including the first start. J2's measured warm
fingerprint time (about 1.4 seconds) does not authorize cold-start payload
hashing; fingerprints remain part of `prepare`. These checks cover wrong file,
wrong variant, and truncation; same-size valid-header payload corruption needs
an explicit full byte check. `prepare --verify` checks complete bytes;
`serve --verify-model` hashes all eight manifest files even when disabled,
plus distinct selected overrides, before engine creation, with file/byte
progress (about 2–4 minutes on SSD; slower storage takes longer). A manifest
is required for `--verify-model`. Runtime files, JSON and unrelated files are
outside that hash extent. No imatrix is prepared or verified.

J2's default shard stamp has a target digest rather than an independently
observed whole-file digest. It cannot authorize persistent disk-cache identity.
Manifest-backed `--cache-disk` therefore requires full `--verify-model` in the
current invocation; missing/stale stamps never trigger a full first-start hash.
The default check does not promise full payload equality. Read leases deny
concurrent write/delete while preparation checks and engine mapping complete.
No sidecar cold-start hash cache is created.
With no manifest, an explicit disk cache is rejected rather than triggering
the older backend's implicit whole-file identity scan. This applies the
owner's no-startup-payload-hashing rule before the older design's compatibility
exception for unprepared disk caches.

Manifest paths are relative to its folder and must exactly match compiled
entries. Lexical traversal (`..`), absolute/drive/UNC paths and colons are
rejected; legitimate junctions used for model reads are allowed. No alternate
manifest filename or parent-directory search is performed. Schema version 1
contains four main rows and four artefact rows, with size and SHA-256, recipe
IDs on every artefact and generator compatibility markers on twin/head.
A changed package/recipe requires a new compiled package identity.
