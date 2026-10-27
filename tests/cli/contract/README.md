# CLI contract v1

`event.schema.json` defines the output envelope. `input.schema.json` defines
prepare input. Events use UTF-8 without a BOM, one object per LF-terminated
line. Sequence numbers start at 1 and increase within an operation. Times use
UTC. Clients must check `schema_version` and retain typed errors.

`prepare --json` retains the requested streaming behavior. `prepare --jsonl`
uses the same stream. Diagnose and status with `--json` emit one object.
`--jsonl` emits a capability header before their result. Existing diagnose
v1.0.0 fields remain at the top level and appear in `payload` too.

Prepare emits `capabilities`, `preflight`, numbered `plan` parts,
`approval_required`, `progress`, and `result` or `error`. Plan parts share the
digest of the complete component plan. A part with `record_kind: license`
contains legal text; concatenate chunks by path and `text_offset` until
`text_bytes` bytes have been received. All parts arrive before approval.
Each source contains its immutable revision, size, hash and destination.
Source, path and volume items also use separate records. The header has
empty collections. Merge the parts before you display or approve the plan.
Free-space readings can change; paths, required space, file states and legal
text bind the approval digest.

Reply to `approval_required` with `action: approve` or `action: decline` and
the exact `operation_id` and `plan_digest` from that event. After approval,
send `action: cancel` with the operation ID to cancel. Cancel is also accepted
before approval and during native preflight verification. EOF before approval
declines. EOF during work cancels. Keep stdin open until the final event,
including when using the explicit `--yes` option. Input records are limited
to 64 KiB. Output records are limited to 1 MiB.

The prepared-default pointer is `%LOCALAPPDATA%\Bubo\prepared-default.json`.
Selection order is explicit model/manifest, `BUBO_MANIFEST`, the saved pointer,
then the standard prepared location. A damaged saved selection fails; it does
not cause a disk search. Prepare saves a custom `--output` location too.

Status and switch listing run before engine loading. Instance discovery,
verified control and watch are implemented. Runtime data comes from the
identified server. Null settings, slots and estimates mean unavailable.
Unknown estimates have a reason. Status uses publication metadata and file
sizes for preparation readiness; memory planning never scales file sizes.
Serve retains the full existing verification path.

J2b adds effective settings, inference defaults, reasoning controls, a
versioned planning estimate, actual connection/Pi inputs, prepared models,
slot data, a 50-row request ring and request_completed events. Support export
uses a typed allow-list, drops unknown raw log lines and writes only a new
local ZIP. Preview text and hashes match the saved members and ZIP.

`fixture-index.json` indexes the event/input valid and invalid directories
and the byte-preserved J1a/J1b/J2a examples in `v1-legacy/`.
`compatibility_test.ps1` verifies old fixtures, unknown output fields,
missing required fields and the directory index. New output keys are additive.
`switches.golden.json`, `switches.schema.json` and `engine-panels.json` come
from gen_switches.py. `pi-snippet.schema.json` validates the provider fragment.
Registry gui_control and audience decide the visible panels; consumers must
not implement profile, memory or visibility rules themselves.

J5 adds the object `machine` (read-only Windows power, ADLX GPU power and clock,
VGM carve in bytes, OEM identity). `status-machine.json` is a real capture from
the ASUS test laptop, `status-machine-unavailable.json` has every reading
unavailable, and `invalid/machine-*.json` break the reading rules. The `dxgi`
object (J11: Windows DXGI dedicated/shared memory, budget, usage, and the
`agree` flag against the ADLX carve) is part of the same fixtures.

`ready-observed.json` and `status-observed.json` retain the CPU fake server's
emitted records, including unavailable fields. `settings-memory-v1.json`
combines native resolver and memory-helper output plus the pinned prepared
catalogue to exercise their nested types; it does not assert that the fake
server loaded the shipped model.
`support_bundle.json` retains an actual allow-listed preview. Invalid partners
exercise missing evidence, units, sources, hashes and an unknown slot state.

Exit codes 11–18 apply only to machine lifecycle/control commands. Prepare
keeps 0, 2–9 and reserved 10. Diagnose keeps 0 for pass/warn, 1 for failed checks,
and 2 for usage errors. Human serve keeps its legacy exits.

CPU tests use compiled test-only model fixtures and fake transport. They use
no production network. The package test-symbol gate must run on the built
production bootstrap, engine and prepare library before release.

Public contract: [CLI-CONTRACT.md](../../../docs/CLI-CONTRACT.md).
The switch extension proof is `python tests/cli/contract/switches_recipe_test.py
<build>/release_profile_test.exe` (or the native executable path on Linux).
CTest runs it as `switches_recipe_test`, with labels `cpu;cli;contract;registry`.
It adds one row and its evidence entry at a time in temporary data, generates
all nine outputs, and runs enum/boundary, reader-grammar and secret cases through
the same C++ validator, resolver and effective-settings emitter used by serve.
No model loads, server starts, or repository edits are needed for the proof.
Run `python tests/cli/contract/check_docs.py --self-test` and
`pwsh -NoProfile -File tests/cli/contract/docs_schema_test.ps1` for documentation
coverage and example validation. Native writer, stdin and loader tests inject
faults through OS pipes and copied executable imports. `prepare_fault_test`
also tests the existing model prepare lock with two concurrent fake sources.
These tests require a built candidate; static fixtures do not replace them.

J16 adds `bubo models list` and `bubo models estimate` (events `models` and
`model_estimate`), the model catalogue, `prepare --catalogue-id` and
`prepare --plan`. The synthetic examples in `valid/models.json` and
`valid/model-estimate*.json` come from `models_test --write-fixtures`
(run `python tests/cli/contract/regenerate_models.py <build>/models_test.exe`
to regenerate the payloads and documentation together). These use synthetic
GGUF headers. `valid/models-shipped-catalogue.json` is the exact real CLI capture
from `gui/tests/scenarios/recordings/v12/models-list-empty.json`, recorded by
`capture-v12.ps1`; copy that capture after regeneration instead of retaining the
synthetic writer's fixed envelope and memory value. All new keys are additive; see
`docs/CLI-CONTRACT.md`, "Models and the model catalogue".

v0.1.1 removes the optional NPU side-jobs step of `bubo prepare` (the
`--npu-sidejobs` and `--accept-gemma-terms` options, exit code 11 and the
`npu_sidejobs` object in `result`): it checked for features Bubo does not have.

`models_measured_admission_test` checks recorded real Flash Next header/package
facts against three retained device-use observations, with a strict safe-side
3 GiB band and independent byte totals. To repeat against the actual files
without loading: `models_test --measured-manifest PATH/TO/bubo-package.json
tests/cli/contract/data/flash-next-admission.json`. `models_test` also compares
fixed-budget estimate rows with serve resolution for sessions 1 through 4,
checks independent context picks, default source, and explicit context.
