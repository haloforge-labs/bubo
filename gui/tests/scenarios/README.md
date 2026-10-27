# Shared CLI scenarios

From `gui/tests`, run `go run ./scenarios`. It generates 43 manifests and uses
`supervisor.ViewBuilder` to write `gui/wails/frontend/src/scenarios/*.json`.

## V12: merged CLI integration

The active generator and fake CLI use `recordings/v12/`. Run
`pwsh -NoProfile -File gui/tests/scenarios/capture-v12.ps1` from the repository
root with `-Exe <final-bubo.exe> -Scratch <isolated-evidence-directory>` after
rebuilding the final CLI. Supply `-PreparedDefault` (saved model pointer), `-RuntimeDir` (prepared runtime),
`-DependenciesDir` (runtime dependency DLLs), and `-GuardDir` (hold and test-window
sentinels) as well. All paths are explicit and recordings use an isolated scratch folder. The recorder reuses the E21/E23
command inventories and E23's raw-byte capture method. `-Only` accepts filenames
for a bounded recapture and preserves the other command provenance. Metadata
pins the CLI source revision, executable/engine hashes, environment, command,
exit code, output size and SHA-256 for each of the 72 raw outputs. Model files
and the existing C4 runtime are read only. The engine loaded is the rebuilt CPU
DLL beside this executable, not the C4 engine. No server or model is started.

This capture includes current profiles, labels, audiences, suggested bounds,
safe published integers, model catalogue/defaults and 200,000-token estimates,
while retaining explicit 262,144-token requests as such. Runtime machine mode
has no `--plan` option: the recorder closes stdin without approving. The ready
preview reports three `reuse` sources. The refresh preview uses 263 hard links
to the existing pinned payload in the isolated scratch directory, without notices
or a ready marker. It reports `ready: false`, three `reuse` sources and requires
approval; EOF exits 3 before any refresh or transfer. It never modifies the
installed runtime or the linked payload.

Run `pwsh -NoProfile -File gui/tests/scenarios/derive-v12.ps1` after recapture,
with `-PreparedModelPath` and `-PreparedPackageID` identifying the captured model.
It copies seven historical statuses and replaces product/version identity, profile rows,
`allowed` metadata (including nested effective-switch rows), and catalogue-linked
model display names from the fresh CLI captures. These are explicitly
**fixtures**, not fresh running observations. `derived-statuses.metadata.json`
pins both sources and outputs, and a test compares every other JSON value with
the original. Live rates, requests, effective values and
resolved historical contexts remain their recorded observations. The scan
warning is captured from the CLI's real drive-root refusal alongside its real
model list. The older raw captures remain intact. Fresh runtime and lower-memory
plan recordings also generate `tests/runtime-plan.json`, both lower-memory
destination plans and `tests/gemma-qat-plan.json` through ViewBuilder and the
existing frontend formatter, so consent screenshots use
real package filenames, sizes, paths and licences. Preview commands and closed
stdin never authorize acquisition, including from private release repositories.

The two old fake-model page-fact arrangements also receive the prepared package's
real model path and served ID; every identity replacement is recorded explicitly.
`progress-model.json` is derived from the verified fourth-shard transfer pinned
in `frontend/tests/recorded-observations.json`, with the final CLI's product
envelope. That source also pins real Windows GPU samples. Browser captures
preserve measured values and intervals while arranging their display time.
The scenario generator owns `src/generated/memory96-estimate-reply.json` too.

Finally run `go run ./scenarios` from `gui/tests`. All 43 Views and their
manifests are generated; none is edited by hand. The fake lifecycle's hand-written
defaults now follow the captured capabilities, 200,000-token default and
package-ID/GGUF-stem serving rule, including the `stock` to `vanilla` alias.
There is no recorded GUI `/v1/models` response to replace; the affected CLI
HTTP suite checks real-ID-first ordering, the `bubo` alias and `owned_by: bubo`
without launching a product server.

## E21: the recordings of the candidate exe

The sections below describe archived recording generations and their use before
V12. V12 is now the active source for both scenarios and fake-CLI defaults.

`recordings/e21/` holds the stdout bytes of the candidate `bubo.exe` (revision
`4924e51e`, sha256 in `metadata.json`) for `status --json` (a profile with the
default model and an empty profile), `serve --list-switches --json`,
`models list --json` (five argument sets), `models estimate` (every package at
the recorded context and session pairs), and `prepare model --plan --jsonl` for
every catalogue id. The fake CLI replays these files unchanged for those commands
(`gui/tests/fakecli/recorded_windows.go`), and the scenarios below take their
status, model list, estimates, plan and registry from them. Files are marked
`-text` in `.gitattributes`: their hashes are of the recorded bytes.

What E21 could not record stays as the older recordings or as a named overlay:
a running server (`gate3`, `lrel8`), the engine-missing status (`b4/status.json`),
the page facts (`b4/page-facts-*`), a scan warning (`v8s20`), a short drive and a
model that fits at no context size (`no_row_fits`, `volume-short`), and the
download progress. A host-floor-only failure did not come out of the recorded
sessions grid, so the former `memory-host-floor` scenario is gone. The synthetic
`f3/` fixtures and the catalogue and budget overlays are deleted. The paragraphs
below describe the older recordings that remain.
Do not edit generated View JSON. Tests and the fake CLI replay these manifests.

`recordings/gate3/` preserves real CLI records from revision `16d5a483`:
registry, catalogue, no-server status, index and P1 samples 0001 (idle), 0007
(busy), 0015 (stopped). P2 does not exist. Busy is one active slot, so the old
synthetic `ready-busy-2-sessions` is now `ready-busy`. External-server
uses these unmodified status facts and `_source: recorded`.
The supervisor-owned selected-instance/state fields come from replay inputs.

`recordings/lrel8/` replaces idle/busy with the later CLI recordings from
revision `7704f9d6`, PID 60380: samples 0027 (generating, both live rates filled)
and 0029 (idle, live rates filled for the preceding interval). Both preserve the
resolved slot context and CLI endpoints. `sources.json` records original paths,
sample times and byte-for-byte SHA256 hashes. These remain `_source: recorded`;
an idle slot does not erase the CLI's last aggregate interval throughput.

`recordings/b2/` contains fresh status, registry, catalogue and full-context
estimate output from the frozen `bubo-b2-1eca3852/bubo.exe` + engine pair.
`metadata.json` records the SHA256, commands, environment, times and exit codes.
No server/model/GPU workload was started. The pinned dependency DLL directory is
supplied as `BUBO_RUNTIME_DIR` for engine DLL lookup; this is a CPU recording,
not runtime-readiness qualification. Current free host memory was about 8.25 GB,
and the CLI truthfully reports that the full-context estimate does not fit.
These older records remain as inputs to the owner-memory regression fixture.

`recordings/b4/` contains fresh status, registry, catalogue and Flash Next
`prepare model --plan` output from the frozen `bubo-b4-d7cc0a53` directory.
`metadata.json` records every command, time, exit, source and SHA256, including
the verified binary import closure. Only system directories were on PATH;
discovery used an isolated LOCALAPPDATA. No server, model or GPU was started.
Scenarios retain this catalogue; stopped and not-prepared use its
unmodified no-server status. Preparing consumes its real plan and per-volume
bytes-to-write, safety margin and free-after-download fields without arithmetic.

`recordings/n1/` retains the former scenario registry from a registry-only
read of the frozen `bubo-n1-c9d12b97/bubo.exe`. Its 214 rows all carry `secret`;
three are marked secret. Metadata pins the executable, command, record and the
10 existing dependency DLL hashes. Those DLLs were reused in place from the
frozen B4 folder on the child PATH. No server, engine workload or binary copy
was made. The three `npu-prefill-*` scenarios combine the original idle recording
with N1's canonical status fixture for on/faulted/off. These are `_source: fixture`,
with all NPU values copied verbatim; they are not hardware observations. Recorded
scenarios reject this overlay. The rejected-switch fixture uses the CLI error's
`switch` identifier and retains an unrelated override for exact-clear coverage.

The two `cli-page-facts-*` scenarios are explicitly `_source: fixture`. Their
status files are byte-for-byte copies of B4's existing CPU fake-control captures,
with copied-source hashes in metadata. They cover loopback default keys and
authenticated imported models (null catalogue/manifest), all four CLI client
templates, and active identity through the View and selected/discovered instances.
The real idle/busy recordings predate these additions; absent identity/snippet
fields stay null. They are not inferred from served names or paths.

`owner-budget-status.jsonl` is B2's existing CPU fake-control capture copied
byte-for-byte from its retained `captures/captured-status.jsonl`. The
`owner-memory-fit` fixture combines the fresh full-context allocation terms with
that CLI capture's injected GPU/host budgets and recommendation row. Replay
requires matching sessions, context, total minimum, host need/floor and GPU
reserve before copying its `fits`/reason; it contains no memory formula. The
result is explicitly `_source: fixture`: GPU 102892924928, host 11900000000 bytes,
1 x 262144 tokens, fits, unverified startup lower bound with automatic cache.
This is the CLI planning regression, not measured peak memory or GPU admission.

The seven `update-*` fixtures cover every frozen Update state. Installed version
comes from CLI status; available version, notes, progress, size, errors and the
rollback-pending notice are explicit app-state fixture inputs. U4 supplies real
updater state later. Unwired live update actions return typed unsupported.

`recordings/machine/` retains the earlier CPU command captures. Starting, stalled, LAN preference,
failed and download-progress states remain explicitly fixture-derived. Recorded
manifests reject synthetic overlays, so these cannot masquerade as recordings.
The preserved key reset and store-failure recordings are in `recordings/legacy-key/`.
These raw legacy frames are used only for their key metadata in Connect tests and
captures; they are not generated Views or new CLI recordings.

`recordings/b5/` supplies every scenario's current registry from one unmodified
`serve --list-switches --json` read of the frozen `bubo-b5-08a8c900/bubo.exe`
and sibling engine. Metadata pins both binaries, the command, child environment,
record times/hash and available dependency-directory DLL hashes. Existing
vcpkg/SDK folders were reused in place; this is no import-closure or GPU-readiness
claim. Its 214 rows carry three `allowed.suggested` hints, four secrets and 19
excluded GUI controls. The typed View preserves hints inside `allowed.suggested`;
they do not become limits. Older registry captures retain their original bytes
and provenance and no longer feed generated Views or scenario-selected fake reads.

Scenarios without an explicit updater overlay keep update state `none`: no
updater was attached and no check was performed. The `update-current` fixture
separately represents a completed successful check with no newer release.

`stopped-saved-estimate` is an explicit fixture for the saved setup before Start.
Its authentic B4 no-server status keeps `memory_estimate` null and its original
reason. The matching prepared-default model ID selects B2's recorded full-context
estimate into `saved_estimate`, with that record's context/session request. It
reuses the `owner-memory-fit` fixture's compatible CLI budget/recommendation
overlay against the saved estimate only, so its fit and presets remain CLI
values. `prepared_ready: true` explicitly supplies fixture runtime/model readiness:
the isolated CPU status source reported both false and did not have an installed
package. This preparation-complete overlay also clears a source/runtime/model
missing error from that unprepared source. This is UI/admission proof, not
measured readiness or GPU admission.

No View JSON is authored manually and no engine fact is inferred for old records:
for example, gate3's absent endpoints remain null.

Set `BUBO_FAKE_SCENARIO` to an absolute scenario folder to replay its status,
registry, model list or prepare plan through the fake CLI. Unsupported reads
fail visibly. Dynamic serve/stop tests retain the fake lifecycle implementation.

The Tight restart screenshot cases consume actual supervisor workflow snapshots
from an external evidence directory. Set `BUBO_SHOTS_INPUT_DIRECTORY` to a folder
containing `workflow-ready-tight-blocked.json` and
`workflow-ready-tight-allowed.json`, retained from
`TestTightRestartSavedCheckboxWorkflow` with `BUBO_TIGHT_RESTART_EVIDENCE` set.
Generated Views and machine-specific run scripts stay outside the repository.
The update offer uses the real release manifest summary and payload decisions
in `recordings/release-update/`, without substituting release text.
