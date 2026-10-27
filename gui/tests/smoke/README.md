# bubo-smoke

Headless driver for the v0.1.1 MVP and retained GUI v1 smoke. It runs `bubo.exe`
through the same supervisor and ViewBuilder as the app, on one loopback port
(default 18120; 8081–8086 are refused before any socket operation). No window,
real input or cgo. Only the user runs a real package after plan review.

The PowerShell entry point builds the Go driver incrementally and runs it hidden.
The fake rehearsal uses no GPU probe, real runtime, model, production discovery
or real CLI. Use a new absolute evidence directory:

```powershell
pwsh -NoProfile -File gui/tests/smoke/run-mvp.ps1 -Fake `
  -Out C:\bubo\run\app\proof\R\dry-run
```

For the real run, the user must stop production and hold both existing
GPU/build locks. Supply the **assembled K-b package**, its existing prepared
model manifest, and the exact owner line in both locks:

```powershell
pwsh -NoProfile -File gui/tests/smoke/run-mvp.ps1 `
  -Package '<absolute K-b package folder>' `
  -Manifest '<absolute prepared bubo-package.json>' `
  -Out '<new absolute evidence folder>' -AllowLockOwner '<exact owner line>'
```

Run from the final checkout used to build K-b: its supervisor/View source must
match the package's GUI commit. Ka's package checker must be integrated (the
entry refuses without its `RequireGUI` contract); structure-only CPU packages
are refused. The driver uses **package/bin/bubo.exe**, with the CLI's shared
runtime resolution. It does not run the GUI executable.

Before K-b is assembled, the user may instead supply
`-RuntimePackage '<absolute GPU runtime folder>'` with the same `-Manifest`,
`-Out` and `-AllowLockOwner` arguments. This mode verifies the declared
SHA256 hashes of `bin/bubo.exe` and `bin/bubo-engine.dll` before building or
starting the driver. The declaration comes from `MANIFEST-int.txt` when the
folder has one (integration runtimes: a copied base runtime whose
BUILD-MANIFEST.json still describes the base), otherwise from BUILD-MANIFEST.json.
Missing, duplicate or mismatched hashes refuse the run. The declared source
commit, the provenance file's name and hash (`provenance_file`,
`provenance_sha256`; `build_manifest_sha256` is always BUILD-MANIFEST.json's) and
the verified binary hashes are saved in both `entry.json` and `result/report.json`,
with the label
**pre-package smoke (CLI + supervisor on the GPU runtime; GUI exe not involved)**.
A real run starts the driver and the CLI server at Normal priority. It requires neither GUI nor helper and does not run the release-package checker.
Its result qualifies only the CLI and in-process supervisor; K-b qualification
still requires the assembled package. `-Package`, `-RuntimePackage` and `-Fake`
are alternatives. The six steps, real-hardware assertions and cleanup are the
same in both real modes.

This entry never acquires/releases the user locks or stops/restores
production. `entry.json` records the checkout
commit and script/driver hashes; `driver-pid.json` records the hidden driver's
exact identity. The outer user wrapper can write the recorded stop file.

MVP mode (`--mvp`) checks:

1. CLI-prepared View, shared runtime path/kind and CLI memory budgets/recommendations.
2. One session Ready within 900 seconds, both CLI endpoints, fits/recommended memory
   verdict and real GPU temperature/chip power.
3. One short story while the owned status stream runs: each arrival gap and the
   request's boundary gaps must be at most two seconds; live CLI rates must both
   be non-null in a busy View; the completion must appear in View requests.
4. Normal supervisor stop, exact recorded server identities gone, smoke port free.
5. Two sessions with no profile/context override: compare the effective context
   and its source against the fresh pre-start CLI recommendation; two concurrent
   stories, two busy slots/session count, the same cadence checks, normal stop.
6. Save every constructed View, scan all supervisor disk logs for errors, prove
   cleanup without a forced PID stop. Cancellation still executes cleanup.

`result/views.jsonl` contains `{observed_at,status_received_at,view}` envelopes;
the nested View is built by the production builder. Harness-only receipt times
come from the supervisor's validated-record clock, so repeated reads cannot
masquerade as new records. Views are constructed/saved once per observed status
publication/lifecycle transition rather than writing repeated polling copies.
Real snapshots have `_source: recorded`; fake snapshots have `_source: fixture`.
These are retained scenario inputs; they do not overwrite the shared fixtures.

Fake mode permits nullable live rates and skips the GPU probe. It checks rate
structure, real in-flight fake requests, busy sessions, strict cadence and
normal process shutdown. Its success verdict is **DRY RUN PASS**, never REAL
PASS. Only real mode enforces non-null CLI rates and physical GPU readings.
The fake copies its memory recommendation from the existing CLI recording;
neither harness nor supervisor calculates a context or a fit. Both serve
invocations omit `--profile` and `--context`. No timing/prediction is a pass margin.

Each smoke run writes its predictions and retained evidence to its output folder.

The **legacy v1** run plan, preconditions, pass criteria and cleanup rules are in
the smoke workflow documented here.
Only the user runs it against the real model, after reviewing the run plan.

Tests use the fake CLI (`../fakecli`, mode `BUBO_FAKE_HTTP`) and load no model:

```powershell
cd gui\tests
go vet ./...
go test -race -count=1 ./...
```

Exit codes: 0 pass, 1 a step failed, 2 usage, 3 refused, 4 a step did not run, 130 interrupted.

Modes (J12b): `--preflight-only` runs step 1 only (no server starts; overall `PREFLIGHT PASS`/`FAIL`);
`--only-step9` re-tests the log-export leak check without a model load (`STEP9 PASS`/`FAIL`);
only a full run can write `"overall": "PASS"`. `--stop-file <path>` acts like Ctrl+C when the file
appears (the user wrapper `run-smoke.ps1` uses it for its 45 minute limit). `--pins`, `--proc-patterns`,
`--build-lock-path` and `--require-locks` are described in SMOKE-PLAN.md sections 3 and 4.
