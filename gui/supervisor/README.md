# Windows supervisor

`AppService` owns the authoritative app settings copy, lifecycle availability,
close/tray decisions, model-folder rules, LAN address selection and key staleness.
Wails supplies native window, tray, dialog and event hooks. Settings changes,
including model folders, use the backend's persistence path. This module owns
only the jobs it starts. It never
selects a process by name or port. `New` starts a bounded status handshake and
reconnect check before it handles Start. A discovered server starts read-only.
`Manage` permits CLI control. Exit leaves an external server running.

Use an absolute installation path for `Config.Exe`, a selected loopback port,
and a local log directory. Set `SettingsPath` to `SettingsPath()` to retain a
safe last error. GUI opt-ins are all off in `DefaultSettings`. The window must
execute `AppService`'s window hooks; the service applies the start-server and
close-to-tray choices without depending on Wails or a native window type.
`SetLoginStart` creates or removes only the current user's `Bubo.Control.lnk`.

Acquire `AcquireSingleInstance` before constructing the supervisor. A second
launch receives `primary=false` after it signals Show window. Listen to the
primary lease's `ShowWindow` channel. Release the lease after `Exit` completes.

Start, Prepare, Stop, Restart, Exit, Discover, SelectInstance, Manage, Consent, Diagnose and DismissError
return buffered result channels without waiting for process or pipe I/O. Read
`Snapshot()` for a copy of the reducer state. Marshal window changes on the
window thread. `Initialized` marks completion of the launch handshake.
`Approval`, `PlanParts`, `Preflight` and `Progress` supply prepare content.
Approve only on an explicit user action. Stop cancels a prepare operation.
Never add `--yes`, host, port or instance arguments: the supervisor owns them.

Ready requires matching lifecycle and status identity, model, endpoint and
effective settings plus the live PID, creation time and executable path.
An absent status sample marks data stale. Lifecycle failures have no automatic retry.
Ongoing status uses one hidden, owned `status --jsonl --watch` child after
the startup handshake. Schema, operation, sequence and selected identity are
checked on every record. A stream exit restarts with capped exponential backoff;
it does not restart or stop the server. `View.status_feed` exposes sample time,
age and freshness; after 5 seconds without a fresh record it is `not_updating`.
Stop waits 45 seconds, then terminates the owned job and waits for handle exit.
Loading has a fixed 600-second deadline. Warning events do not reset it.
Drain completion has a 2-second budget after root exit. Forced outcomes remain
Failed, and a prior protocol or load error remains visible with `Forced=true`.
Restart waits for the old process and handles before it starts a new identity.
Managed external Restart performs a CLI stop, waits on the verified external
process handle, then starts a GUI-owned replacement. It cannot force an external
process. One-shot status helpers have independent jobs and bounded query
deadlines. Stopping the status stream closes stdin and waits for its exact job;
only an unresponsive owned child is forced after the query deadline.

Stdout records are limited to 1 MiB. Stderr records are limited to 64 KiB.
Local logs use three 10 MiB files and a 2000-line ring. Plan content is limited
to 8 MiB per prepare operation. A limit breach is a visible protocol error.
Raw logs stay local. Persisted errors use allowed codes and fixed safe remedies;
CLI-provided remedies, component names, paths and log text are not persisted.

The embedded schema is generated from `tests/cli/contract/event.schema.json`.
Tests classify every current valid and invalid fixture and guard byte drift.
Lifecycle identity includes PID, creation time and executable path; ready/status
must agree before Ready is authorized. An older CLI returns a visible unsupported error.

Run the proof from PowerShell:

```powershell
pwsh -NoProfile -File ..\tests\test.ps1 -EvidenceDir C:\path\to\proof -GuardDir C:\path\to\guards
```

This pins Go 1.27.2, GCC 16.2.0, x/sys v0.31.0 and wincred v1.2.3, with read-only module sums.
All process fixtures use a fake CLI. They bind no port and load no engine.
Tests shorten deadlines only in test configurations.

The J4 window adapter calls `SetVisible` to use 2-second status samples while
visible and 10-second samples while hidden, without changing the CLI's stream
cadence. After 30 seconds hidden, stdin closes and the stream is reaped. Showing
the window resumes it; paused status stays explicitly stale. Exit is idempotent after completion
and publishes its final snapshot. Settings also retain the requested data
folder, prepared manifest, NPU option and effort. Preflight records remain
bounded and available to the Prepare page. The embedded J1 schema was refreshed
for the current required Diagnose schema version and unprepared Status error.
See [the Wails adapter](../wails/README.md) for the binding contract and
[scenario provenance](../tests/scenarios/README.md) for recorded and fixture-derived fields.

The guard folder holds the shared hold files, GPU timing lock and
`WINDOW-OPEN.txt` test-window sentinel. Put Go 1.27.2 and the C compiler on PATH.
`gui/tests/proof-c2.ps1 -EvidenceDir <folder>` runs the contract proof subset;
`gui/tests/proof-u4.ps1 -EvidenceDir <folder> -GuardDir <guards>` runs the
generation and Go proof subset, with optional `-Race`. Both retain their
original suite checks and write evidence to the supplied folder.
