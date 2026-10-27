# Updater integration

This separate Go module implements the portable Windows package transaction.
Its only external module is the pinned `golang.org/x/sys` in `go.mod`/`go.sum`.
See the package comment for the Velopack and single-executable updater tradeoffs.
No network request is made by construction, installation, result reading, or recovery.

The supervisor owns consent, startup preference, server shutdown, and application
lifetime. U4 wires these APIs and the GUI half of the health protocol:

```go
client, failure := updater.New(updater.Config{
    InstalledVersion: installedVersionFromCLI,
    InstallDir:       absolutePackageRoot,
    ProcessIDs:       exactGUIAndServerPIDs,
})
// Handle failure and defer client.Close().

check := client.Check(ctx) // UpToDate, Available, or Failed with typed details.
// Only after explicit Download consent:
staged, failure := client.Download(ctx, onProgress)
// Stop the server gracefully, retain its PID, and obtain explicit Install consent.
handoff, failure := client.Install(staged)
// On a successful handoff, exit the GUI. Do not wait for the helper in the GUI.

last, failure := client.LastResult() // nil, nil means no result yet.
// If rollback is pending, close the candidate/server, explicitly Recover, then exit:
handoff, failure = client.Recover()
```

Actual signatures:

```go
func New(Config) (*Client, *Failure)
func (*Client) Check(context.Context) CheckResult
func (*Client) Download(context.Context, func(Progress)) (*Staged, *Failure)
func (*Client) Install(*Staged) (*Handoff, *Failure)
func (*Client) LastResult() (*InstallResult, *Failure)
func (*Client) Recover() (*Handoff, *Failure)
func (*Client) Close()
```

`ProcessIDs` is a `func() ([]uint32, error)` evaluated at handoff. It must include
the GUI and every server child that could hold package files; use the supervisor's
owned processes. The helper opens exact process handles, checks the captured
creation times against PID reuse, and waits without terminating anything. Default
process-exit and health deadlines are 60 and 30 seconds, respectively. They are
constructor parameters for tests and supervisor policy.

The supervisor should call Check once at GUI start unless the user disabled the
default-on preference; an explicit Check action may call it again. Check makes
one logical GET of `ManifestURL` (redirects use HTTPS and the allowlist), with
`User-Agent: bubo/<installed version>`, no REST API, cookies, credentials, or
telemetry. Network failures clear the available selection. Download and Install
are never invoked by Check. Allowlisted hosts and TLS roots are constructor
parameters; production defaults are `github.com` and `*.githubusercontent.com`.
Proxy discovery is disabled so this package never sends credentials through an
environment-provided proxy. An unsupported manifest or required intermediate
release returns a typed failure and a manual/intermediate route; there is no
implicit second manifest request or intermediate installation.
For `intermediate_required`, the supervisor may explicitly create a client with
that failure's exact tagged manifest URL and call Check once, then use the same
Download/Install flow. A saved feed's named `via` is the only exception to
newest-seen ordering; it must still advance the installed version. The highest
observed release remains saved, and arbitrary older tagged feeds stay refused.

## Files and preservation

For an install at `<parent>/<name>`, the staging root is
`<parent>/.<name>.bubo-update`. Each asset has its own hash-named transaction
folder containing `archive.part`, the fully verified `archive.zip`, `package/`,
the copied helper, `transaction.json`, and `.bubo-update-result.json`.
`active.json` at the staging root selects the last handoff; `seen.json` prevents
accepting an older previously observed manifest across GUI restarts. The helper
writes only its transaction folder, the install, and `<name>.previous`.
Result files stay in staging so a rolled-back old package remains byte-identical.
Staging files are retained for resume and interrupted-update recovery. Do not
delete a staging folder while its helper or transaction journal exists.

The reusable partial is counted exactly once. With free space already excluding
existing files, the additional requirement is
`archive_bytes - existing_partial_bytes + unpacked_bytes + installed_package_bytes`.
Unrelated partials consume free space without receiving credit. Even a completely
downloaded or resumed archive is hashed before staging. A mismatching archive is
removed where possible; failures say whether it was kept and how to retry.

The ZIP's declared top folder is stripped, validated against the release name,
then installed at the user's existing stable folder name. Required package files
are `bin/bubo.exe`, `bin/Bubo.Control.Web.exe`, and `bin/bubo-update.exe`. Every ZIP
entry is bounded by the declared unpacked total, CRC-checked, constrained below
260 UTF-16 path units in staging/install/rollback, and checked for traversal,
devices, case collisions, symlinks and reparse points. Download/Install recheck
the staged tree seal so modified staging cannot be silently installed.

The update swaps the complete package. User edits to package files are replaced.
Models, settings, logs, API keys, wheel cache and AMD runtime must live outside
the package; the updater does not discover, migrate, or write any of them. The
release must contain K1's external-runtime change before this updater is enabled.

## Health and rollback: inherited by U4

The exported constants in `types.go` define the protocol. The helper launches
`bin/Bubo.Control.Web.exe` with `HealthFlag <random-token>` and
`HealthDeadlineFlag <UTC-RFC3339Nano-deadline>`. U4 must parse these before normal
startup, initialize the GUI/supervisor, then atomically write
`updater.HealthContent(token)` to `<install>/<updater.HealthFileName>`. It must
exit if initialization fails or cannot acknowledge health by the deadline. Health
must not require starting an engine, loading a model, or preparing a runtime.

The helper keeps `.previous` until that exact token is received from a live new
GUI. A failed/exited candidate is moved back to staging and `.previous` is
restored before the old GUI is started. Sharing violations retry six times with
50/100/200/400/800 ms delays. Install first copies the shipped helper to a temp
folder inside same-volume staging and uses that as its working directory, so
neither its image nor its working directory blocks the folder rename.
Metadata readers use Windows deletion sharing, and atomic metadata replacements
also retry sharing violations, so GUI polling cannot itself break the journal.

An uncooperative GUI still running after both deadlines makes rollback
`rollback_pending`: Windows cannot move its live image and process kills are
forbidden. The old folder is preserved, no second GUI is started, and explicit
Recover retries after that process exits. Package locks can also defer rollback.
The caller must display this typed failure instead of claiming rollback passed.

Each rename's intent and the original/staged tree identities are flushed to the
journal first. Recovery covers interruption before journal creation, before the first rename, between the
renames, before GUI start, after restoration before a result was written, and
after health was committed before cleanup. An interruption in the tiny interval
between process launch and persisting its identity is handled conservatively:
close the candidate GUI/server and provide their exact PIDs to Recover; a sharing
violation preserves `.previous` rather than forcing a rename. If no current GUI
can launch, invoke the already copied helper with the saved `active.json` paths,
version and seal, `--recover`, and the exact PIDs to wait for. `New` also permits
an absent current folder when its safe `.previous` exists for this recovery.
This is filesystem recovery evidence, not a power-cut durability certification.

## Building a release manifest

The offline builder reads binary provenance, sizes and hashes from the package.
Release decisions must be supplied explicitly; omitted flags fail the build,
including when the intended value is false, zero or an empty model list:

```powershell
go -C gui/updater run ./cmd/bubo-release -Zip <package.zip> -Out <bubo-release.json> `
  -MinUpdatableFrom 0.1.0 -Released <UTC-RFC3339> -Notes <summary.txt> `
  -RuntimeChanged=false -RuntimeDownloadBytes=0 -ModelsChanged=false -ModelsNeedsPrepare=
```

`Notes` is a required UTF-8 file containing a nonempty user-facing summary, as
displayed verbatim on the update offer. Use release-approved prose from
WHATSNEW.md; do not use qualification labels, revisions or build instructions.
The scenario generator displays the manifest summary without substituting text.

`RuntimeChanged` and `ModelsChanged` are comparisons against the supported prior
release, supplied by the release owner. `RuntimeDownloadBytes` is the actual
engine transfer size; unchanged runtime requires zero. `ModelsNeedsPrepare` is
an explicitly supplied comma-separated list of catalogue IDs, or an explicitly
empty value when none need preparation. The manifest currently has no model
download-size field, so no model byte count is invented. The Go API likewise
requires non-nil decision pointers and a non-nil model list. This command does
not install dependencies or execute packaged binaries.

## Local proof commands

```powershell
pwsh -NoProfile -File gui/updater/proof.ps1 -Go go
```

The script prints vet/package counts, all eight named cases, total/subtest counts,
the dependency guard, and the hidden helper's byte size. Tests use only loopback
TLS servers, temporary folders, exact test-owned PIDs, and fake GUI executables
built headlessly with `CGO_ENABLED=0`. The PE header assertion checks the Windows
GUI subsystem. No production GUI, CLI engine, server, model or GPU is started.
Lane E can call this script from its test entry point. The helper release build is:

```powershell
go build -ldflags=-H=windowsgui -o <package>/bin/bubo-update.exe ./cmd/bubo-update
```
