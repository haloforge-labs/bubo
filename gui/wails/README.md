# Bubo desktop app

Wails v2.16.0 hosts the React frontend in `frontend/`. The supervisor owns CLI
processes, persistence and the typed View; this module provides binding methods,
events and OS window, dialog and tray API glue. `supervisor.AppService` owns
the settings copy, close/tray decisions, model-folder rules, LAN address
selection and credential staleness. Follow
[ARCHITECTURE.md](../ARCHITECTURE.md) and [the frontend guide](frontend/README.md).

The frontend reads `GetView(): supervisor.View` and uses typed action methods.
`PatchPreferences` changes only edited GUI preferences; `SetSwitchOverrides`
stores registry requests for CLI validation. Action errors retain CLI codes and
remedies. The update service owns update actions and progress. Stream names are
declared in `events.go`; secrets are scrubbed before storage or publication.
Missing CLI facts remain unavailable.

The CLI is the package's `bin/bubo.exe` beside the GUI executable, or the explicit
`-cli` override. `runtime_path` is a separate folder override: empty delegates to
the CLI; custom paths reach owned children through `BUBO_RUNTIME_DIR`. Model
confirmation retains the runtime folder used by its displayed plan.

The supervisor retains Job Object ownership, identity checks and its per-user
single-instance lease. Closing an owned running server offers keep running,
stop and quit, or cancel. Attached external servers keep running. Tray failure
returns an explicit unsupported result. Tests inject native adapters and do not
open a visible window.

## Build and verify

Use Go 1.27.2 and Node 24.19+. Dependencies are pinned in `go.mod`, `go.sum`,
`frontend/package.json` and its lockfile. From this directory:

```powershell
pwsh -NoProfile -File ./build.ps1 -OutputDir C:/bubo/run/app/exe
```

The script runs Wails with `production,wv2runtime.error`, builds the updater
helper, checks executable dependencies and records sizes and hashes. Wails
builds the production frontend before embedding `frontend/dist` and selects
the Windows GUI subsystem. Do not launch the executable during headless
proof. The real app requires the Microsoft WebView2 runtime; the runtime error
tag prevents an interactive installation prompt.

Only `frontend/dist/.gitkeep` is committed. Vite clears stale assets and restores
that placeholder after building. A fresh Go-only build compiles with the
placeholder; build the frontend before packaging. `build-frontend.ps1` builds it
independently.

Release packaging must supply explicit engine/model change decisions and a
nonempty user-facing update summary. See the required flags in
[Building a release manifest](../updater/README.md#building-a-release-manifest).
The package's binary provenance alone does not establish payload reuse.

Run `pwsh -NoProfile -File gui/tests/test.ps1` from the repository root for every
Go module, generated-code and dependency guards, and frontend checks. See
`gui/tests/scenarios/README.md` for recording provenance and fake-CLI replay.

Regenerate copies, scenarios and bindings with the pinned tools:

```powershell
cd gui/supervisor
go generate
cd ../tests
go run ./scenarios
cd ../wails
& "$(go env GOPATH)/bin/wails.exe" generate module -nocolour
```

`wailsjsdir` and `frontend:dir` both select `frontend`. The `bindings` entry point
reflects `&App{}` without a lease, backend, settings, credential manager or GPU
source. `TestBindingsGenerationIgnoresHeldInstanceLease` verifies this headless
path. Generated Wails conversion helpers are excluded from handwritten-code
lint; the View remains typed.
