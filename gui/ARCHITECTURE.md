# Bubo desktop app

See [the Wails guide](wails/README.md) for build and development commands.

## 3. One source per fact

The desktop app consumes the CLI contract. See [the CLI contract](../docs/CLI-CONTRACT.md) and [the supervisor guide](supervisor/README.md).

## 8. Build
`wails build` (wails CLI v2.16.0) builds the frontend and the exe. frontend/dist is not committed (a tracked placeholder
keeps `go:embed` compiling). See `gui/updater/README.md` for the release package and updater contract;
the updater helper lives in gui/updater (separate small exe; waits for our processes, swaps the folder, rolls back).
The `bubo_test` build tag compiles in an instance suffix, its own login-start path, its own credential
name, start-hidden, loopback-only listening, copy capture, fake native dialogs, folder opening and tray, a fixed CDP port
and updater manifest settings, so tests can never touch the normal application state. Production builds have none of it.

