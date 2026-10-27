// Package updater implements Bubo's explicit check, download, and install flow.
//
// This package owns the portable, multi-file ZIP transaction because Velopack
// has no official no-cgo Go SDK, requires a different installed layout, and
// does not provide the required post-start health rollback. Single-executable
// go-selfupdate libraries cannot transact this package. We consequently own
// recovery, locking, and diagnostics here. net/http and crypto/tls enforce the
// feed policy, crypto/sha256 verifies every completed archive (including resumed
// downloads), archive/zip stages it, and os/exec starts the external helper.
// Only golang.org/x/sys/windows is needed beyond the standard library.
//
// Check never downloads an asset. Download never installs. The caller must stop
// the server gracefully, supply its exact PID together with the GUI PID, call
// Install, then exit. There are no process kills, runtime migrations, or writes
// to models, preferences, credentials, runtime, or logs in this package.
package updater
