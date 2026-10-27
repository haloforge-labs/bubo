//go:build !windows

package updater

import "context"

type processIdentity struct {
	PID     uint32 `json:"pid"`
	Created uint64 `json:"created"`
}
type watchedProcess struct{ identity processIdentity }

func watchProcess(processIdentity) (*watchedProcess, error) {
	return nil, fail(CodeUnsupported, "process handoff requires Windows")
}
func (*watchedProcess) close() {}
func (*watchedProcess) exited() (bool, error) {
	return false, fail(CodeUnsupported, "process handoff requires Windows")
}
func (*watchedProcess) wait(context.Context) error {
	return fail(CodeUnsupported, "process handoff requires Windows")
}
func snapshotProcesses([]uint32) ([]processIdentity, error) {
	return nil, fail(CodeUnsupported, "process handoff requires Windows")
}
func startDetached(string, string, ...string) (*watchedProcess, error) {
	return nil, fail(CodeUnsupported, "process handoff requires Windows")
}
