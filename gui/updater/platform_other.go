//go:build !windows

package updater

import (
	"fmt"
	"os"
	"os/exec"
)

func noReparse(name string) error {
	i, err := os.Lstat(name)
	if err != nil {
		return err
	}
	if i.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("symlink refused: %s", name)
	}
	return nil
}

func openSharedRead(name string) (*os.File, error) { return os.Open(name) }
func diskFree(string) (uint64, error) {
	return 0, fail(CodeUnsupported, "installation requires Windows")
}
func replaceFile(from, to string) error { return os.Rename(from, to) }
func moveDirectory(string, string) error {
	return fail(CodeUnsupported, "installation requires Windows")
}
func hideProcess(*exec.Cmd) {}

func sharingViolation(error) bool { return false }
func lockTransaction(string) (func(), error) {
	return nil, fail(CodeUnsupported, "installation requires Windows")
}

func lockInstall(string) (func(), error) {
	return nil, fail(CodeUnsupported, "installation requires Windows")
}

func lockHandoff(string) (func(), error) {
	return nil, fail(CodeUnsupported, "installation requires Windows")
}
