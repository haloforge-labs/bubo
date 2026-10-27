//go:build windows

package updater

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"golang.org/x/sys/windows"
)

// A per-install kernel mutex also serializes helpers for different asset hashes.
// It creates no file outside the three permitted transaction locations.
func lockInstall(install string) (func(), error) { return lockNamed("helper", install) }
func lockHandoff(install string) (func(), error) { return lockNamed("handoff", install) }

func lockNamed(scope, install string) (func(), error) {
	// Win32 mutex ownership belongs to an OS thread, not a Go goroutine.
	runtime.LockOSThread()
	key := sha256.Sum256([]byte(strings.ToLower(filepath.Clean(install))))
	name, err := windows.UTF16PtrFromString(fmt.Sprintf("Local\\BuboUpdate-%s-%x", scope, key))
	if err != nil {
		runtime.UnlockOSThread()
		return nil, err
	}
	h, err := windows.CreateMutex(nil, false, name)
	if err != nil && !errors.Is(err, windows.ERROR_ALREADY_EXISTS) {
		runtime.UnlockOSThread()
		return nil, err
	}
	status, err := windows.WaitForSingleObject(h, 0)
	if err != nil || (status != windows.WAIT_OBJECT_0 && status != windows.WAIT_ABANDONED) {
		windows.CloseHandle(h)
		runtime.UnlockOSThread()
		if err != nil {
			return nil, err
		}
		return nil, errInstallBusy
	}
	return func() { windows.ReleaseMutex(h); windows.CloseHandle(h); runtime.UnlockOSThread() }, nil
}

func sharingViolation(err error) bool {
	return errors.Is(err, windows.ERROR_SHARING_VIOLATION) || errors.Is(err, windows.ERROR_LOCK_VIOLATION) || errors.Is(err, windows.ERROR_ACCESS_DENIED)
}

func lockTransaction(root string) (func(), error) {
	name := root + string(windowsPathSeparator) + "transaction.lock"
	if err := safeAncestors(name); err != nil {
		return nil, err
	}
	p, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return nil, err
	}
	h, err := windows.CreateFile(p, windows.GENERIC_READ|windows.GENERIC_WRITE, 0, nil, windows.OPEN_ALWAYS, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, err
	}
	return func() { windows.CloseHandle(h) }, nil
}

const windowsPathSeparator = '\\'

// Readers of atomically replaced metadata must permit deletion of the old
// directory entry. Go's ordinary os.Open omits FILE_SHARE_DELETE on Windows.
func openSharedRead(name string) (*os.File, error) {
	p, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return nil, err
	}
	h, err := windows.CreateFile(p, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(h), name), nil
}

func noReparse(name string) error {
	p, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return err
	}
	attrs, err := windows.GetFileAttributes(p)
	if err != nil {
		return err
	}
	if attrs&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		return fmt.Errorf("reparse point refused: %s", name)
	}
	return nil
}

func diskFree(name string) (uint64, error) {
	p, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return 0, err
	}
	var free, total, totalFree uint64
	err = windows.GetDiskFreeSpaceEx(p, &free, &total, &totalFree)
	return free, err
}

func replaceFile(from, to string) error {
	return moveFile(from, to, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
}

func moveDirectory(from, to string) error { return moveFile(from, to, windows.MOVEFILE_WRITE_THROUGH) }

func moveFile(from, to string, flags uint32) error {
	f, err := windows.UTF16PtrFromString(from)
	if err != nil {
		return err
	}
	t, err := windows.UTF16PtrFromString(to)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(f, t, flags)
}

func hideProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
}
