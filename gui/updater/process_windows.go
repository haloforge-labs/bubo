//go:build windows

package updater

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"golang.org/x/sys/windows"
)

type processIdentity struct {
	PID     uint32 `json:"pid"`
	Created uint64 `json:"created"`
}

type watchedProcess struct {
	identity processIdentity
	handle   windows.Handle
}

func watchProcess(id processIdentity) (*watchedProcess, error) {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, id.PID)
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return &watchedProcess{identity: id}, nil
	}
	if err != nil {
		return nil, err
	}
	var created, exited, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(h, &created, &exited, &kernel, &user); err != nil {
		windows.CloseHandle(h)
		return nil, err
	}
	stamp := uint64(created.HighDateTime)<<32 | uint64(created.LowDateTime)
	if id.Created != 0 && id.Created != stamp {
		windows.CloseHandle(h)
		return &watchedProcess{identity: id}, nil
	}
	return &watchedProcess{identity: processIdentity{PID: id.PID, Created: stamp}, handle: h}, nil
}

func (p *watchedProcess) close() {
	if p.handle != 0 {
		windows.CloseHandle(p.handle)
		p.handle = 0
	}
}
func (p *watchedProcess) exited() (bool, error) {
	if p.handle == 0 {
		return true, nil
	}
	status, err := windows.WaitForSingleObject(p.handle, 0)
	if err != nil {
		return false, err
	}
	return status == windows.WAIT_OBJECT_0, nil
}

func (p *watchedProcess) wait(ctx context.Context) error {
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for {
		done, err := p.exited()
		if err != nil || done {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func snapshotProcesses(pids []uint32) ([]processIdentity, error) {
	seen := make(map[uint32]bool)
	ids := make([]processIdentity, 0, len(pids))
	for _, pid := range pids {
		if pid == 0 {
			return nil, fmt.Errorf("PID zero is invalid")
		}
		if seen[pid] {
			continue
		}
		seen[pid] = true
		p, err := watchProcess(processIdentity{PID: pid})
		if err != nil {
			return nil, fmt.Errorf("cannot identify PID %d: %w", pid, err)
		}
		ids = append(ids, p.identity)
		p.close()
	}
	return ids, nil
}

func startDetached(exe, cwd string, args ...string) (*watchedProcess, error) {
	cmd := exec.Command(exe, args...)
	cmd.Dir = cwd
	hideProcess(cmd)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p, err := watchProcess(processIdentity{PID: uint32(cmd.Process.Pid)})
	cmd.Process.Release()
	return p, err
}
