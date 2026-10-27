//go:build windows

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// creationTimeOf returns the decimal FILETIME of a live process, the same text
// the CLI contract uses for process_creation_time.
func creationTimeOf(pid uint32) (string, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
	if err != nil {
		return "", err
	}
	defer windows.CloseHandle(h)
	var c, e, k, u windows.Filetime
	if err = windows.GetProcessTimes(h, &c, &e, &k, &u); err != nil {
		return "", err
	}
	return strconv.FormatUint(uint64(c.HighDateTime)<<32|uint64(c.LowDateTime), 10), nil
}

// imagePathOf reads a process's full executable path (query rights only).
func imagePathOf(pid uint32) (string, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
	if err != nil {
		return "", err
	}
	defer windows.CloseHandle(h)
	buf := make([]uint16, windows.MAX_LONG_PATH)
	n := uint32(len(buf))
	if err = windows.QueryFullProcessImageName(h, 0, &buf[0], &n); err != nil {
		return "", err
	}
	return windows.UTF16ToString(buf[:n]), nil
}

// isLive reports whether the exact process (pid and creation time) is alive.
// A different creation time means the PID was reused: that is not our process.
func isLive(pid uint32, creation string) bool {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)
	if st, _ := windows.WaitForSingleObject(h, 0); st != uint32(windows.WAIT_TIMEOUT) {
		return false
	}
	var c, e, k, u windows.Filetime
	if windows.GetProcessTimes(h, &c, &e, &k, &u) != nil {
		return true
	}
	return strconv.FormatUint(uint64(c.HighDateTime)<<32|uint64(c.LowDateTime), 10) == creation
}

// terminateExact ends one process by PID after it checks the creation time.
// It never walks a tree and never selects by name or port.
func terminateExact(pid uint32, creation string) (string, error) {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.PROCESS_TERMINATE, false, pid)
	if err != nil {
		return "already gone", nil
	}
	defer windows.CloseHandle(h)
	if st, _ := windows.WaitForSingleObject(h, 0); st != uint32(windows.WAIT_TIMEOUT) {
		return "already gone", nil
	}
	var c, e, k, u windows.Filetime
	if err = windows.GetProcessTimes(h, &c, &e, &k, &u); err != nil {
		return "", err
	}
	if got := strconv.FormatUint(uint64(c.HighDateTime)<<32|uint64(c.LowDateTime), 10); got != creation {
		return "pid reused by another process; not touched", nil
	}
	if err = windows.TerminateProcess(h, 1); err != nil {
		return "", err
	}
	if st, _ := windows.WaitForSingleObject(h, 10000); st != windows.WAIT_OBJECT_0 {
		return "", fmt.Errorf("pid %d did not exit after TerminateProcess", pid)
	}
	return "terminated by exact pid", nil
}

// Ledger records every process this tool starts or owns, with creation time.
type Ledger struct {
	mu   sync.Mutex
	rec  *Recorder
	file string
	list []PIDRecord
}

func NewLedger(rec *Recorder, out string) *Ledger {
	return &Ledger{rec: rec, file: filepath.Join(out, "pids.jsonl")}
}

// Add records a process once. It writes and syncs the ledger file at once, so
// a crash still leaves the list of what to stop.
func (l *Ledger) Add(pid uint32, creation, role string) {
	if pid == 0 || creation == "" {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, p := range l.list {
		if p.PID == pid && p.CreationTime == creation {
			return
		}
	}
	rec := PIDRecord{PID: pid, CreationTime: creation, Role: role, AddedAt: stamp()}
	l.list = append(l.list, rec)
	if f, err := os.OpenFile(l.file, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600); err == nil {
		b, _ := json.Marshal(rec)
		f.Write(append(b, '\n'))
		f.Sync()
		f.Close()
	}
	l.rec.Logf("ledger: pid %d created %s role %s", pid, creation, role)
}

func (l *Ledger) Snapshot() []PIDRecord {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]PIDRecord(nil), l.list...)
}

// Alive lists the ledger processes that are still running.
func (l *Ledger) Alive() []PIDRecord {
	var out []PIDRecord
	for _, p := range l.Snapshot() {
		if isLive(p.PID, p.CreationTime) {
			out = append(out, p)
		}
	}
	return out
}

// Cleanup stops only ledger processes that are still alive, by exact PID, and
// fills the end state of every record. It returns the PIDs it had to stop and
// any that could not be stopped.
func (l *Ledger) Cleanup() (stopped []PIDRecord, stuck []PIDRecord) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for i := range l.list {
		p := &l.list[i]
		if isLive(p.PID, p.CreationTime) {
			action, err := terminateExact(p.PID, p.CreationTime)
			if err != nil {
				p.EndAction = "terminate failed: " + err.Error()
			} else {
				p.EndAction = action
				stopped = append(stopped, *p)
			}
			l.rec.Logf("cleanup: pid %d: %s", p.PID, p.EndAction)
		} else {
			p.EndAction = "exited by itself"
		}
		alive := isLive(p.PID, p.CreationTime)
		p.AliveAtEnd = &alive
		if alive {
			stuck = append(stuck, *p)
		}
	}
	l.rec.mu.Lock()
	l.rec.report.PIDs = append([]PIDRecord(nil), l.list...)
	l.rec.mu.Unlock()
	return
}

// hiddenCommand builds a command with no console window and no inherited
// stdin. Its context kills only that exact process.
func hiddenCommand(ctx context.Context, exe string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
	cmd.WaitDelay = 3 * time.Second
	return cmd
}

type memStatus struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

// hostMemory returns total and available physical memory in bytes.
func hostMemory() (total, avail uint64, err error) {
	var m memStatus
	m.Length = uint32(unsafe.Sizeof(m))
	proc := windows.NewLazySystemDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
	r, _, e := proc.Call(uintptr(unsafe.Pointer(&m)))
	if r == 0 {
		return 0, 0, e
	}
	return m.TotalPhys, m.AvailPhys, nil
}

// freeDisk returns the free bytes of the volume that holds path.
func freeDisk(path string) (uint64, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var free, total, totalFree uint64
	if err = windows.GetDiskFreeSpaceEx(p, &free, &total, &totalFree); err != nil {
		return 0, err
	}
	return free, nil
}

func unsafeSizeof(e windows.ProcessEntry32) uintptr { return unsafe.Sizeof(e) }
