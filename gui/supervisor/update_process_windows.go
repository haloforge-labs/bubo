//go:build windows

package supervisor

import (
	"context"
	"errors"
	"fmt"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// x/sys exposes Job Object queries but not IsProcessInJob.
var isProcessInJob = windows.NewLazySystemDLL("kernel32.dll").NewProc("IsProcessInJob")

// Windows' Job Object is the source of child ownership; never enumerate by
// process name or walk unrelated system processes. x/sys lacks this variable
// length struct, so use the documented JOBOBJECT_BASIC_PROCESS_ID_LIST layout.
func (p *OwnedProcess) jobProcessIDs() ([]uint32, error) {
	if p.job == 0 {
		return nil, nil
	}
	for count := 16; count <= 65536; count *= 2 {
		buf := make([]uintptr, count+1) // two DWORD header fields = one uintptr on x64
		err := windows.QueryInformationJobObject(p.job, windows.JobObjectBasicProcessIdList, uintptr(unsafe.Pointer(&buf[0])), uint32(len(buf)*int(unsafe.Sizeof(buf[0]))), nil)
		if errors.Is(err, windows.ERROR_MORE_DATA) {
			continue
		}
		if err != nil {
			return nil, err
		}
		header := (*[2]uint32)(unsafe.Pointer(&buf[0]))
		n := int(header[1])
		if n > count {
			return nil, fmt.Errorf("invalid Job Object process count")
		}
		out := []uint32{p.PID}
		for _, pid := range buf[1 : n+1] {
			out = append(out, uint32(pid))
		}
		return out, nil
	}
	return nil, fmt.Errorf("owned Job Object process list exceeds limit")
}

// Keep handles until lane C snapshots identities at handoff. Otherwise Windows
// could reuse a stopped PID between our clean stop and the helper's snapshot.
func (p *OwnedProcess) retainUpdateProcesses() ([]uint32, func(), error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	ids, err := p.jobProcessIDs()
	var handles []windows.Handle
	release := func() {
		for _, h := range handles {
			windows.CloseHandle(h)
		}
	}
	if err != nil {
		return nil, release, err
	}
	var retained []uint32
	for _, pid := range ids {
		h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
			continue
		}
		if err != nil {
			return nil, release, err
		}
		var owned int32
		ok, _, queryErr := isProcessInJob.Call(uintptr(h), uintptr(p.job), uintptr(unsafe.Pointer(&owned)))
		if ok == 0 {
			windows.CloseHandle(h)
			return nil, release, fmt.Errorf("IsProcessInJob: %w", queryErr)
		}
		if owned == 0 {
			windows.CloseHandle(h)
			continue
		}
		handles = append(handles, h)
		retained = append(retained, pid)
	}
	return retained, release, nil
}

func (s *Supervisor) trackProcess(p *OwnedProcess) func() {
	s.processMu.Lock()
	if s.processes == nil {
		s.processes = make(map[*OwnedProcess]struct{})
	}
	s.processes[p] = struct{}{}
	s.processMu.Unlock()
	return func() { s.processMu.Lock(); delete(s.processes, p); s.processMu.Unlock() }
}

func (s *Supervisor) updateProcesses() ([]uint32, func(), error) {
	s.processMu.Lock()
	defer s.processMu.Unlock()
	var ids []uint32
	var releases []func()
	release := func() {
		for _, done := range releases {
			done()
		}
	}
	for p := range s.processes {
		owned, done, err := p.retainUpdateProcesses()
		releases = append(releases, done)
		if err != nil {
			return nil, release, err
		}
		ids = append(ids, owned...)
	}
	return ids, release, nil
}

// QuiesceUpdate uses the normal stop path and waits for completion, rather
// than treating Stop's enqueue acknowledgment as proof that the server exited.
// A failed handoff releases the action gate and leaves the supervisor usable.
func (b *NativeBackend) QuiesceUpdate(ctx context.Context) ([]uint32, func(), error) {
	b.mu.Lock()
	if b.busy || b.exiting || b.updating {
		b.mu.Unlock()
		return nil, nil, errors.New("Finish the active supervisor action first.")
	}
	b.updating = true
	s := b.s
	b.mu.Unlock()
	pids, releaseProcesses, err := s.updateProcesses()
	release := func() { releaseProcesses(); b.mu.Lock(); b.updating = false; b.mu.Unlock() }
	if err != nil {
		return nil, release, err
	}
	v := s.Snapshot()
	if !v.Initialized {
		return nil, release, errors.New("The supervisor is still initializing.")
	}
	if v.State == AttachedExternal || len(v.Discovered) > 1 {
		return nil, release, appError("update_external_server", "Stop external CLI instances before installing an update.")
	}
	if v.Instance != nil || v.State == Preparing || v.State == Starting || v.State == Ready || v.State == Stopping {
		if v.State != Stopping {
			if r := <-s.Stop(); r.Err != nil {
				return nil, release, r.Err
			}
		}
		tick := time.NewTicker(10 * time.Millisecond)
		defer tick.Stop()
		for {
			v = s.Snapshot()
			if v.Instance == nil && (v.State == Stopped || v.State == NotPrepared || v.State == Failed) {
				break
			}
			select {
			case <-ctx.Done():
				return nil, release, ctx.Err()
			case <-tick.C:
			}
		}
		if v.Forced || v.State == Failed {
			return nil, release, errors.New("The CLI did not complete a clean stop; update was not handed off.")
		}
	}
	return pids, release, nil
}

// PauseForUpdate drains bounded model queries without terminating them. A
// running download must be explicitly canceled before installation.
func (m *ModelService) PauseForUpdate(ctx context.Context) ([]uint32, func(), error) {
	m.mu.Lock()
	if m.closed || m.updating || m.busy() {
		m.mu.Unlock()
		return nil, nil, errors.New("Finish or cancel model work before installing an update.")
	}
	m.updating = true
	var processes []*OwnedProcess
	for p := range m.processes {
		processes = append(processes, p)
	}
	m.mu.Unlock()
	var releases []func()
	release := func() {
		for _, done := range releases {
			done()
		}
		m.mu.Lock()
		m.updating = false
		m.mu.Unlock()
	}
	var pids []uint32
	for _, p := range processes {
		ids, done, err := p.retainUpdateProcesses()
		releases = append(releases, done)
		if err != nil {
			return nil, release, err
		}
		pids = append(pids, ids...)
		select {
		case <-p.finished:
		case <-ctx.Done():
			return nil, release, ctx.Err()
		}
	}
	return pids, release, nil
}

func (b *NativeBackend) QuiesceForUpdate(models *ModelService) func(context.Context) ([]uint32, func(), error) {
	return func(ctx context.Context) ([]uint32, func(), error) {
		var modelPIDs []uint32
		releaseModels := func() {}
		if models != nil {
			var err error
			modelPIDs, releaseModels, err = models.PauseForUpdate(ctx)
			if err != nil {
				return nil, releaseModels, err
			}
		}
		pids, releaseBackend, err := b.QuiesceUpdate(ctx)
		return append(pids, modelPIDs...), func() {
			if releaseBackend != nil {
				releaseBackend()
			}
			releaseModels()
		}, err
	}
}
