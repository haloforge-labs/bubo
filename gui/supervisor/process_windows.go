//go:build windows

package supervisor

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

type ProcessResult struct {
	ExitCode      uint32
	DrainTimedOut bool
	Err           error
}
type OwnedProcess struct {
	mu           sync.Mutex
	inputMu      sync.Mutex
	job          windows.Handle
	process      windows.Handle
	stdin        *os.File
	stdout       *os.File
	stderr       *os.File
	PID          uint32
	CreationTime string
	Done         chan ProcessResult
	StdoutEOF    chan struct{}
	drained      chan struct{}
	finished     chan struct{}
}

// CommandLine uses the Windows CRT/CommandLineToArgvW quoting rules. No shell.
func CommandLine(exe string, args []string) string {
	all := append([]string{exe}, args...)
	for i, a := range all {
		all[i] = windows.EscapeArg(a)
	}
	return strings.Join(all, " ")
}

// launchOwned assigns the suspended process before any child code can run.
// Tests can inject assignment failure without replacing the ownership boundary.
func launchOwned(exe string, args []string, drainTimeout time.Duration, out, errLine func([]byte, bool), assign func(windows.Handle, windows.Handle) error) (*OwnedProcess, error) {
	return launchOwnedEnv(exe, args, nil, drainTimeout, out, errLine, assign)
}

// ServeKeyEnv is the environment variable through which the GUI hands the API
// key to `bubo serve`. A key on the command line is visible to every tool that
// lists process command lines; the environment block is not.
const ServeKeyEnv = "BUBO_API_KEY"

const RuntimeDirEnv = "BUBO_RUNTIME_DIR"

// childEnvironment returns the parent environment without any inherited
// API key/runtime overrides, then adds explicit values. Empty runtime selects
// the shared CLI default even if the GUI inherited a runtime override.
func childEnvironment(parent []string, key, runtimeDir string) []string {
	out := make([]string, 0, len(parent)+1)
	for _, kv := range parent {
		if name, _, ok := strings.Cut(kv, "="); ok && (strings.EqualFold(name, ServeKeyEnv) || strings.EqualFold(name, RuntimeDirEnv)) {
			continue
		}
		out = append(out, kv)
	}
	if key != "" {
		out = append(out, ServeKeyEnv+"="+key)
	}
	if runtimeDir != "" {
		out = append(out, RuntimeDirEnv+"="+runtimeDir)
	}
	return out
}

func environmentBlock(env []string) (*uint16, error) {
	var block []uint16
	for _, kv := range env {
		if strings.ContainsRune(kv, 0) {
			return nil, fmt.Errorf("environment entry contains NUL")
		}
		u, err := windows.UTF16FromString(kv)
		if err != nil {
			return nil, err
		}
		block = append(block, u...)
	}
	block = append(block, 0)
	if len(env) == 0 {
		block = append(block, 0)
	}
	return &block[0], nil
}

// launchOwnedEnv is launchOwned with an explicit child environment. A nil env
// inherits the parent environment.
func launchOwnedEnv(exe string, args []string, env []string, drainTimeout time.Duration, out, errLine func([]byte, bool), assign func(windows.Handle, windows.Handle) error) (*OwnedProcess, error) {
	if !filepath.IsAbs(exe) {
		return nil, fmt.Errorf("CLI path must be absolute")
	}
	info, err := os.Stat(exe)
	if err != nil || info.IsDir() {
		return nil, fmt.Errorf("CLI path is not a file: %v", err)
	}
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, err
	}
	limit := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	limit.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err = windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&limit)), uint32(unsafe.Sizeof(limit))); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	var files []*os.File
	cleanup := func() {
		for _, f := range files {
			f.Close()
		}
		windows.CloseHandle(job)
	}
	inR, inW, err := os.Pipe()
	if err != nil {
		cleanup()
		return nil, err
	}
	files = append(files, inR, inW)
	outR, outW, err := os.Pipe()
	if err != nil {
		cleanup()
		return nil, err
	}
	files = append(files, outR, outW)
	errR, errW, err := os.Pipe()
	if err != nil {
		cleanup()
		return nil, err
	}
	files = append(files, errR, errW)
	handles := []windows.Handle{windows.Handle(inR.Fd()), windows.Handle(outW.Fd()), windows.Handle(errW.Fd())}
	for _, h := range handles {
		if err = windows.SetHandleInformation(h, windows.HANDLE_FLAG_INHERIT, windows.HANDLE_FLAG_INHERIT); err != nil {
			cleanup()
			return nil, err
		}
	}
	attrs, err := windows.NewProcThreadAttributeList(1)
	if err != nil {
		cleanup()
		return nil, err
	}
	defer attrs.Delete()
	if err = attrs.Update(windows.PROC_THREAD_ATTRIBUTE_HANDLE_LIST, unsafe.Pointer(&handles[0]), uintptr(len(handles))*unsafe.Sizeof(handles[0])); err != nil {
		cleanup()
		return nil, err
	}
	si := windows.StartupInfoEx{}
	si.Cb = uint32(unsafe.Sizeof(si))
	si.Flags = windows.STARTF_USESTDHANDLES
	si.StdInput = handles[0]
	si.StdOutput = handles[1]
	si.StdErr = handles[2]
	si.ProcThreadAttributeList = attrs.List()
	app, err := windows.UTF16PtrFromString(exe)
	if err != nil {
		cleanup()
		return nil, err
	}
	cmd, err := windows.UTF16PtrFromString(CommandLine(exe, args))
	if err != nil {
		cleanup()
		return nil, err
	}
	pi := windows.ProcessInformation{}
	flags := uint32(windows.CREATE_SUSPENDED | windows.CREATE_NO_WINDOW | windows.EXTENDED_STARTUPINFO_PRESENT)
	var envp *uint16
	if env != nil {
		if envp, err = environmentBlock(env); err != nil {
			cleanup()
			return nil, err
		}
		flags |= windows.CREATE_UNICODE_ENVIRONMENT
	}
	if err = windows.CreateProcess(app, cmd, nil, nil, true, flags, envp, nil, &si.StartupInfo, &pi); err != nil {
		cleanup()
		return nil, err
	}
	if assign == nil {
		assign = windows.AssignProcessToJobObject
	}
	if err = assign(job, pi.Process); err != nil {
		windows.TerminateProcess(pi.Process, 1)
		windows.WaitForSingleObject(pi.Process, windows.INFINITE)
		windows.CloseHandle(pi.Thread)
		windows.CloseHandle(pi.Process)
		cleanup()
		return nil, fmt.Errorf("Job assignment failed: %w", err)
	}
	creation, err := creationTime(pi.Process)
	if err != nil {
		windows.TerminateJobObject(job, 1)
		windows.CloseHandle(pi.Thread)
		windows.CloseHandle(pi.Process)
		cleanup()
		return nil, err
	}
	inR.Close()
	outW.Close()
	errW.Close()
	p := &OwnedProcess{job: job, process: pi.Process, stdin: inW, stdout: outR, stderr: errR, PID: pi.ProcessId, CreationTime: creation, Done: make(chan ProcessResult, 1), drained: make(chan struct{}), StdoutEOF: make(chan struct{}), finished: make(chan struct{})}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); defer close(p.StdoutEOF); drain(outR, 1<<20, out) }()
	go func() { defer wg.Done(); drain(errR, 64<<10, errLine) }()
	go func() { wg.Wait(); close(p.drained) }()
	_, resumeErr := windows.ResumeThread(pi.Thread)
	windows.CloseHandle(pi.Thread)
	if resumeErr != nil {
		p.Force()
	}
	go func() {
		result := ProcessResult{Err: resumeErr}
		_, waitErr := windows.WaitForSingleObject(pi.Process, windows.INFINITE)
		if waitErr != nil {
			result.Err = waitErr
		}
		windows.GetExitCodeProcess(pi.Process, &result.ExitCode)
		timer := time.NewTimer(drainTimeout)
		select {
		case <-p.drained:
			timer.Stop()
		case <-timer.C:
			result.DrainTimedOut = true
			p.Force()
			p.stdout.Close()
			p.stderr.Close()
			<-p.drained
		}
		p.stdout.Close()
		p.stderr.Close()
		p.mu.Lock()
		windows.CloseHandle(p.job)
		p.job = 0
		p.stdin.Close()
		windows.CloseHandle(p.process)
		p.process = 0
		p.mu.Unlock()
		close(p.finished)
		p.Done <- result
		close(p.Done)
	}()
	return p, nil
}
func (p *OwnedProcess) Force() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.job == 0 {
		return nil
	}
	return windows.TerminateJobObject(p.job, 1)
}
func (p *OwnedProcess) Input(line []byte) error {
	if len(line) > 64<<10 {
		return fmt.Errorf("input record exceeds 64 KiB")
	}
	p.inputMu.Lock()
	defer p.inputMu.Unlock()
	p.mu.Lock()
	if p.job == 0 {
		p.mu.Unlock()
		return errors.New("operation has exited")
	}
	p.mu.Unlock()
	_, err := p.stdin.Write(append(line, '\n'))
	return err
}
func (p *OwnedProcess) CloseInput() { p.stdin.Close() }
func creationTime(h windows.Handle) (string, error) {
	var c, e, k, u windows.Filetime
	if err := windows.GetProcessTimes(h, &c, &e, &k, &u); err != nil {
		return "", err
	}
	return strconv.FormatUint(uint64(c.HighDateTime)<<32|uint64(c.LowDateTime), 10), nil
}

// VerifyIdentity retains an OS handle during all identity checks. No PID-only
// action is exposed; external stop always goes through the CLI contract.
func VerifyIdentity(i Instance, exe string) error {
	if i.PID == 0 || i.ID == "" || i.ExePath == "" {
		return protocol("incomplete instance identity")
	}
	if err := ValidateEndpoint(i.Endpoint); err != nil {
		return err
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.SYNCHRONIZE, false, i.PID)
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	state, err := windows.WaitForSingleObject(h, 0)
	if err != nil || state != uint32(windows.WAIT_TIMEOUT) {
		return fmt.Errorf("instance has exited")
	}
	created, err := creationTime(h)
	if err != nil {
		return err
	}
	if created != i.CreationTime {
		return protocol("PID creation time does not match")
	}
	buf := make([]uint16, 32768)
	size := uint32(len(buf))
	if err = windows.QueryFullProcessImageName(h, 0, &buf[0], &size); err != nil {
		return err
	}
	actual := windows.UTF16ToString(buf[:size])
	// Windows reports the path the image was started from, which may run
	// through a junction: compare the real path on every side.
	actual = canonicalPath(actual)
	want, err := realPath(exe)
	if err != nil {
		return err
	}
	declared, err := realPath(i.ExePath)
	if err != nil {
		return err
	}
	if !strings.EqualFold(filepath.Clean(actual), filepath.Clean(want)) || !strings.EqualFold(filepath.Clean(actual), filepath.Clean(declared)) {
		return protocol("instance executable path does not match")
	}
	return nil
}

// realPath is the final path of an existing file: symbolic links, junctions and
// letter case resolved as Windows resolves them (Go's EvalSymlinks does not
// follow mount points). It fails when the file cannot be opened.
func realPath(p string) (string, error) {
	name, err := windows.UTF16PtrFromString(p)
	if err != nil {
		return "", err
	}
	h, err := windows.CreateFile(name, 0, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return "", err
	}
	defer windows.CloseHandle(h)
	buf := make([]uint16, 32768)
	n, err := windows.GetFinalPathNameByHandle(h, &buf[0], uint32(len(buf)), 0)
	if err != nil {
		return "", err
	}
	out := windows.UTF16ToString(buf[:n])
	out = strings.TrimPrefix(out, `\?\UNC\`) // network share: keep the form that follows
	if strings.HasPrefix(windows.UTF16ToString(buf[:n]), `\?\UNC\`) {
		return `\` + out, nil
	}
	return filepath.Clean(strings.TrimPrefix(out, `\?\`)), nil
}

// canonicalPath is realPath, or the cleaned input when the file cannot be opened.
func canonicalPath(p string) string {
	if r, err := realPath(p); err == nil {
		return r
	}
	return filepath.Clean(p)
}
