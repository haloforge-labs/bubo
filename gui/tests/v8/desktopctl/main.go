//go:build windows

// desktopctl creates a private desktop in WinSta0 and launches one test process
// there. It never switches the input desktop or creates a window of its own.
package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

type config struct {
	Exe                 string   `json:"exe"`
	CWD                 string   `json:"cwd"`
	Args                []string `json:"args"`
	StateFile           string   `json:"state_file"`
	BaselineFile        string   `json:"baseline_file"`
	ViolationFile       string   `json:"violation_file"`
	ProbeDiagnostics    bool     `json:"probe_diagnostics"`
	OutsideJobDecoyExe  string   `json:"outside_job_decoy_exe"`
	OutsideJobDecoyRoot string   `json:"outside_job_decoy_root"`
	StopFile            string   `json:"stop_file"`
	ExitFile            string   `json:"exit_file"`
	// StderrFile, when set, receives the GUI process's stderr (and stdout). A
	// windowsgui exe has no console, so a startup panic is otherwise lost.
	StderrFile string `json:"stderr_file"`
}

type processExit struct {
	PID          uint32 `json:"pid"`
	CreationTime string `json:"creation_time"`
	Image        string `json:"image"`
	Code         uint32 `json:"code"`
	CodeHex      string `json:"code_hex"`
}

type window struct {
	Handle          uintptr `json:"handle"`
	PID             uint32  `json:"pid"`
	Class           string  `json:"class"`
	Visible         *bool   `json:"visible,omitempty"`
	JobCreationTime string  `json:"job_creation_time,omitempty"`
}

type snapshot struct {
	Time         string       `json:"time"`
	Default      []window     `json:"default"`
	Hidden       []window     `json:"hidden"`
	JobProcesses []jobProcess `json:"job_processes,omitempty"`
}

type violation struct {
	Time   string `json:"time"`
	Kind   string `json:"kind"`
	Window window `json:"window"`
}

var user32 = syscall.NewLazyDLL("user32.dll")
var kernel32 = syscall.NewLazyDLL("kernel32.dll")
var getCurrentThreadID = kernel32.NewProc("GetCurrentThreadId")
var createDesktop = user32.NewProc("CreateDesktopW")
var closeDesktop = user32.NewProc("CloseDesktop")
var getThreadDesktop = user32.NewProc("GetThreadDesktop")
var enumDesktopWindows = user32.NewProc("EnumDesktopWindows")
var getWindowThreadProcessID = user32.NewProc("GetWindowThreadProcessId")
var getClassName = user32.NewProc("GetClassNameW")
var isWindowVisible = user32.NewProc("IsWindowVisible")
var postMessage = user32.NewProc("PostMessageW")
var getProcessWindowStation = user32.NewProc("GetProcessWindowStation")
var getUserObjectInformation = user32.NewProc("GetUserObjectInformationW")

func winStationName() (string, error) {
	h, _, e := getProcessWindowStation.Call()
	if h == 0 {
		return "", e
	}
	var required uint32
	// UOI_NAME = 2. Query the name only; never change the window station.
	getUserObjectInformation.Call(h, 2, 0, 0, uintptr(unsafe.Pointer(&required)))
	if required == 0 {
		return "", fmt.Errorf("GetUserObjectInformationW returned no name size")
	}
	buf := make([]uint16, (required+1)/2)
	ok, _, e := getUserObjectInformation.Call(h, 2, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)*2), uintptr(unsafe.Pointer(&required)))
	if ok == 0 {
		return "", e
	}
	return syscall.UTF16ToString(buf), nil
}

type windowEnumeration struct {
	members  map[uint32]jobProcess
	hidden   bool
	baseline bool
	out      []window
}

// NewCallback has a process lifetime allocation on Windows. Reusing this one
// callback lets long hidden-desktop walks keep taking 75 ms snapshots.
var currentWindowEnumeration *windowEnumeration
var windowCallback = syscall.NewCallback(func(hwnd, _ uintptr) uintptr {
	state := currentWindowEnumeration
	if state == nil {
		return 0
	}
	var pid uint32
	getWindowThreadProcessID.Call(hwnd, uintptr(unsafe.Pointer(&pid)))
	// The window may be destroyed after EnumDesktopWindows yields its handle.
	if pid == 0 {
		return 1
	}
	item := window{Handle: hwnd, PID: pid}
	if member, ok := state.members[pid]; ok {
		// A reused PID must also match the job member's creation time.
		handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
		if err == nil {
			created, timeErr := processCreationTime(handle)
			windows.CloseHandle(handle)
			if timeErr == nil && created == member.CreationTime {
				item.JobCreationTime = created
			}
		}
	}
	if !state.hidden && !state.baseline && item.JobCreationTime == "" {
		return 1
	}
	class := make([]uint16, 256)
	getClassName.Call(hwnd, uintptr(unsafe.Pointer(&class[0])), uintptr(len(class)))
	item.Class = syscall.UTF16ToString(class)
	if state.hidden || item.JobCreationTime != "" {
		visible, _, _ := isWindowVisible.Call(hwnd)
		isVisible := visible != 0
		item.Visible = &isVisible
	}
	state.out = append(state.out, item)
	return 1
})

func windowsOn(desktop uintptr, members map[uint32]jobProcess, hidden, baseline bool) ([]window, error) {
	if currentWindowEnumeration != nil {
		return nil, fmt.Errorf("nested desktop enumeration")
	}
	state := &windowEnumeration{members: members, hidden: hidden, baseline: baseline}
	currentWindowEnumeration = state
	defer func() { currentWindowEnumeration = nil }()
	ok, _, e := enumDesktopWindows.Call(desktop, windowCallback, 0)
	if ok == 0 {
		// An empty desktop returns zero without setting LastError. It is not
		// yet evidence of a failed enumeration.
		if e == syscall.Errno(0) {
			return state.out, nil
		}
		return nil, e
	}
	return state.out, nil
}

func takeSnapshot(defaultDesktop, hiddenDesktop uintptr, identities, members map[uint32]jobProcess, baseline bool) (snapshot, error) {
	def, err := windowsOn(defaultDesktop, identities, false, baseline)
	if err != nil {
		return snapshot{}, fmt.Errorf("default desktop: %w", err)
	}
	var hidden []window
	if hiddenDesktop != 0 {
		hidden, err = windowsOn(hiddenDesktop, identities, true, false)
		if err != nil {
			return snapshot{}, fmt.Errorf("hidden desktop: %w", err)
		}
	}
	result := snapshot{Time: time.Now().UTC().Format(time.RFC3339Nano), Default: def, Hidden: hidden}
	for _, member := range members {
		result.JobProcesses = append(result.JobProcesses, member)
	}
	sort.Slice(result.JobProcesses, func(i, j int) bool { return result.JobProcesses[i].PID < result.JobProcesses[j].PID })
	return result, nil
}

func knownJobMembers(current map[uint32]jobProcess, tracked map[string]trackedProcess) map[uint32]jobProcess {
	known := make(map[uint32]jobProcess, len(tracked))
	for _, member := range tracked {
		known[member.PID] = member.jobProcess
	}
	for pid, member := range current {
		known[pid] = member
	}
	return known
}

func writeJSONLine(file *os.File, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = file.Write(append(data, '\n'))
	return err
}

func launch(c config, desktop string) (syscall.ProcessInformation, error) {
	exe, err := syscall.UTF16PtrFromString(c.Exe)
	if err != nil {
		return syscall.ProcessInformation{}, err
	}
	cmd := []string{syscall.EscapeArg(c.Exe)}
	for _, arg := range c.Args {
		cmd = append(cmd, syscall.EscapeArg(arg))
	}
	cmdline, err := syscall.UTF16PtrFromString(strings.Join(cmd, " "))
	if err != nil {
		return syscall.ProcessInformation{}, err
	}
	cwd, err := syscall.UTF16PtrFromString(c.CWD)
	if err != nil {
		return syscall.ProcessInformation{}, err
	}
	name, err := syscall.UTF16PtrFromString("WinSta0\\" + desktop)
	if err != nil {
		return syscall.ProcessInformation{}, err
	}
	startup := syscall.StartupInfo{Cb: uint32(unsafe.Sizeof(syscall.StartupInfo{})), Desktop: name, Flags: syscall.STARTF_USESHOWWINDOW, ShowWindow: syscall.SW_HIDE}
	var process syscall.ProcessInformation
	inherit := false
	if c.StderrFile != "" {
		path, err := syscall.UTF16PtrFromString(c.StderrFile)
		if err != nil {
			return process, err
		}
		sa := &syscall.SecurityAttributes{Length: uint32(unsafe.Sizeof(syscall.SecurityAttributes{})), InheritHandle: 1}
		h, err := syscall.CreateFile(path, syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE, sa, syscall.CREATE_ALWAYS, syscall.FILE_ATTRIBUTE_NORMAL, 0)
		if err != nil {
			return process, err
		}
		defer syscall.CloseHandle(h)
		startup.Flags |= syscall.STARTF_USESTDHANDLES
		startup.StdOutput, startup.StdErr = h, h
		// A zero stdin handle is accepted with USESTDHANDLES.
		inherit = true
	}
	// BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW | CREATE_SUSPENDED.
	// The suspended process must join the job before any of its code runs.
	err = syscall.CreateProcess(exe, cmdline, nil, nil, inherit, 0x00004000|0x08000000|windows.CREATE_SUSPENDED, nil, cwd, &startup, &process)
	return process, err
}

func run(c config) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	station, err := winStationName()
	if err != nil || station != "WinSta0" {
		return fmt.Errorf("interactive window station required: name=%q err=%v", station, err)
	}
	threadID, _, _ := getCurrentThreadID.Call()
	defaultDesktop, _, e := getThreadDesktop.Call(uintptr(threadID))
	if defaultDesktop == 0 {
		return fmt.Errorf("GetThreadDesktop: %w", e)
	}
	output, err := os.Create(c.ViolationFile)
	if err != nil {
		return err
	}
	defer output.Close()
	before, err := takeSnapshot(defaultDesktop, 0, nil, nil, true)
	if err != nil {
		return err
	}
	baselineData, err := json.Marshal(before)
	if err != nil {
		return err
	}
	if err := os.WriteFile(c.BaselineFile, baselineData, 0600); err != nil {
		return err
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		return err
	}
	name := fmt.Sprintf("BuboA0_%d_%s", os.Getpid(), hex.EncodeToString(random))
	namePtr, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		return err
	}
	// GENERIC_ALL. This opens only the newly created desktop; there is
	// deliberately no SwitchDesktop or SetThreadDesktop call.
	hiddenDesktop, _, e := createDesktop.Call(uintptr(unsafe.Pointer(namePtr)), 0, 0, 0, 0x10000000, 0)
	if hiddenDesktop == 0 {
		return fmt.Errorf("CreateDesktopW: %w", e)
	}
	defer closeDesktop.Call(hiddenDesktop)
	job, err := createTestJob()
	if err != nil {
		return err
	}
	// Closing the job handle kills all its remaining members. It closes before
	// the hidden desktop handle, so native child dialogs remain contained.
	defer windows.CloseHandle(job)
	process, err := launch(c, name)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(process.Thread)
	if err := windows.AssignProcessToJobObject(job, windows.Handle(process.Process)); err != nil {
		// This process is still suspended. There is no safe non-job process to
		// terminate under the job-only cleanup rule, so report the exact PID.
		return fmt.Errorf("AssignProcessToJobObject(suspended PID %d): %w", process.ProcessId, err)
	}
	created, err := processCreationTime(windows.Handle(process.Process))
	if err != nil {
		return fmt.Errorf("GetProcessTimes(root PID %d): %w", process.ProcessId, err)
	}
	tracked := map[string]trackedProcess{
		fmt.Sprintf("%d/%s", process.ProcessId, created): {
			jobProcess: jobProcess{PID: process.ProcessId, CreationTime: created, Image: processImage(windows.Handle(process.Process))},
			handle:     windows.Handle(process.Process),
		},
	}
	defer closeTracked(tracked)
	initial, err := jobPIDs(job)
	if err != nil {
		return err
	}
	foundRoot := false
	for _, pid := range initial {
		if pid == process.ProcessId {
			foundRoot = true
		}
	}
	if !foundRoot {
		return fmt.Errorf("suspended root PID %d absent from its job", process.ProcessId)
	}
	if _, err := windows.ResumeThread(windows.Handle(process.Thread)); err != nil {
		return fmt.Errorf("ResumeThread(root PID %d): %w", process.ProcessId, err)
	}
	state, _ := json.Marshal(struct {
		PID          uint32 `json:"pid"`
		CreationTime string `json:"creation_time"`
		Desktop      string `json:"desktop"`
	}{process.ProcessId, created, name})
	if err := os.WriteFile(c.StateFile, state, 0600); err != nil {
		return err
	}
	if c.OutsideJobDecoyExe != "" {
		if c.OutsideJobDecoyRoot == "" {
			return fmt.Errorf("outside-job decoy root is required")
		}
		// This negative control is itself attached to a separate Job Object
		// before resuming, on the same private desktop as the primary job.
		decoyJob, err := createTestJob()
		if err != nil {
			return err
		}
		defer windows.CloseHandle(decoyJob)
		decoyConfig := config{Exe: c.OutsideJobDecoyExe, CWD: c.CWD, Args: []string{"parent", c.OutsideJobDecoyRoot}}
		decoy, err := launch(decoyConfig, name)
		if err != nil {
			return err
		}
		defer syscall.CloseHandle(decoy.Thread)
		defer syscall.CloseHandle(decoy.Process)
		if err := windows.AssignProcessToJobObject(decoyJob, windows.Handle(decoy.Process)); err != nil {
			return fmt.Errorf("AssignProcessToJobObject(suspended decoy PID %d): %w", decoy.ProcessId, err)
		}
		if _, err := windows.ResumeThread(windows.Handle(decoy.Thread)); err != nil {
			return fmt.Errorf("ResumeThread(decoy PID %d): %w", decoy.ProcessId, err)
		}
	}
	stop := make(chan struct{})
	closeRequest := make(chan struct{}, 1)
	go func() {
		reader := bufio.NewScanner(os.Stdin)
		for reader.Scan() {
			switch reader.Text() {
			case "close":
				select {
				case closeRequest <- struct{}{}:
				default:
				}
			case "stop":
				close(stop)
				return
			}
		}
		close(stop)
	}()
	ticker := time.NewTicker(75 * time.Millisecond)
	defer ticker.Stop()
	stopping := false
	lastDiagnostic := ""
	var stopDeadline time.Time
	stopSignal := (<-chan struct{})(stop)
	for {
		members, err := refreshJobMembers(job, tracked)
		if err != nil {
			return err
		}
		codes, err := jobExits(tracked)
		if err != nil {
			return err
		}
		if c.ExitFile != "" {
			data, _ := json.Marshal(codes)
			if err := os.WriteFile(c.ExitFile, data, 0600); err != nil {
				return err
			}
		}
		known := knownJobMembers(members, tracked)
		current, err := takeSnapshot(defaultDesktop, hiddenDesktop, known, members, false)
		if err != nil {
			return err
		}
		for _, item := range current.Default {
			if item.JobCreationTime != "" {
				if err := writeJSONLine(output, violation{current.Time, "default_job_window", item}); err != nil {
					return err
				}
				_ = windows.TerminateJobObject(job, 1)
				return fmt.Errorf("job window on owner's default desktop: handle=%d PID=%d class=%s", item.Handle, item.PID, item.Class)
			}
		}
		for _, item := range current.Hidden {
			if item.JobCreationTime == "" {
				// A job member can exit between the job PID query and window
				// enumeration. Confirm the same HWND/PID after that teardown
				// interval before treating it as an outside process.
				time.Sleep(10 * time.Millisecond)
				latestMembers, err := refreshJobMembers(job, tracked)
				if err != nil {
					return err
				}
				latest, err := windowsOn(hiddenDesktop, knownJobMembers(latestMembers, tracked), true, false)
				if err != nil {
					return err
				}
				confirmed := false
				for _, again := range latest {
					if again.Handle == item.Handle && again.PID == item.PID && again.JobCreationTime == "" {
						confirmed = true
						break
					}
				}
				if !confirmed {
					continue
				}
				if err := writeJSONLine(output, violation{current.Time, "hidden_outside_job", item}); err != nil {
					return err
				}
				_ = windows.TerminateJobObject(job, 1)
				return fmt.Errorf("window outside job on hidden desktop: handle=%d PID=%d class=%s", item.Handle, item.PID, item.Class)
			}
		}
		// Only the isolation probe asks for live diagnostics. This stream
		// contains job members and private-desktop windows, never owner windows.
		if c.ProbeDiagnostics {
			diagnostic, err := json.Marshal(struct {
				Hidden       []window     `json:"hidden"`
				JobProcesses []jobProcess `json:"job_processes"`
			}{current.Hidden, current.JobProcesses})
			if err != nil {
				return err
			}
			if string(diagnostic) != lastDiagnostic {
				fmt.Fprintln(os.Stdout, "PROBE_WINDOWS "+string(diagnostic))
				lastDiagnostic = string(diagnostic)
			}
		}
		if stopping && len(members) == 0 {
			return nil
		}
		if stopping && time.Now().After(stopDeadline) {
			_ = windows.TerminateJobObject(job, 1)
			return fmt.Errorf("job did not empty after stop deadline")
		}
		select {
		case <-closeRequest:
			// WM_CLOSE is sent only to this launch's Wails root HWND on the
			// private desktop. It never targets a default-desktop window.
			var target uintptr
			for _, item := range current.Hidden {
				if item.Class == "wailsWindow" && item.PID == process.ProcessId && item.JobCreationTime == created {
					if target != 0 {
						return fmt.Errorf("multiple owned Wails root windows")
					}
					target = item.Handle
				}
			}
			if target == 0 {
				return fmt.Errorf("owned Wails root window not found on hidden desktop")
			}
			if ok, _, e := postMessage.Call(target, 0x0010, 0, 0); ok == 0 {
				return fmt.Errorf("PostMessageW(WM_CLOSE): %w", e)
			}
		case <-stopSignal:
			stopSignal = nil
			if !stopping {
				stopping = true
				stopDeadline = time.Now().Add(5 * time.Second)
				if c.StopFile != "" {
					if err := os.WriteFile(c.StopFile, []byte("stop"), 0600); err != nil {
						return err
					}
				} else {
					if err := windows.TerminateJobObject(job, 1); err != nil {
						return err
					}
				}
			}
		case <-ticker.C:
		}
	}
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: desktopctl <launch-config.json>")
		os.Exit(2)
	}
	input, err := os.ReadFile(os.Args[1])
	var c config
	if err == nil {
		err = json.Unmarshal(input, &c)
	}
	if err == nil {
		err = run(c)
	}
	if err != nil {
		if c.StateFile != "" {
			_ = os.WriteFile(c.StateFile+".error", []byte(err.Error()), 0600)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if c.StateFile != "" {
		if err := os.WriteFile(c.StateFile+".done", []byte("job empty and desktop closed"), 0600); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
