//go:build windows

// holdbrowser makes one GUI's WebView2 browser process outlive the GUI by a fixed
// time, as a slow browser shutdown would on a loaded machine. Test-only.
//
// Usage: holdbrowser <gui pid> <hold ms> <log file>
//
// It finds the GUI's direct child msedgewebview2.exe that has no --type (the
// browser process), keeps a handle to it, waits for the GUI to exit, then
// suspends the browser at once. A suspended browser keeps its user-data folder
// lock and its debugging port. After the hold time it resumes the browser, which
// then shuts down as it would have without the hold.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var ntdll = syscall.NewLazyDLL("ntdll.dll")
var ntSuspend = ntdll.NewProc("NtSuspendProcess")
var ntResume = ntdll.NewProc("NtResumeProcess")
var logFile *os.File

func note(event string, fields map[string]any) {
	if fields == nil {
		fields = map[string]any{}
	}
	fields["t"] = time.Now().Format("15:04:05.000")
	fields["ev"] = event
	b, _ := json.Marshal(fields)
	logFile.Write(append(b, '\n'))
}

func commandLine(h windows.Handle) string {
	buf := make([]byte, 8192)
	var n uint32
	if err := windows.NtQueryInformationProcess(h, 60, unsafe.Pointer(&buf[0]), uint32(len(buf)), &n); err != nil {
		return ""
	}
	return (*windows.NTUnicodeString)(unsafe.Pointer(&buf[0])).String()
}

func findBrowser(gui uint32) (windows.Handle, uint32) {
	s, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return 0, 0
	}
	defer windows.CloseHandle(s)
	var e windows.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	for err = windows.Process32First(s, &e); err == nil; err = windows.Process32Next(s, &e) {
		if e.ParentProcessID != gui || !strings.EqualFold(windows.UTF16ToString(e.ExeFile[:]), "msedgewebview2.exe") {
			continue
		}
		h, err := windows.OpenProcess(windows.PROCESS_SUSPEND_RESUME|windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.SYNCHRONIZE, false, e.ProcessID)
		if err != nil {
			continue
		}
		if strings.Contains(commandLine(h), "--type=") {
			windows.CloseHandle(h)
			continue
		}
		return h, e.ProcessID
	}
	return 0, 0
}

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: holdbrowser <gui pid> <hold ms> <log file>")
		os.Exit(2)
	}
	gui64, _ := strconv.ParseUint(os.Args[1], 10, 32)
	holdMs, _ := strconv.Atoi(os.Args[2])
	var err error
	if logFile, err = os.OpenFile(os.Args[3], os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err != nil {
		os.Exit(2)
	}
	guiPID := uint32(gui64)
	guiHandle, err := windows.OpenProcess(windows.SYNCHRONIZE, false, guiPID)
	if err != nil {
		note("error", map[string]any{"msg": "cannot open GUI: " + err.Error()})
		os.Exit(1)
	}
	var browser windows.Handle
	var browserPID uint32
	for deadline := time.Now().Add(30 * time.Second); browser == 0 && time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		browser, browserPID = findBrowser(guiPID)
	}
	if browser == 0 {
		note("error", map[string]any{"msg": "no browser process under the GUI"})
		os.Exit(1)
	}
	note("armed", map[string]any{"gui": guiPID, "browser": browserPID, "hold_ms": holdMs})
	// A poll shorter than the browser's own exit delay (about 45 ms when idle).
	for {
		ev, _ := windows.WaitForSingleObject(guiHandle, 2)
		if ev == windows.WAIT_OBJECT_0 {
			break
		}
	}
	ntSuspend.Call(uintptr(browser))
	if ev, _ := windows.WaitForSingleObject(browser, 0); ev == windows.WAIT_OBJECT_0 {
		note("late", map[string]any{"msg": "browser had already exited when the GUI exit was seen"})
		return
	}
	note("suspended", map[string]any{"browser": browserPID})
	time.Sleep(time.Duration(holdMs) * time.Millisecond)
	ntResume.Call(uintptr(browser))
	note("resumed", map[string]any{"browser": browserPID})
	windows.WaitForSingleObject(browser, 20000)
	note("browser_exited", nil)
}
