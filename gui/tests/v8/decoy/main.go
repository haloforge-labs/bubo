//go:build windows

// decoy creates an invisible top-level window. Its child repeats the same
// action, proving both processes inherit the private desktop.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var user32 = syscall.NewLazyDLL("user32.dll")
var createWindow = user32.NewProc("CreateWindowExW")
var destroyWindow = user32.NewProc("DestroyWindow")
var isWindowVisible = user32.NewProc("IsWindowVisible")

func run(role, root string) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	class, _ := syscall.UTF16PtrFromString("STATIC")
	title, _ := syscall.UTF16PtrFromString("A0 invisible desktop decoy")
	// dwStyle=0: no WS_VISIBLE. Never call ShowWindow or SetWindowPos.
	hwnd, _, e := createWindow.Call(0, uintptr(unsafe.Pointer(class)), uintptr(unsafe.Pointer(title)), 0, 0, 0, 1, 1, 0, 0, 0, 0)
	if hwnd == 0 {
		return fmt.Errorf("CreateWindowExW: %w", e)
	}
	defer destroyWindow.Call(hwnd)
	visible, _, _ := isWindowVisible.Call(hwnd)
	if visible != 0 {
		return fmt.Errorf("decoy window unexpectedly visible")
	}
	self, err := windows.GetCurrentProcess()
	if err != nil {
		return err
	}
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(self, &creation, &exit, &kernel, &user); err != nil {
		return err
	}
	created := fmt.Sprintf("%08X%08X", creation.HighDateTime, creation.LowDateTime)
	ready := root + "." + role + ".json"
	data, _ := json.Marshal(struct {
		PID          int     `json:"pid"`
		CreationTime string  `json:"creation_time"`
		Handle       uintptr `json:"handle"`
		Visible      bool    `json:"visible"`
	}{os.Getpid(), created, hwnd, false})
	if err := os.WriteFile(ready, data, 0600); err != nil {
		return err
	}
	var child *exec.Cmd
	if role == "parent" {
		child = exec.Command(os.Args[0], "child", root)
		child.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
		if err := child.Start(); err != nil {
			return err
		}
		defer child.Wait()
	}
	stop := root + ".stop"
	for {
		if _, err := os.Stat(stop); err == nil {
			return nil
		} else if !os.IsNotExist(err) {
			return err
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func main() {
	if len(os.Args) != 3 || (os.Args[1] != "parent" && os.Args[1] != "child") {
		os.Exit(2)
	}
	if err := run(os.Args[1], os.Args[2]); err != nil {
		// This is a GUI-subsystem exe with redirected stderr; no dialog.
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
