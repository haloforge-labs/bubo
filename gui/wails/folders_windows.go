//go:build windows

package main

import (
	"context"
	"os/exec"
	"path/filepath"
	"syscall"

	"golang.org/x/sys/windows"
)

type nativeFolderOpener struct{}

// Folder policy lives in AppService. Invoke Explorer directly, with a single
// path argument and no shell or console window; no command text is interpreted.
func (nativeFolderOpener) OpenFolder(_ context.Context, path string) error {
	cmd, err := nativeFolderCommand(path)
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() { _ = cmd.Wait() }()
	return nil
}

// Command construction is testable without launching any native window.
func nativeFolderCommand(path string) (*exec.Cmd, error) {
	dir, err := windows.GetWindowsDirectory()
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(filepath.Join(dir, "explorer.exe"), path)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NO_WINDOW}
	return cmd, nil
}
