//go:build windows

// This inert updater candidate simulates a first status answer at 40 seconds.
// It opens no window, network socket, engine, or user-data path. The v8 gate
// launches it only through the private-desktop Job Object controller.
package main

import (
	"os"
	"path/filepath"
	"time"
)

func main() {
	if len(os.Args) != 5 || os.Args[1] != "--bubo-update-health" || os.Args[3] != "--bubo-update-health-deadline" {
		os.Exit(2)
	}
	time.Sleep(40 * time.Second)
	exe, err := os.Executable()
	if err != nil {
		os.Exit(3)
	}
	marker := filepath.Join(filepath.Dir(filepath.Dir(exe)), ".bubo-update-health")
	if err := os.WriteFile(marker+".tmp", []byte("bubo-update-health-v1 "+os.Args[2]+"\n"), 0600); err != nil {
		os.Exit(4)
	}
	if err := os.Rename(marker+".tmp", marker); err != nil {
		os.Exit(5)
	}
	time.Sleep(200 * time.Millisecond)
}
