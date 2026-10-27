// This fixture never opens a window or loads an engine. The same tiny program
// stands in for the exiting original GUI, a server child, and the candidate GUI.
package main

import (
	"os"
	"path/filepath"
	"time"
)

func main() {
	if len(os.Args) == 3 && os.Args[1] == "--fixture-wait" {
		d, err := time.ParseDuration(os.Args[2])
		if err != nil {
			os.Exit(2)
		}
		time.Sleep(d)
		return
	}
	if len(os.Args) == 1 {
		return
	} // Old GUI restart; deliberately no filesystem writes.
	if len(os.Args) != 5 || os.Args[1] != "--bubo-update-health" || os.Args[3] != "--bubo-update-health-deadline" {
		os.Exit(2)
	}
	exe, err := os.Executable()
	if err != nil {
		os.Exit(3)
	}
	root := filepath.Dir(filepath.Dir(exe))
	mode, err := os.ReadFile(filepath.Join(root, "fixture-mode"))
	if err != nil {
		os.Exit(4)
	}
	switch string(mode) {
	case "good":
		// Independently spell the protocol instead of importing the implementation.
		name := filepath.Join(root, ".bubo-update-health")
		if err := os.WriteFile(name+".tmp", []byte("bubo-update-health-v1 "+os.Args[2]+"\n"), 0600); err != nil {
			os.Exit(5)
		}
		if err := os.Rename(name+".tmp", name); err != nil {
			os.Exit(6)
		}
		time.Sleep(200 * time.Millisecond)
	case "exit":
		return
	case "slow":
		time.Sleep(800 * time.Millisecond)
	case "timeout":
		deadline, err := time.Parse(time.RFC3339Nano, os.Args[4])
		if err != nil {
			os.Exit(7)
		}
		time.Sleep(max(time.Until(deadline), 0) + 50*time.Millisecond)
	case "stale":
		os.WriteFile(filepath.Join(root, ".bubo-update-health"), []byte("bubo-update-health-v1 wrong-token\n"), 0600)
		time.Sleep(100 * time.Millisecond)
	default:
		os.Exit(8)
	}
}
