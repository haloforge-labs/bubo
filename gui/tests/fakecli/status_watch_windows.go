//go:build windows

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/windows"
)

// The fake deliberately runs faster than the real CLI to exercise draining,
// View throttling and ownership. This is a test clock, not a CLI interval flag.
func statusWatch() {
	if dir == "" {
		cliError("invalid_arguments", "fakecli", "status", false, "Set BUBO_FAKE_DIR for status watch tests.", 2)
	}
	previous, _ := os.ReadFile(filepath.Join(dir, "watches.jsonl"))
	for _, line := range strings.Fields(string(previous)) {
		pid, _ := strconv.ParseUint(line, 10, 32)
		h, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(pid))
		if err != nil {
			continue
		}
		state, _ := windows.WaitForSingleObject(h, 0)
		windows.CloseHandle(h)
		if state == uint32(windows.WAIT_TIMEOUT) {
			_ = os.WriteFile(filepath.Join(dir, "watch-overlap"), []byte(line), 0600)
		}
	}
	f, err := os.OpenFile(filepath.Join(dir, "watches.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(f, "%d\n", os.Getpid())
	f.Close()
	closed := make(chan struct{})
	if mode != "watch-ignore-stdin" {
		go func() { _, _ = io.Copy(io.Discard, os.Stdin); close(closed) }()
	}
	markClosed := func() { _ = os.WriteFile(filepath.Join(dir, fmt.Sprintf("watch-%d-closed", os.Getpid())), nil, 0600) }
	ms, _ := strconv.Atoi(os.Getenv("BUBO_FAKE_WATCH_MS"))
	if ms <= 0 {
		ms = 40
	}
	tick := time.NewTicker(time.Duration(ms) * time.Millisecond)
	defer tick.Stop()
	event("capabilities", capabilities)
	if mode == "watch-last-error" {
		cliError("storage_failed", "status", "watch", true, "Final status error fixture.", 7)
	}
	for {
		select {
		case <-closed:
			markClosed()
			return
		default:
		}
		if _, err := os.Stat(filepath.Join(dir, "watch-exit")); err == nil {
			return
		}
		if _, err := os.Stat(filepath.Join(dir, "lose-status")); err == nil {
			os.Exit(1)
		}
		if _, err := os.Stat(filepath.Join(dir, "watch-stall")); err != nil {
			p := statusPayload()
			p["sample_time"] = now()
			if sampled := os.Getenv("BUBO_FAKE_WATCH_SAMPLE_TIME"); sampled != "" {
				p["sample_time"] = sampled
			}
			if mode == "watch-bad-json" {
				fmt.Println("invalid watch JSON")
			} else if mode == "watch-wrong-instance" {
				p["instances"] = []any{}
				event("status", p)
			} else if mode == "watch-bad-sequence" {
				seq = 0
				event("status", p)
			} else {
				event("status", p)
			}
		}
		select {
		case <-closed:
			markClosed()
			return
		case <-tick.C:
		}
	}
}
