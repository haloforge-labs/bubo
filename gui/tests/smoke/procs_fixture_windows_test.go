//go:build windows

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The 04:06 real run refused at step 1 because a lane's supervisor test had a
// fake bubo.exe running from %TEMP%\bubo-supervisor-tests-*. Only processes
// under the temp folder are ignored; anything else, or an unreadable path,
// still blocks the run.
func TestSplitTestFixtures(t *testing.T) {
	fake := filepath.Join(os.TempDir(), "bubo-supervisor-tests-1", "bubo.exe")
	list := []ProcInfo{
		{PID: 1, Name: "bubo.exe", Path: fake},
		{PID: 2, Name: "bubo.exe", Path: `C:\bubo\runtimes\bubo-0.1.0-3ff74dce\bin\bubo.exe`},
		{PID: 3, Name: "bubo.exe"},
		{PID: 4, Name: "bubo.exe", Path: strings.ToUpper(fake)},
		{PID: 5, Name: "bubo.exe", Path: strings.TrimSuffix(filepath.Clean(os.TempDir()), `\`) + `-other\bubo.exe`},
	}
	blocking, fixtures := splitTestFixtures(list)
	if len(fixtures) != 2 || fixtures[0].PID != 1 || fixtures[1].PID != 4 {
		t.Fatalf("fixtures=%+v", fixtures)
	}
	if len(blocking) != 3 || blocking[0].PID != 2 || blocking[1].PID != 3 || blocking[2].PID != 5 {
		t.Fatalf("blocking=%+v", blocking)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	got, err := imagePathOf(uint32(os.Getpid()))
	if err != nil || !strings.EqualFold(filepath.Clean(got), filepath.Clean(self)) {
		t.Fatalf("imagePathOf(self)=%q err=%v want %q", got, err, self)
	}
}
