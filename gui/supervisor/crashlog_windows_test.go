//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// E14 item 7: a crash leaves its cause and stack in the app log folder.
func TestReportPanicWritesTheCauseAndStack(t *testing.T) {
	dir := t.TempDir()
	setCrashLog(filepath.Join(dir, "logs"))
	t.Cleanup(func() { setCrashLog("") })
	func() {
		defer func() { _ = recover() }() // no handler installed: ReportPanic re-raises
		defer ReportPanic()
		panic("boom-cause-42")
	}()
	b, err := os.ReadFile(filepath.Join(dir, "logs", "gui-crash.log"))
	if err != nil {
		t.Fatal("no crash log:", err)
	}
	text := string(b)
	if !strings.Contains(text, "boom-cause-42") || !strings.Contains(text, "TestReportPanicWritesTheCauseAndStack") {
		t.Fatalf("cause or stack missing:\n%s", text)
	}
}
