//go:build windows

package supervisor

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func backendFor(t *testing.T, mode string) *NativeBackend {
	t.Helper()
	c := configFor(t, mode)
	c.SettingsPath = filepath.Join(filepath.Dir(c.LogDir), "settings.json")
	settings := DefaultSettings()
	settings.Port = c.Port
	b, err := NewBackend(c, settings, c.Exe)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, b.Action("exit", b.settings, "")) })
	waitBackend(t, b, func(v Snapshot) bool { return v.Initialized })
	return b
}

func waitBackend(t *testing.T, b *NativeBackend, pred func(Snapshot) bool) Snapshot {
	t.Helper()
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		v := b.Snapshot()
		if pred(v) {
			return v
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("backend timeout: %+v, notice: %s", b.Snapshot(), b.Notice())
	return Snapshot{}
}

func TestNativeBackendFakeCLIWorkflow(t *testing.T) {
	b := backendFor(t, "ui")
	s := b.settings
	reply(t, b.Action("prepare", s, ""))
	v := waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
	if len(v.Approval.Components) != 1 || v.Approval.Components[0] != "runtime" || len(v.PlanParts) != 5 {
		t.Fatal("first consent includes more than the runtime", v.Approval, len(v.PlanParts))
	}
	reply(t, b.Action("approve", s, ""))
	waitBackend(t, b, func(v Snapshot) bool {
		return v.State == Stopped && v.Status != nil && v.Status.Readiness.Runtime && !v.Status.Readiness.Model
	})
	reply(t, b.s.Prepare("model"))
	v = waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
	if len(v.Approval.Components) != 1 || v.Approval.Components[0] != "model" {
		t.Fatal("model consent was not separate", v.Approval)
	}
	reply(t, b.Action("approve", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped && v.Status != nil && v.Status.Readiness.Model })
	reply(t, b.Action("start", s, ""))
	first := waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready }).Instance.ID
	reply(t, b.Action("diagnose", s, ""))
	if !strings.Contains(b.Diagnostic(), "PASS") {
		t.Fatal("CLI diagnostic missing")
	}
	reply(t, b.Action("preview", s, ""))
	path := filepath.Join(t.TempDir(), "support.zip")
	reply(t, b.Action("bundle", s, path))
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	reply(t, b.Action("restart", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready && v.Instance.ID != first })
	reply(t, b.Action("stop", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped && v.Instance == nil })
	reply(t, b.Action("exit", s, ""))
	reply(t, b.Action("exit", s, ""))
}

func TestNativeBackendApprovalWithoutOptionalQuestion(t *testing.T) {
	b := backendFor(t, "split")
	var coded *AppError
	if r := <-b.Action("preview", b.settings, ""); !errors.As(r.Err, &coded) || coded.Code != "diagnostic_failed" || b.BundlePreview() != "" {
		t.Fatal("missing bundle fields accepted", r.Err)
	}
	reply(t, b.Action("prepare", b.settings, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.Approval != nil })
	reply(t, b.Action("approve", b.settings, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped && v.Instance == nil })
}
