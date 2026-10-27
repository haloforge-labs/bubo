//go:build windows

package supervisor

import (
	"time"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// E14 item 6: the CLI reports its canonical exe path; the GUI may have reached
// it through a junction or with other letter case. Both are the same file.
func TestServerStartsWhenTheCLIIsReachedThroughAJunction(t *testing.T) {
	c := configFor(t, "normal")
	link := filepath.Join(t.TempDir(), "link")
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, filepath.Dir(c.Exe)).CombinedOutput(); err != nil {
		t.Skipf("no junction: %v %s", err, out)
	}
	c.Exe = filepath.Join(link, filepath.Base(c.Exe))
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { <-s.Exit() })
	reply(t, s.Start())
	v := waitFor(t, s, "ready or failed", 10*time.Second, func(v Snapshot) bool { return v.State == Ready || v.State == Failed })
	if v.State == Failed {
		t.Fatalf("failed: label %q err %+v", v.Label, v.LastError)
	}
	if v.LastError != nil {
		t.Fatalf("a junction in the CLI path broke the identity check: %+v", v.LastError)
	}
}

func TestServerStartsWhenTheCLIPathDiffersInLetterCase(t *testing.T) {
	c := configFor(t, "normal")
	c.Exe = filepath.Join(strings.ToUpper(filepath.Dir(c.Exe)), strings.ToUpper(filepath.Base(c.Exe)))
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { <-s.Exit() })
	reply(t, s.Start())
	waitState(t, s, Ready)
}

// E14 item 6: Stop while the server is still loading ends the owned job at
// once and cleanly; it is not a failure and does not wait for StopTimeout.
func TestStopBeforeTheFirstLifecycleRecordEndsTheJobCleanly(t *testing.T) {
	t.Setenv("BUBO_FAKE_START_DELAY_MS", "20000")
	s := supervisorFor(t, "normal")
	reply(t, s.Start())
	waitFor(t, s, "an owned process with no instance id yet", 6*time.Second, func(v Snapshot) bool {
		return v.State == Starting && v.Instance != nil && v.Instance.ID == ""
	})
	began := time.Now()
	reply(t, s.Stop())
	v := waitFor(t, s, "the end of the stop", 8*time.Second, func(v Snapshot) bool { return v.Instance == nil && (v.State == Stopped || v.State == Failed) })
	if v.State != Stopped || v.LastError != nil || v.Forced {
		t.Fatalf("stop during load must end cleanly: %s %+v forced=%v", v.State, v.LastError, v.Forced)
	}
	if took := time.Since(began); took > 2*time.Second {
		t.Fatalf("stop waited %s for the stop timeout", took)
	}
}
