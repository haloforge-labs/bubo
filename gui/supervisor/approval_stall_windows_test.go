//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// BLOCKER (E14 item 1): the engine download's idle watchdog must not run while
// the CLI waits for the user's approval. A person reads the licence for longer
// than the stall limit.
func TestApprovalWaitIsNotAStall(t *testing.T) {
	if testing.Short() {
		t.Skip("waits 60 s")
	}
	s := supervisorFor(t, "normal")
	reply(t, s.Prepare())
	deadline := time.Now().Add(3 * time.Second)
	for s.Snapshot().Approval == nil && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if s.Snapshot().Approval == nil {
		t.Fatal("no consent event")
	}
	time.Sleep(60 * time.Second)
	if v := s.Snapshot(); v.State != Preparing || v.LastError != nil || v.Approval == nil {
		t.Fatalf("the approval wait ended the job: state %s err %+v", v.State, v.LastError)
	}
	reply(t, s.Consent(true))
	v := waitState(t, s, Stopped)
	if v.LastError != nil || !v.Prepared {
		t.Fatalf("the download did not continue after a late approval: %+v", v.LastError)
	}
}

func shortStallFloor(t *testing.T) {
	t.Helper()
	old := downloadStallFloor
	downloadStallFloor = 400 * time.Millisecond
	t.Cleanup(func() { downloadStallFloor = old })
}

// E14 item 2: a stalled engine download gets a cancel line first; the process
// is forced only after StopTimeout.
func TestEngineStallSendsCancelBeforeForce(t *testing.T) {
	shortStallFloor(t)
	c := configFor(t, "normal")
	c.StallTimeout = 400 * time.Millisecond
	dir := os.Getenv("BUBO_FAKE_DIR")
	t.Setenv("BUBO_FAKE_ENGINE_STALL", "1")
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { <-s.Exit() })
	reply(t, s.Prepare())
	for i := 0; i < 300 && s.Snapshot().Approval == nil; i++ {
		time.Sleep(10 * time.Millisecond)
	}
	reply(t, s.Consent(true))
	v := waitFor(t, s, "the stall", 8*time.Second, func(v Snapshot) bool { return v.State == Failed })
	if v.LastError == nil || v.LastError.Code != "download_stalled" {
		t.Fatalf("stall: %+v", v.LastError)
	}
	if _, err := os.Stat(filepath.Join(dir, "cancel-received")); err != nil {
		t.Fatal("the CLI was forced without a cancel line first")
	}
	if v.Forced {
		t.Fatal("a CLI that answered the cancel was marked forced")
	}
}

func TestModelStallSendsCancelBeforeForce(t *testing.T) {
	shortStallFloor(t)
	r := newModelRig(t, "model-stall")
	r.setStall(400 * time.Millisecond)
	if _, err := r.svc.StartDownload(e21Flash, "download"); err != nil {
		t.Fatal(err)
	}
	if err := r.svc.ConfirmDownload(e21Flash); err != nil {
		t.Fatal(err)
	}
	j := r.waitJob(t, ended)
	if j.State != "failed" || j.Error == nil || j.Error.Code != "download_stalled" {
		t.Fatalf("job: %+v", j)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "cancel-received")); err != nil {
		t.Fatal("the CLI was forced without a cancel line first")
	}
}

func TestDownloadStallLimitIsAtLeastTheFloor(t *testing.T) {
	if got := stallLimit(25 * time.Second); got != 90*time.Second {
		t.Fatalf("25 s user limit gave %s, want the 90 s floor", got)
	}
	if got := stallLimit(120 * time.Second); got != 120*time.Second {
		t.Fatalf("120 s gave %s", got)
	}
	if got := stallLimit(-1); got != 0 {
		t.Fatal("a disabled watchdog stays disabled")
	}
}
