//go:build windows

package supervisor

import (
	"context"
	"errors"
	"testing"
	"time"
)

// E14 (first launch): Windows can scan a fresh unsigned exe on its first run, so
// the first CLI call can take 15 s. Calls before the first answer wait longer
// than the ordinary query limit, and a first call that waits more than a few
// seconds shows one plain line.

func firstCallSupervisor(t *testing.T, delays string) *Supervisor {
	t.Helper()
	cfg := configFor(t, "normal")
	cfg.QueryTimeout = 150 * time.Millisecond
	cfg.ColdStartTimeout = 5 * time.Second
	cfg.FirstCallNotice = 300 * time.Millisecond
	t.Setenv("BUBO_FAKE_STATUS_DELAYS", delays)
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-s.Exit():
		case <-time.After(6 * time.Second):
			t.Error("supervisor did not close")
		}
	})
	return s
}

func TestSlowFirstCallShowsOneLineThenContinuesCleanly(t *testing.T) {
	s := firstCallSupervisor(t, "1200ms")
	waitFor(t, s, "the first-call line", 3*time.Second, func(v Snapshot) bool { return v.StartupWaiting })
	v := new(ViewBuilder).Build(ViewInput{Snapshot: s.Snapshot()})
	if v.Initialized || v.StatusLabel != "Starting for the first time…" || v.Failure != nil {
		t.Fatalf("waiting view: initialized=%v label=%q failure=%+v", v.Initialized, v.StatusLabel, v.Failure)
	}
	ready := waitFor(t, s, "discovery", 6*time.Second, func(v Snapshot) bool { return v.Initialized })
	if ready.StartupWaiting || ready.State == Failed || ready.LastError != nil {
		t.Fatalf("the first call did not continue cleanly: %+v", ready.LastError)
	}
	if label := new(ViewBuilder).Build(ViewInput{Snapshot: ready}).StatusLabel; label == "Starting for the first time…" {
		t.Fatal("the line stayed after the first answer")
	}
}

func TestFastFirstCallShowsNoLine(t *testing.T) {
	s := firstCallSupervisor(t, "0s")
	ready := waitFor(t, s, "discovery", 3*time.Second, func(v Snapshot) bool { return v.Initialized })
	time.Sleep(400 * time.Millisecond)
	if ready.StartupWaiting || s.Snapshot().StartupWaiting {
		t.Fatal("a first call under the notice time must show no line")
	}
}

// Any CLI call made before the first answer gets the cold-start limit, not the
// ordinary query limit.
func TestCallsBeforeTheFirstAnswerUseTheColdLimit(t *testing.T) {
	s := firstCallSupervisor(t, "1s,1s")
	if _, err := s.query(context.Background(), nil); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			t.Fatal("a call before the first answer used the ordinary limit")
		}
		t.Fatal(err)
	}
	waitFor(t, s, "discovery", 6*time.Second, func(v Snapshot) bool { return v.Initialized })
}
