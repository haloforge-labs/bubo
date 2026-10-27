//go:build windows

package supervisor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func coldSupervisor(t *testing.T, delays string, cold, backoff time.Duration, query ...time.Duration) *Supervisor {
	t.Helper()
	cfg := configFor(t, "normal")
	cfg.QueryTimeout = 150 * time.Millisecond
	if len(query) != 0 {
		cfg.QueryTimeout = query[0]
	}
	cfg.ColdStartTimeout, cfg.ColdStartRetryBackoff = cold, backoff
	t.Setenv("BUBO_FAKE_STATUS_DELAYS", delays)
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-s.Exit():
		case <-time.After(4 * time.Second):
			t.Error("supervisor did not close")
		}
	})
	return s
}

func waitStartup(t *testing.T, s *Supervisor, pred func(Snapshot) bool) Snapshot {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		v := s.Snapshot()
		if pred(v) {
			return v
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("startup condition not met: %+v", s.Snapshot())
	return Snapshot{}
}

func statusAttempts(t *testing.T) int {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "status-oneshots.txt"))
	if os.IsNotExist(err) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return len(strings.Fields(string(body)))
}

func TestInitialDiscoverHasColdBudgetAndTypedPhase(t *testing.T) {
	s := coldSupervisor(t, "500ms", 2*time.Second, 20*time.Millisecond)
	waitStartup(t, s, func(Snapshot) bool { return statusAttempts(t) == 1 })
	time.Sleep(200 * time.Millisecond) // exceeds the ordinary 150 ms budget
	v := new(ViewBuilder).Build(ViewInput{Snapshot: s.Snapshot()})
	if v.Initialized || v.StartupPhase != StartingCLI || v.State == Failed || v.Failure != nil {
		t.Fatal("slow cold start was reported as failure", v)
	}
	ready := waitStartup(t, s, func(v Snapshot) bool { return v.Initialized })
	if ready.StartupPhase != "" || ready.State == Failed || ready.LastError != nil || statusAttempts(t) != 1 {
		t.Fatal("cold startup did not complete once", ready)
	}
}

func TestInitialDiscoverRetriesTimeoutWithBackoff(t *testing.T) {
	s := coldSupervisor(t, "3s,0s", 400*time.Millisecond, 250*time.Millisecond)
	waitStartup(t, s, func(Snapshot) bool { return statusAttempts(t) == 1 })
	time.Sleep(450 * time.Millisecond)
	v := s.Snapshot()
	if v.Initialized || v.State == Failed || v.LastError != nil || v.StartupPhase != "startup_slow" || statusAttempts(t) != 1 {
		t.Fatal("timeout escaped while retry was pending", v)
	}
	v = waitStartup(t, s, func(v Snapshot) bool { return v.Initialized })
	if v.State == Failed || v.LastError != nil || statusAttempts(t) != 2 {
		t.Fatal("warm retry did not recover", v)
	}
}

func TestInitialDiscoverProtocolFailureIsNotRetried(t *testing.T) {
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	s := supervisorFor(t, "major")
	v := waitStartup(t, s, func(v Snapshot) bool { return v.Initialized })
	if v.State != Failed || v.LastError == nil || v.LastError.Code != "protocol_mismatch" || statusAttempts(t) != 1 {
		t.Fatal("protocol failure was retried or hidden", v.LastError, statusAttempts(t))
	}
}

func TestLaterDiscoverTimeoutIsTypedAndDoesNotRetry(t *testing.T) {
	s := coldSupervisor(t, "0s,3s", 2*time.Second, 20*time.Millisecond, 800*time.Millisecond)
	waitStartup(t, s, func(v Snapshot) bool { return v.Initialized })
	select {
	case r := <-s.Discover():
		if !errors.Is(r.Err, context.DeadlineExceeded) {
			t.Fatal("later timeout was hidden", r.Err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("later discovery did not respect its ordinary budget")
	}
	v := waitStartup(t, s, func(v Snapshot) bool { return v.State == Failed })
	if v.LastError == nil || v.LastError.Code != "query_timeout" || v.StartupPhase != "" || statusAttempts(t) != 2 {
		t.Fatal("later timeout retried or lost its typed error", v.LastError, statusAttempts(t))
	}
}

func TestInitialDiscoverCancellationInterruptsBackoff(t *testing.T) {
	cfg := configFor(t, "normal")
	cfg.ColdStartTimeout, cfg.ColdStartRetryBackoff = 300*time.Millisecond, 3*time.Second
	t.Setenv("BUBO_FAKE_STATUS_DELAYS", "3s,3s")
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	logs, err := NewLogs(cfg.LogDir)
	if err != nil {
		t.Fatal(err)
	}
	defer logs.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := &Supervisor{config: cfg, ctx: ctx, logs: logs, keys: NewKeyManager(nil)}
	done := make(chan error, 1)
	backoff := make(chan struct{})
	started := time.Now()
	go func() {
		_, err := s.initialDiscoverWithSlow(func() { close(backoff) })
		done <- err
	}()
	// The counter is written before the helper sleeps, not when the supervisor
	// finishes killing/draining it. Observe backoff itself within the original
	// two-second counter wait plus 400 ms sleep budget before cancelling.
	select {
	case <-backoff:
	case <-time.After(2*time.Second + 400*time.Millisecond):
		t.Fatal("initial discovery did not reach backoff")
	}
	t.Logf("initial discovery reached backoff in %s; status attempts: %d", time.Since(started), statusAttempts(t))
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal("cancellation lost its cause", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancellation waited for the startup backoff")
	}
	if statusAttempts(t) != 1 {
		t.Fatal("cancelled startup started another helper; status attempts:", statusAttempts(t))
	}
}
