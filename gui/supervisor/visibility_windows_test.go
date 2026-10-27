package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestVisibilityPolling(t *testing.T) {
	c := configFor(t, "ui")
	if err := os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "prepared"), []byte("true"), 0600); err != nil {
		t.Fatal(err)
	}
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	reply(t, s.Start())
	waitState(t, s, Ready)
	s.SetVisible(false)
	if s.PollInterval() != 10*time.Second {
		t.Fatal("hidden interval")
	}
	// Allow the one helper already in flight at Hide to finish. New hidden
	// queries must not occur at the test's 40ms visible cadence.
	time.Sleep(100 * time.Millisecond)
	first := s.Snapshot().Status.SampleTime
	// CLI sample timestamps have whole-second resolution. Observe a full second
	// so a hidden-payload regression cannot pass inside one timestamp bucket.
	time.Sleep(1100 * time.Millisecond)
	if v := s.Snapshot(); v.Status.SampleTime != first {
		t.Fatal("status kept sampling while hidden")
	}
	s.SetVisible(true)
	deadline := time.Now().Add(1500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if s.Snapshot().Status.SampleTime != first {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("visible sampling did not resume")
}
