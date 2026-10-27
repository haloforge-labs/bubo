//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestExternalLostSampleAndExit(t *testing.T) {
	c := configFor(t, "normal")
	p, i := startExternal(t, c)
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	reply(t, s.Discover())
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "lose-status"), nil, 0600)
	deadline := time.Now().Add(time.Second)
	for !s.Snapshot().Stale && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if v := s.Snapshot(); !v.Stale || v.State != AttachedExternal {
		t.Fatal("external sample loss became exit", v)
	}
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal(err)
	}
	p.Force()
	<-p.Done
	waitState(t, s, Stopped)
	reply(t, s.Exit())
}
