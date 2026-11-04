//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The owner's walk-through of 0.1.2: the first `status --json` of a start stalled
// for a minute (the next one answers in a second), and the saved setup, which has
// a q8 cache and fast prefill, showed "Not checked" until a value was edited.

func TestStalledFirstStatusCallIsHedgedAndTheStallIsLogged(t *testing.T) {
	cfg := configFor(t, "normal")
	cfg.ColdStartTimeout, cfg.ColdStartHedge = 30*time.Second, 300*time.Millisecond
	t.Setenv("BUBO_FAKE_STATUS_DELAYS", "20s,0s")
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
	began := time.Now()
	waitStartup(t, s, func(v Snapshot) bool { return v.Initialized })
	if took := time.Since(began); took > 3*time.Second {
		t.Fatalf("a stalled first call held the start for %s", took)
	}
	if n := statusAttempts(t); n != 2 {
		t.Fatalf("status calls: %d, want the stalled one and its hedge", n)
	}
	lines := strings.Join(s.logs.Ring(), "\n")
	for _, want := range []string{"startup_hedge: the first status call has run for", "startup_hedge: call 2 of 2 answered", "cli_call: stopped before it answered pid="} {
		if !strings.Contains(lines, want) {
			t.Errorf("the log lost %q:\n%s", want, lines)
		}
	}
}

// A fast first call starts nothing beside it.
func TestFastFirstStatusCallIsNotHedged(t *testing.T) {
	cfg := configFor(t, "normal")
	cfg.ColdStartHedge = 300 * time.Millisecond
	t.Setenv("BUBO_FAKE_STATUS_DELAYS", "0s")
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	waitStartup(t, s, func(v Snapshot) bool { return v.Initialized })
	time.Sleep(600 * time.Millisecond)
	if n := statusAttempts(t); n != 1 {
		t.Fatalf("status calls: %d, want 1", n)
	}
}

// End to end, a fresh launch with the owner's saved setup: the real supervisor,
// the first status call stalled. The fit shows without any input, within a bound.
func TestFreshLaunchWithSavedOverridesShowsTheFitWithoutInput(t *testing.T) {
	cfg := configFor(t, "normal")
	cfg.ColdStartTimeout, cfg.ColdStartHedge = 30*time.Second, 300*time.Millisecond
	cfg.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
	t.Setenv("BUBO_FAKE_STATUS_DELAYS", "20s,0s")
	t.Setenv("BUBO_FAKE_RECORD_STATUS", "1")
	saved := DefaultSettings()
	saved.ManifestPath = filepath.Join(e21Folder(t, e21Flash), "bubo-package.json")
	saved.SwitchOverrides = map[string]string{"--kv-cache-type": "q8_0", "--prefill-mode": "fast"}
	if err := SaveSettings(cfg.SettingsPath, saved); err != nil {
		t.Fatal(err)
	}
	b, err := NewBackend(cfg, saved, cfg.Exe)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, b.s.Exit()) })
	a := newTestApp(b, saved, nil, "")
	a.AttachModels(cfg.SettingsPath, cfg.Exe, "")
	t.Cleanup(func() { a.models.Close() })
	began := time.Now()
	var v View
	for time.Since(began) < 8*time.Second {
		v = a.GetView()
		if v.Initialized && (v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error != nil) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if took := time.Since(began); took > 5*time.Second {
		t.Fatalf("the fit took %s to appear", took)
	}
	if v.SavedEstimate.Error != nil || v.SavedEstimate.MemoryEstimate == nil || v.Fit.Class == "fit_unknown" {
		t.Fatalf("a fresh launch with a saved q8 cache and fast prefill shows no fit: error %+v, fit %+v", v.SavedEstimate.Error, v.Fit)
	}
}
