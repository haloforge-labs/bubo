//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Exercise the real supervisor and model service together against the fake CLI;
// no engine, server, desktop window or network listener is started.
func s15App(t *testing.T, mode, savedVersion string) (*AppService, *NativeBackend, Config) {
	t.Helper()
	c := configFor(t, mode)
	c.InitialPrepared = false
	c.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "gui", "settings.json")
	t.Setenv("LOCALAPPDATA", filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "local"))
	t.Setenv("USERPROFILE", filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "profile"))
	t.Setenv("BUBO_FAKE_PROGRESS_PAUSE_MS", "0")
	settings := DefaultSettings()
	settings.LastError = &Failure{Code: "unexpected_failure", Time: time.Now().UTC()}
	data, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	var saved map[string]json.RawMessage
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatal(err)
	}
	if savedVersion != "" {
		saved["last_error_version"], _ = json.Marshal(savedVersion)
	}
	data, _ = json.Marshal(saved)
	if err := os.MkdirAll(filepath.Dir(c.SettingsPath), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(c.SettingsPath, data, 0600); err != nil {
		t.Fatal(err)
	}
	install := t.TempDir()
	if err := os.WriteFile(filepath.Join(install, "BUILD-MANIFEST.json"), []byte(`{"product":{"product_version":"0.1.1"}}`), 0600); err != nil {
		t.Fatal(err)
	}
	b, err := NewBackend(c, settings, filepath.Join(install, "bubo-gui.exe"))
	if err != nil {
		t.Fatal(err)
	}
	a := newTestApp(b, settings, nil, "")
	a.AttachModels(c.SettingsPath, fakeExe, "")
	b.SetEventSink(a.emitStream)
	t.Cleanup(func() { a.models.Close(); reply(t, b.s.Exit()) })
	s15Wait(t, func() bool { return b.Snapshot().Initialized }, "initial discovery")
	return a, b, c
}

func s15Wait(t *testing.T, pred func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if pred() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out waiting for " + what)
}

func s15ListCount(t *testing.T) int {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "models-argv-*.json"))
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var args []string
		if err := json.Unmarshal(data, &args); err != nil {
			t.Fatal(err)
		}
		if len(args) > 1 && args[0] == "models" && args[1] == "list" {
			n++
		}
	}
	return n
}

func TestS15EngineCompletionRefreshesFailedCatalogueOnce(t *testing.T) {
	a, b, _ := s15App(t, "ui", "0.1.1")
	s15Wait(t, func() bool {
		v := a.GetView()
		return v.Models != nil && !v.Models.Refreshing && v.Models.ReasonCode == "runtime_missing"
	}, "initial runtime_missing list")
	a.mu.Lock()
	a.estimate.at = time.Now().Add(time.Hour)
	a.mu.Unlock()
	// Do not let a later watch sample conceal a missing completion hook. The
	// fake's existing watch-stall injection suppresses status payloads.
	if err := os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "watch-stall"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	reply(t, b.s.Prepare())
	s15Wait(t, func() bool { return b.Snapshot().Approval != nil }, "engine approval")
	reply(t, b.Action("approve", DefaultSettings(), ""))
	waitState(t, b.s, Stopped)
	s15Wait(t, func() bool {
		v := a.GetView()
		return v.Models != nil && !v.Models.Refreshing && v.Models.Status == ModelsOK && len(v.Models.Catalogue) > 0 && v.Models.ReasonCode == ""
	}, "automatic catalogue recovery")
	if b.Snapshot().LastError != nil {
		t.Fatal("engine success retained the saved error")
	}
	a.mu.RLock()
	at := a.estimate.at
	a.mu.RUnlock()
	if at.After(time.Now()) {
		t.Fatal("engine readiness did not reset the estimate time")
	}
	// A successful status discovery after completion must not cause a second
	// refresh for that same transition.
	reply(t, b.s.Discover())
	for range 10 {
		a.GetView()
	}
	if a.models.State().Refreshing {
		t.Fatal("unchanged ready status started another catalogue refresh")
	}
	if got := s15ListCount(t); got != 2 {
		t.Fatalf("list reads = %d, want initial failure + one recovery", got)
	}
}

func TestS15StatusRuntimeReadinessRefreshesFailedCatalogue(t *testing.T) {
	a, b, _ := s15App(t, "ui", "0.1.1")
	s15Wait(t, func() bool {
		v := a.GetView()
		return v.Models != nil && !v.Models.Refreshing && v.Models.ReasonCode == "runtime_missing"
	}, "initial list failure")
	if err := os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "runtime-prepared"), []byte("true"), 0600); err != nil {
		t.Fatal(err)
	}
	reply(t, b.s.Discover())
	s15Wait(t, func() bool {
		v := a.GetView()
		return v.Models != nil && !v.Models.Refreshing && v.Models.Status == ModelsOK && len(v.Models.Catalogue) > 0
	}, "catalogue after status readiness")
	reply(t, b.s.Discover())
	if a.models.State().Refreshing {
		t.Fatal("unchanged ready status started another catalogue refresh")
	}
	if got := s15ListCount(t); got != 2 {
		t.Fatalf("list reads = %d, want 2", got)
	}
}

func TestS15SavedErrorVersionAndLaterSuccess(t *testing.T) {
	for _, version := range []string{"", "0.1.0", "0.1.1"} {
		t.Run("saved-"+version, func(t *testing.T) {
			a, b, c := s15App(t, "ui", version)
			_ = a
			retained := b.Snapshot().LastError != nil
			if retained != (version == "0.1.1") {
				t.Fatalf("saved version %q: retained = %v", version, retained)
			}
			saved, warning := LoadSettings(c.SettingsPath)
			if warning != "" || (saved.LastError != nil) != retained {
				t.Fatalf("persisted stale error: %+v, %q", saved.LastError, warning)
			}
			if retained {
				// A later model readiness transition is also a successful operation.
				if err := os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "prepared"), []byte("true"), 0600); err != nil {
					t.Fatal(err)
				}
				reply(t, b.s.Discover())
				s15Wait(t, func() bool { return b.Snapshot().LastError == nil }, "model readiness to clear the error")
				saved, _ = LoadSettings(c.SettingsPath)
				if saved.LastError != nil {
					t.Fatal("model readiness did not clear persisted error")
				}
			}
		})
	}
}

func TestS15SuccessfulModelJobClearsSavedError(t *testing.T) {
	a, b, c := s15App(t, "normal", "0.1.1")
	if b.Snapshot().LastError == nil {
		t.Fatal("same-version error should survive initial ready status")
	}
	if result := a.Download(e21Flash); result.Error != nil {
		t.Fatal(result.Error)
	}
	if result := a.ConfirmDownload(e21Flash); result.Error != nil {
		t.Fatal(result.Error)
	}
	s15Wait(t, func() bool { job := a.models.State().Download; return job != nil && job.State == "done" }, "successful model job")
	s15Wait(t, func() bool { return b.Snapshot().LastError == nil }, "model success to clear error")
	saved, _ := LoadSettings(c.SettingsPath)
	if saved.LastError != nil {
		t.Fatal("model success retained persisted error")
	}
}

func TestS15NewFailureUsesCurrentVersionAndServerSuccessClearsIt(t *testing.T) {
	_, b, c := s15App(t, "normal", "0.1.0")
	b.s.emit(update{b.Snapshot().Generation, "failure", Failure{Code: "unexpected_failure", Source: "supervisor"}})
	s15Wait(t, func() bool { return b.Snapshot().LastError != nil }, "new failure")
	data, err := os.ReadFile(c.SettingsPath)
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		Version string `json:"last_error_version"`
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Version != "0.1.1" {
		t.Fatalf("saved error version = %q", saved.Version)
	}
	reply(t, b.s.Start())
	waitState(t, b.s, Ready)
	if b.Snapshot().LastError != nil {
		t.Fatal("Ready retained error")
	}
	settings, _ := LoadSettings(c.SettingsPath)
	if settings.LastError != nil {
		t.Fatal("Ready retained persisted error")
	}
	reply(t, b.s.Stop())
	waitState(t, b.s, Stopped)
	// A new failure after success survives repeated ready status samples.
	b.s.emit(update{b.Snapshot().Generation, "failure", Failure{Code: "unexpected_failure", Source: "supervisor"}})
	s15Wait(t, func() bool { return b.Snapshot().LastError != nil }, "later failure")
	reply(t, b.s.Discover())
	if b.Snapshot().LastError == nil {
		t.Fatal("unchanged ready status cleared a later failure")
	}
	// Persistence is updated before the discovery reply, while Snapshot
	// publication follows it. Check disk too so a stale Snapshot cannot pass.
	settings, _ = LoadSettings(c.SettingsPath)
	if settings.LastError == nil {
		t.Fatal("unchanged ready status cleared the persisted later failure")
	}
	reply(t, b.s.DismissError())
	s15Wait(t, func() bool { return b.Snapshot().LastError == nil }, "explicit dismiss")
	settings, _ = LoadSettings(c.SettingsPath)
	if settings.LastError != nil {
		t.Fatal("dismiss retained persisted error")
	}
}
