//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// E14 item 4: a catalogue model the user downloaded and selected must start
// even though the default package (Flash Next) is not prepared. The status
// readiness describes the default package only.
func catalogueSelectionRig(t *testing.T, selected bool) *Supervisor {
	t.Helper()
	c := configFor(t, "ui")
	dir := os.Getenv("BUBO_FAKE_DIR")
	// The engine files are ready; the default model is not.
	if err := os.WriteFile(filepath.Join(dir, "runtime-prepared"), []byte("true"), 0600); err != nil {
		t.Fatal(err)
	}
	c.SettingsPath = filepath.Join(t.TempDir(), "settings.json")
	st := DefaultSettings()
	if selected {
		st.ManifestPath = filepath.Join(e21Folder(t, e21Hauhau), "bubo-package.json")
	}
	if err := SaveSettings(c.SettingsPath, st); err != nil {
		t.Fatal(err)
	}
	c.InitialPrepared = false
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { <-s.Exit() })
	waitFor(t, s, "discovery", 8*time.Second, func(v Snapshot) bool { return v.Initialized })
	return s
}

func TestSelectedCatalogueModelStartsWithoutTheDefaultPackage(t *testing.T) {
	s := catalogueSelectionRig(t, true)
	v := waitFor(t, s, "not blocked by the default package", 8*time.Second, func(v Snapshot) bool { return v.State == Stopped })
	if v.State != Stopped {
		t.Fatalf("state %s", v.State)
	}
	saved, warning := LoadSettings(s.config.SettingsPath)
	if warning != "" {
		t.Fatal(warning)
	}
	// The desktop backend supplies the selected setup as launch arguments.
	reply(t, s.Start(startArgs(saved)...))
	waitState(t, s, Ready)
}

func TestWithoutASelectionTheDefaultPackageIsStillRequired(t *testing.T) {
	s := catalogueSelectionRig(t, false)
	time.Sleep(300 * time.Millisecond)
	if v := s.Snapshot(); v.State != NotPrepared {
		t.Fatalf("an unselected default package must stay required: %s", v.State)
	}
}

// The models list reports every prepared package with its manifest and its main
// model file (CLI-CONTRACT), and the supervisor keeps both as the CLI wrote them
// (E21: the rows are the recorded ones).
func TestModelsListKeepsBothPathsOfEveryRecordedPackage(t *testing.T) {
	r := newModelRig(t, "")
	state := r.refresh(t)
	recorded := e21Models(t).PreparedModels
	if len(state.PreparedModels) != len(recorded) || len(recorded) == 0 {
		t.Fatalf("rows: %d, recorded %d", len(state.PreparedModels), len(recorded))
	}
	for _, want := range recorded {
		found := false
		for _, row := range state.PreparedModels {
			if row.ID != want.ID || row.ManifestPath == nil || want.ManifestPath == nil || *row.ManifestPath != *want.ManifestPath {
				continue
			}
			found = true
			if row.ModelPath == nil || want.ModelPath == nil || *row.ModelPath != *want.ModelPath {
				t.Fatalf("model_path lost: %+v, recorded %+v", row, want)
			}
		}
		if !found {
			t.Fatalf("the prepared package %s is missing: %+v", want.ID, state.PreparedModels)
		}
	}
}

// E14 item 7: the saved-setup estimate passes the saved profile; the CLI's
// estimate depends on it.
func TestSavedSetupEstimatePassesTheProfile(t *testing.T) {
	r := newModelRig(t, "")
	if _, err := r.svc.estimateRuntime(e21Flash, 0, 0, "", "exact"); err != nil {
		t.Fatal(err)
	}
	args := r.argv(t, "models", 1)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--profile exact") {
		t.Fatalf("no --profile: %q", joined)
	}
	if _, err := r.svc.estimateRuntime(e21Flash, 0, 0, "", ""); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.Join(r.argv(t, "models", 2), " "), "--profile") {
		t.Fatal("an unset profile must add no flag")
	}
}
