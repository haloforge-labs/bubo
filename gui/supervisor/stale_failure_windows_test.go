//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A saved failure is shown as current only while the setup it happened under still
// holds. The owner's walk-through file kept an old start failure of a pre-release
// build (the refusal switch, since removed) and showed "the server did not start
// because of the setting you changed" at every launch.
func TestSavedFailureIsShownOnlyWhileItsSetupHolds(t *testing.T) {
	huihui := `C:\models\huihui\bubo-package.json`
	failed := func(switchName string) Settings {
		s := DefaultSettings()
		s.ManifestPath = huihui
		s.SwitchOverrides = map[string]string{"--kv-cache-type": "q8_0"}
		s.LastError = &Failure{Code: "invalid_arguments", Source: "cli", Time: time.Now().UTC(), Switch: switchName}
		s.lastErrorVersion, s.lastErrorSetup = "0.1.2", s.failureSetup()
		return s
	}
	cases := []struct {
		name   string
		change func(*Settings)
		holds  bool
	}{
		{"same setup, no switch named", func(s *Settings) { s.LastError.Switch = "" }, true},
		{"the named switch is still set", func(s *Settings) { s.SwitchOverrides["--refusal-direction"] = "on" }, true},
		{"the named switch is back at its default", func(s *Settings) {}, false},
		{"another model", func(s *Settings) { s.LastError.Switch = ""; s.ManifestPath = `C:\models\other\bubo-package.json` }, false},
		{"a model file instead of the package", func(s *Settings) { s.LastError.Switch = ""; s.ManifestPath = ""; s.ModelPath = `S:\m\a.gguf` }, false},
		{"saved by a build that did not record the setup", func(s *Settings) { s.LastError.Switch = ""; s.lastErrorSetup = "" }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := failed("--refusal-direction")
			tc.change(&s)
			if got := s.savedFailureHolds(); got != tc.holds {
				t.Fatalf("savedFailureHolds = %v, want %v", got, tc.holds)
			}
		})
	}
}

// The owner's final6 walk-through: a start was refused for memory at 262144 x 4
// with a q8_0 cache. The refusal says nothing about another context, session count,
// profile or cache type, so it must not outlive such a change or an update.
func TestMemoryRefusalIsShownOnlyForTheSetupItWasMadeFor(t *testing.T) {
	refused := func() Settings {
		s := DefaultSettings()
		s.ManifestPath = `C:\models\iq3\bubo-package.json`
		s.Context, s.Sessions = 262144, 4
		s.SwitchOverrides = map[string]string{"--kv-cache-type": "q8_0", "--prefill-mode": "fast"}
		s.LastError = &Failure{Code: "memory_tight", Source: "supervisor", Time: time.Now().UTC()}
		s.lastErrorVersion, s.lastErrorSetup = "0.1.2", s.failureSetup()
		return s
	}
	cases := []struct {
		name   string
		change func(*Settings)
		holds  bool
	}{
		{"nothing changed", func(s *Settings) {}, true},
		{"prefill mode changed", func(s *Settings) { s.SwitchOverrides["--prefill-mode"] = "balanced" }, true},
		{"smaller context", func(s *Settings) { s.Context = 131072 }, false},
		{"fewer sessions", func(s *Settings) { s.Sessions = 1 }, false},
		{"f16 cache", func(s *Settings) { s.SwitchOverrides["--kv-cache-type"] = "f16" }, false},
		{"cache type back at its default", func(s *Settings) { delete(s.SwitchOverrides, "--kv-cache-type") }, false},
		{"another profile", func(s *Settings) { s.Profile = "low" }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := refused()
			tc.change(&s)
			if got := s.savedFailureHolds(); got != tc.holds {
				t.Fatalf("savedFailureHolds = %v, want %v", got, tc.holds)
			}
		})
	}
	// A new version drops it at the next start, whatever the setup.
	updated := refused()
	updated.lastErrorVersion = "0.1.1"
	v, after := staleFailureStart(t, updated)
	if v.LastError != nil || after.LastError != nil {
		t.Fatalf("a refusal of an older version is shown: %+v", v.LastError)
	}
	// The same setup and version keep it, after a restart.
	if v, _ = staleFailureStart(t, refused()); v.LastError == nil || v.LastError.Code != "memory_tight" {
		t.Fatalf("a refusal of the current setup was dropped: %+v", v.LastError)
	}
}

func staleFailureStart(t *testing.T, saved Settings) (Snapshot, Settings) {
	t.Helper()
	c := configFor(t, "normal")
	c.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
	c.ProductVersion = "0.1.2"
	if err := SaveSettings(c.SettingsPath, saved); err != nil {
		t.Fatal(err)
	}
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { reply(t, s.Exit()) }()
	after, warning := LoadSettings(c.SettingsPath)
	if warning != "" {
		t.Fatal(warning)
	}
	return s.Snapshot(), after
}

// The owner's file: this version, an old failure that names a switch the saved
// setup no longer sets, written by a build that recorded no setup.
func TestOwnerWalkThroughFileDoesNotStartWithAStaleBanner(t *testing.T) {
	s := DefaultSettings()
	s.ManifestPath = `C:\Users\example\AppData\Local\Bubo\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\bubo-package.json`
	s.SwitchOverrides = map[string]string{"--kv-cache-type": "q8_0", "--prefill-mode": "fast"}
	s.LastError = &Failure{Code: "invalid_arguments", Source: "cli", Time: time.Now().UTC(), Switch: "--refusal-direction"}
	s.lastErrorVersion = "0.1.2"
	v, after := staleFailureStart(t, s)
	if v.LastError != nil {
		t.Fatalf("the stale failure is shown: %+v", v.LastError)
	}
	if after.LastError != nil || after.lastErrorVersion != "" || after.lastErrorSetup != "" {
		t.Fatalf("the stale failure was kept on disk: %+v", after.LastError)
	}
}

func TestCurrentFailureSurvivesARestart(t *testing.T) {
	s := DefaultSettings()
	s.ManifestPath = `C:\models\huihui\bubo-package.json`
	s.SwitchOverrides = map[string]string{"--refusal-direction": "on"}
	s.LastError = &Failure{Code: "invalid_arguments", Source: "cli", Time: time.Now().UTC(), Switch: "--refusal-direction"}
	s.lastErrorVersion, s.lastErrorSetup = "0.1.2", s.failureSetup()
	v, _ := staleFailureStart(t, s)
	if v.LastError == nil || v.LastError.Switch != "--refusal-direction" {
		t.Fatalf("a failure of the current setup was dropped: %+v", v.LastError)
	}
}

// Saving settings carries the failure with the version and setup it was recorded
// under. It must not stamp an old failure with the running version.
func TestSettingsSaveKeepsTheFailureStamps(t *testing.T) {
	c := configFor(t, "normal")
	c.SettingsPath = filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "settings.json")
	c.ProductVersion = "0.1.2"
	disk := DefaultSettings()
	disk.ManifestPath = `C:\models\a\bubo-package.json`
	disk.LastError = &Failure{Code: "device_lost", Source: "cli", Time: time.Now().UTC()}
	disk.lastErrorVersion, disk.lastErrorSetup = "0.1.2", disk.failureSetup()
	if err := SaveSettings(c.SettingsPath, disk); err != nil {
		t.Fatal(err)
	}
	install := t.TempDir()
	if err := os.WriteFile(filepath.Join(install, "BUILD-MANIFEST.json"), []byte(`{"product":{"product_version":"0.1.2"}}`), 0600); err != nil {
		t.Fatal(err)
	}
	b, err := NewBackend(c, disk, filepath.Join(install, "bubo-gui.exe"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { reply(t, b.s.Exit()) }()
	next := disk
	next.ManifestPath = `C:\models\b\bubo-package.json`
	if err := b.Save(next); err != nil {
		t.Fatal(err)
	}
	saved, warning := LoadSettings(c.SettingsPath)
	if warning != "" {
		t.Fatal(warning)
	}
	if saved.ManifestPath != next.ManifestPath || saved.lastErrorSetup != disk.failureSetup() {
		t.Fatalf("the save re-stamped the failure: %+v %q", saved.ManifestPath, saved.lastErrorSetup)
	}
	if saved.savedFailureHolds() {
		t.Fatal("a failure of model A holds for model B")
	}
}
