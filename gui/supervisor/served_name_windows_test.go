//go:build windows

package supervisor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The CLI owns the public name (now the package ID by default); "bubo" remains
// a compatibility alias. The app passes a name only when the user set one and
// shows exactly the name the CLI reports.
func TestServeLeavesTheDefaultServedModelNameToTheCLI(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Start())
	waitState(t, s, Ready)
	line := serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))[0]
	if strings.Contains(line, "--served-model-name") {
		t.Fatalf("the app passed a served model name of its own: %q", line)
	}
	if got := s.Snapshot().Instance.ServedModel(); got != e21Flash {
		t.Fatalf("the CLI-reported served model name is %q", got)
	}
}

func TestServeKeepsTheNameTheUserSet(t *testing.T) {
	c := configFor(t, "normal")
	c.SettingsPath = filepath.Join(t.TempDir(), "settings.json")
	st := DefaultSettings()
	st.SwitchOverrides = map[string]string{"--served-model-name": "mine"}
	if err := SaveSettings(c.SettingsPath, st); err != nil {
		t.Fatal(err)
	}
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { <-s.Exit() })
	reply(t, s.Start())
	waitState(t, s, Ready)
	line := serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))[0]
	if strings.Count(line, "--served-model-name") != 1 || !strings.Contains(line, "--served-model-name mine") {
		t.Fatalf("the user's name must be the only one: %q", line)
	}
}
