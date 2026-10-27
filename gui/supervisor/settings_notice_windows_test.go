//go:build windows

package supervisor

import (
	"encoding/json"
	"testing"
)

// E21w W5. Every Settings save set the notice "Restart to apply these settings."
// even for the model download folder (read at each prepare) and when no server
// runs. The notice is true only when a running server would start differently.
//
// Ways this could fail, written before the code:
//
//	N1 a download-folder save with a server running says "Restart".
//	N2 any save with no server running says "Restart" (nothing runs).
//	N3 a serve setting changed while a server runs with another value says
//	   nothing: the one true case is lost.
//	N4 a serve setting set to the value the server already runs says "Restart".
//	N5 clearing an override the running server was started with says nothing
//	   (it would start with the CLI default).
//	N6 an earlier unapplied serve change is forgotten by a later unrelated save
//	   (the server still would start differently, and the notice is still true).
//	N7 the registry is not known yet (CLI has not listed it), a server runs and a
//	   serve setting changed: the notice is dropped.
//	N8 the same registry and settings with the server stopped: the notice returns.
//	N9 the "Dropped stale overrides" notice loses its precedence (Save).
//
// The registry is the one the candidate exe printed (recordings/e21/switches.json).
// The recorded status is an idle server with no effective switches, so the
// running server is built here: one instance whose effective context is set by
// the test.
func e21Registry(t *testing.T) *Switches {
	t.Helper()
	return e21Events(t, "switches.json")[0].Value.(*Switches)
}

func running(effective map[string]string, started map[string]string) Snapshot {
	var rows []EffectiveSwitch
	for name, value := range effective {
		raw, _ := json.Marshal(value)
		rows = append(rows, EffectiveSwitch{Name: name, Effective: raw})
	}
	return Snapshot{State: Ready, Instance: &Instance{},
		Status:                 &Status{RuntimeDetail: RuntimeDetail{EffectiveSwitches: rows}},
		RunningSwitchOverrides: started}
}

func TestSaveNoticeOnlyWhenARunningServerWouldStartDifferently(t *testing.T) {
	registry := e21Registry(t)
	runs := running(map[string]string{"--context": "65536", "--host": "127.0.0.1"}, nil)
	stopped := Snapshot{State: Stopped}

	t.Run("download folder, server running (N1)", func(t *testing.T) {
		s := Settings{Context: 65536, DataFolder: `D:\Bubo downloads`, DownloadFolders: []string{`D:\Bubo downloads`}}
		if detail, code := settingsNotice(Settings{Context: 65536}, s, runs, registry); detail != "" || code != "" {
			t.Fatalf("%q %q", detail, code)
		}
	})
	t.Run("download folder, the server reports no host (N1)", func(t *testing.T) {
		quiet := running(map[string]string{"--context": "65536"}, nil)
		s := Settings{Context: 65536, DataFolder: `D:\Bubo downloads`}
		if detail, _ := settingsNotice(Settings{Context: 65536}, s, quiet, registry); detail != "" {
			t.Fatalf("%q", detail)
		}
	})
	t.Run("serve setting, no server (N2 N8)", func(t *testing.T) {
		s := Settings{Context: 131072}
		if detail, _ := settingsNotice(Settings{}, s, stopped, registry); detail != "" {
			t.Fatalf("%q", detail)
		}
	})
	t.Run("serve setting differs from the running one (N3)", func(t *testing.T) {
		s := Settings{Context: 131072}
		detail, code := settingsNotice(Settings{Context: 65536}, s, runs, registry)
		if detail != "Restart to apply these settings." || code != "settings_saved" {
			t.Fatalf("%q %q", detail, code)
		}
	})
	t.Run("serve setting equals the running one (N4)", func(t *testing.T) {
		s := Settings{Context: 65536}
		if detail, _ := settingsNotice(Settings{Context: 131072}, s, runs, registry); detail != "" {
			t.Fatalf("%q", detail)
		}
	})
	t.Run("clearing an override the server started with (N5)", func(t *testing.T) {
		started := running(map[string]string{"--max-tokens": "512"}, map[string]string{"--max-tokens": "512"})
		if detail, _ := settingsNotice(Settings{SwitchOverrides: map[string]string{"--max-tokens": "512"}}, Settings{}, started, registry); detail == "" {
			t.Fatal("the server would start with the CLI default: the notice is true")
		}
	})
	t.Run("an earlier unapplied change and an unrelated save (N6)", func(t *testing.T) {
		pending := Settings{Context: 131072}
		next := pending
		next.CloseToTray = true
		if detail, _ := settingsNotice(pending, next, runs, registry); detail == "" {
			t.Fatal("the server still would start differently")
		}
	})
	t.Run("registry unknown (N7)", func(t *testing.T) {
		if detail, _ := settingsNotice(Settings{Context: 65536}, Settings{Context: 131072}, runs, nil); detail == "" {
			t.Fatal("a changed serve setting with a running server and no registry")
		}
		if detail, _ := settingsNotice(Settings{Context: 65536}, Settings{Context: 65536, DataFolder: `D:\x`}, runs, nil); detail != "" {
			t.Fatalf("a download folder is no serve setting: %q", detail)
		}
	})
}
