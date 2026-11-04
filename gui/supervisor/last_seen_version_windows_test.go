//go:build windows

package supervisor

import (
	"path/filepath"
	"testing"
)

// The What's new dialog records the last version it showed in the app settings.
func TestLastSeenVersionPatchSaveLoadAndClear(t *testing.T) {
	version := "0.1.2"
	next, err := ApplyPreferencesPatch(DefaultSettings(), PreferencesPatch{Set: PreferenceValues{LastSeenVersion: &version}})
	if err != nil || next.LastSeenVersion != version {
		t.Fatalf("patch: %+v, %v", next, err)
	}
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := SaveSettings(path, next); err != nil {
		t.Fatal(err)
	}
	loaded, warning := LoadSettings(path)
	if warning != "" || loaded.LastSeenVersion != version {
		t.Fatalf("load: %+v, %q", loaded, warning)
	}
	cleared, err := ApplyPreferencesPatch(loaded, PreferencesPatch{Clear: []string{"last_seen_version"}})
	if err != nil || cleared.LastSeenVersion != "" {
		t.Fatalf("clear: %+v, %v", cleared, err)
	}
	// An older settings file without the field loads as never seen.
	older, _ := LoadSettings(filepath.Join(t.TempDir(), "missing.json"))
	if older.LastSeenVersion != "" {
		t.Fatalf("default: %+v", older)
	}
}
