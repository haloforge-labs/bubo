//go:build windows

package supervisor

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"
)

func TestInferenceDisclosurePreferencePersistsWithoutChangingServerSetup(t *testing.T) {
	initial := DefaultSettings()
	initial.Context, initial.AllowTightSetup, initial.TightSetupFingerprint = 200000, true, "approved"
	path := filepath.Join(t.TempDir(), "settings.json")
	for _, open := range []bool{true, false} {
		patch := writePreferencePatch(t, `{"set":{"inference_expanded":`+map[bool]string{true: "true", false: "false"}[open]+`},"clear":[]}`)
		next, err := ApplyPreferencesPatch(initial, patch)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(startArgs(initial), startArgs(next)) || tightSetupChanged(initial, next) || !next.AllowTightSetup || next.TightSetupFingerprint != "approved" {
			t.Fatal("disclosure preference changed the server setup or Tight consent")
		}
		if err := SaveSettings(path, next); err != nil {
			t.Fatal(err)
		}
		loaded, warning := LoadSettings(path)
		if warning != "" {
			t.Fatal(warning)
		}
		view := (&ViewBuilder{}).Build(ViewInput{Saved: loaded})
		data, err := json.Marshal(view.Saved)
		if err != nil {
			t.Fatal(err)
		}
		var saved map[string]json.RawMessage
		if err := json.Unmarshal(data, &saved); err != nil {
			t.Fatal(err)
		}
		want := map[bool]string{true: "true", false: "false"}[open]
		if string(saved["inference_expanded"]) != want {
			t.Fatalf("saved inference_expanded = %s, want %s", saved["inference_expanded"], want)
		}
		initial = loaded
	}
	cleared, err := ApplyPreferencesPatch(initial, PreferencesPatch{Clear: []string{"inference_expanded"}})
	if err != nil {
		t.Fatal(err)
	}
	view := (&ViewBuilder{}).Build(ViewInput{Saved: cleared})
	data, _ := json.Marshal(view.Saved)
	var saved map[string]json.RawMessage
	_ = json.Unmarshal(data, &saved)
	if string(saved["inference_expanded"]) != "false" {
		t.Fatal("cleared disclosure must default to collapsed")
	}
}
