//go:build windows

package supervisor

import "testing"

func partRegistry() *Switches {
	return &Switches{Switches: []Switch{
		{Name: "--twin-model", Kind: "flag", Class: "DEFAULT", Requires: []string{"artefact:twin"}, GUIControl: true},
		{Name: "--draft-head-model", Kind: "flag", Class: "DEFAULT", Requires: []string{"artefact:draft_head"}, GUIControl: true},
		{Name: "--temperature", Kind: "flag", Class: "DEFAULT", GUIControl: true},
	}}
}

// Wrapper review 4.5: part overrides follow the model. Choosing another model
// drops the overrides that point at a part file of the old one.
func TestChangingTheModelDropsPartFileOverridesOnly(t *testing.T) {
	for name, patch := range map[string]PreferencesPatch{
		"another manifest": {Set: PreferenceValues{ManifestPath: ptr(`D:\b\bubo-package.json`)}},
		"another file":     {Set: PreferenceValues{ModelPath: ptr(`D:\b\model.gguf`)}},
		"model cleared":    {Clear: []string{"manifest_path"}},
	} {
		t.Run(name, func(t *testing.T) {
			b := &registryBackend{reg: partRegistry()}
			initial := DefaultSettings()
			initial.ManifestPath = `D:\a\bubo-package.json`
			initial.SwitchOverrides = map[string]string{"--twin-model": `D:\a\twin.gguf`, "--draft-head-model": `D:\a\head.gguf`, "--temperature": "0.5"}
			a := newTestApp(b, initial, nil, "")
			if r := a.PatchPreferences(patch); r.Error != nil {
				t.Fatal(r.Error)
			}
			a.mu.RLock()
			got := a.settings.SwitchOverrides
			a.mu.RUnlock()
			if len(got) != 1 || got["--temperature"] != "0.5" {
				t.Fatalf("overrides after the change: %v", got)
			}
		})
	}
}

func TestOtherPreferencesAndTheSameModelKeepPartFileOverrides(t *testing.T) {
	b := &registryBackend{reg: partRegistry()}
	initial := DefaultSettings()
	initial.ManifestPath = `D:\a\bubo-package.json`
	initial.SwitchOverrides = map[string]string{"--twin-model": `D:\a\twin.gguf`}
	a := newTestApp(b, initial, nil, "")
	same := initial.ManifestPath
	for _, patch := range []PreferencesPatch{
		{Set: PreferenceValues{ManifestPath: &same}},
		{Set: PreferenceValues{Sessions: ptr(uint32(2))}},
	} {
		if r := a.PatchPreferences(patch); r.Error != nil {
			t.Fatal(r.Error)
		}
		a.mu.RLock()
		got := a.settings.SwitchOverrides
		a.mu.RUnlock()
		if got["--twin-model"] == "" {
			t.Fatalf("a patch that kept the model dropped the override: %v", got)
		}
	}
}

func ptr[T any](v T) *T { return &v }
