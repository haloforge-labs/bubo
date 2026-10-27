//go:build windows

package supervisor

import (
	"encoding/json"
	"strings"
	"testing"
)

const tightExplanation = "The server would start, but there is little room for the context cache (the KV cache), which grows as a conversation gets longer. If it runs out, the server stops with an out-of-memory error, or Windows slows down."
const doesNotFitExplanation = "These settings need more memory than this computer has, so Bubo can’t start them."

func TestV12ThreeFitVerdictsAndTightAdmission(t *testing.T) {
	in := actionTestInput(Stopped)
	fit := true
	recommended := false
	in.SavedEstimate = &MemoryEstimate{Fits: &fit, Recommended: &recommended, FitScope: "conservative_startup_estimate"}
	if got := buildFit(in.SavedEstimate); got.Class != "memory_tight" || got.Sentence != tightExplanation {
		t.Fatalf("tight verdict: %+v", got)
	}
	if got := lifecycleActions(in).Start; got.Allowed || got.ReasonCode != "memory_tight" {
		t.Fatalf("tight start: %+v", got)
	}
	in.Saved.AllowTightSetup = true
	in.Saved.ModelPath, in.Saved.Context, in.Saved.Sessions, in.Saved.Profile = `C:\model.gguf`, 65536, 2, "vanilla"
	key, ok := tightSetupFingerprint(in.Saved, in.Snapshot.Status)
	if !ok {
		t.Fatal("explicit setup was not identified")
	}
	in.Saved.TightSetupFingerprint = key
	if got := lifecycleActions(in).Start; !got.Allowed {
		t.Fatalf("opted-in start: %+v", got)
	}
	no := false
	in.SavedEstimate.Fits = &no
	if got := buildFit(in.SavedEstimate); got.Sentence != doesNotFitExplanation {
		t.Fatalf("doesn't fit wording: %+v", got)
	}
	if got := lifecycleActions(in).Start; got.Allowed {
		t.Fatalf("doesn't fit with opt-in: %+v", got)
	}
}

func TestV12TightConsentIsScopedToSavedSetup(t *testing.T) {
	base := DefaultSettings()
	base.AllowTightSetup = true
	base.ModelPath, base.Context, base.Sessions, base.Profile = `C:\model.gguf`, 65536, 2, "vanilla"
	port := 19000
	for _, tc := range []struct {
		name  string
		patch PreferencesPatch
		keep  bool
	}{
		{"unchanged", PreferencesPatch{Set: PreferenceValues{Port: &port}}, true},
		{"model", PreferencesPatch{Clear: []string{"model_path"}}, false},
		{"context", PreferencesPatch{Clear: []string{"context"}}, false},
		{"sessions", PreferencesPatch{Clear: []string{"sessions"}}, false},
		{"profile", PreferencesPatch{Clear: []string{"profile"}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			next, err := ApplyPreferencesPatch(base, tc.patch)
			if err != nil || next.AllowTightSetup != tc.keep {
				t.Fatalf("consent = %v, err = %v", next.AllowTightSetup, err)
			}
		})
	}
	path := t.TempDir() + `\settings.json`
	if err := SaveSettings(path, base); err != nil {
		t.Fatal(err)
	}
	loaded, warning := LoadSettings(path)
	if warning != "" || !loaded.AllowTightSetup {
		t.Fatalf("consent persistence: %+v %s", loaded, warning)
	}
}

func TestV12RegistryNumberIsExactThroughView(t *testing.T) {
	const number = "9007199254740993"
	raw := json.RawMessage(`{"kind":"integer","minimum":9007199254740993,"choices":[9007199254740993]}`)
	view := new(ViewBuilder).Build(ViewInput{Registry: &Switches{Switches: []Switch{{Name: "--large", Allowed: raw}}}})
	b, err := json.Marshal(view.Switches)
	if err != nil || !strings.Contains(string(b), number) || strings.Contains(string(b), "9007199254740992") {
		t.Fatalf("large integer changed: %s %v", b, err)
	}
}

func TestV12AtomicPreferenceAndRegistryPatch(t *testing.T) {
	registry := &Switches{Switches: []Switch{{Name: "--future", Kind: "flag", GUIControl: true}}}
	b := &registryBackend{reg: registry}
	a := newTestApp(b, DefaultSettings(), nil, "")
	context := uint64(131072)
	patch := PreferencesPatch{Set: PreferenceValues{Context: &context}, SwitchValues: []SwitchOverride{{Name: "--future", Value: "enabled"}}}
	if result := a.PatchPreferences(patch); result.Error != nil || a.settings.Context != context || a.settings.SwitchOverrides["--future"] != "enabled" {
		t.Fatalf("combined patch: %+v %+v", result, a.settings)
	}
	if result := a.PatchPreferences(PreferencesPatch{Clear: []string{"context", "--future"}}); result.Error != nil || a.settings.Context != 0 || len(a.settings.SwitchOverrides) != 0 {
		t.Fatalf("combined clear: %+v %+v", result, a.settings)
	}
	patch = PreferencesPatch{Set: PreferenceValues{Context: &context}, SwitchValues: []SwitchOverride{{Name: "--missing", Value: "bad"}}}
	if result := a.PatchPreferences(patch); result.Error == nil || a.settings.Context != 0 {
		t.Fatalf("invalid switch partially saved preferences: %+v %+v", result, a.settings)
	}
}

func TestV12AtomicModelChangeKeepsNewlySelectedPart(t *testing.T) {
	initial := DefaultSettings()
	initial.ManifestPath = `D:\old\bubo-package.json`
	initial.SwitchOverrides = map[string]string{"--twin-model": `D:\old\twin.gguf`, "--draft-head-model": `D:\old\head.gguf`, "--temperature": "0.5"}
	a := newTestApp(&registryBackend{reg: partRegistry()}, initial, nil, "")
	manifest := `D:\new\bubo-package.json`
	twin := `D:\new\twin.gguf`
	result := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{ManifestPath: &manifest}, SwitchValues: []SwitchOverride{{Name: "--twin-model", Value: twin}}})
	if result.Error != nil || a.settings.SwitchOverrides["--twin-model"] != twin || a.settings.SwitchOverrides["--draft-head-model"] != "" || a.settings.SwitchOverrides["--temperature"] != "0.5" {
		t.Fatalf("combined model/part patch lost a new choice or kept an old one: %+v %+v", result, a.settings.SwitchOverrides)
	}
}

func TestV12MarkerListNoneClearsRatherThanPassingFilename(t *testing.T) {
	registry := &Switches{Switches: []Switch{{Name: "GUFO_THINK_MARKER_LIST", Kind: "env", GUIControl: true}}}
	for _, none := range []string{"none", " NONE "} {
		stored, err := ApplySwitchOverrides(map[string]string{"GUFO_THINK_MARKER_LIST": `C:\markers.txt`}, SwitchOverridesPatch{Set: []SwitchOverride{{Name: "GUFO_THINK_MARKER_LIST", Value: none}}}, registry)
		if err != nil || len(stored) != 0 {
			t.Fatalf("marker placeholder persisted: %+v %v", stored, err)
		}
	}
	_, env, err := switchCommand(map[string]string{"GUFO_THINK_MARKER_LIST": "none"}, registry)
	if err != nil || len(env) != 0 {
		t.Fatalf("legacy marker placeholder reached serve: %+v %v", env, err)
	}
}
