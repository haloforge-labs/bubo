//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestStoppedTightConsentUsesRecordedRegistryDefaults(t *testing.T) {
	read := func(name string) Event {
		raw, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", "v12", name))
		if err != nil {
			t.Fatal(err)
		}
		event, err := ParseEvent(raw)
		if err != nil {
			t.Fatal(err)
		}
		return event
	}
	st := read("status-main.json").Value.(*Status)
	registry := read("switches.json").Value.(*Switches)
	if st.Effective != nil {
		t.Fatal("fixture must exercise a stopped CLI without effective settings")
	}
	b := &registryBackend{fakeBackend: fakeBackend{view: Snapshot{Status: st, State: Stopped, Initialized: true, Prepared: true}}, reg: registry}
	a := newTestApp(b, DefaultSettings(), nil, "")
	yes := true
	if result := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{AllowTightSetup: &yes}}); result.Error != nil || !a.GetView().Saved.AllowTightSetup {
		t.Fatalf("recorded stopped setup could not be approved: %+v", result)
	}
}

func tightTestStatus(manifest string) *Status {
	fit, recommended := true, false
	return &Status{
		PreparedDefault: &manifest,
		Effective: map[string]json.RawMessage{
			"profile":  json.RawMessage(`{"effective":"vanilla"}`),
			"context":  json.RawMessage(`{"effective":65536}`),
			"sessions": json.RawMessage(`{"effective":2}`),
		},
		Readiness: Readiness{Runtime: true, Model: true},
		RuntimeDetail: RuntimeDetail{MemoryEstimate: &MemoryEstimate{
			Fits: &fit, Recommended: &recommended, FitScope: "conservative_startup_estimate",
		}},
	}
}

func TestTightConsentTracksCLIDefaultsAndProjection(t *testing.T) {
	saved := DefaultSettings()
	st := tightTestStatus(`C:\models\first\bubo-package.json`)
	key, ok := tightSetupFingerprint(saved, st)
	if !ok || key == "" {
		t.Fatal("CLI facts did not identify the saved setup")
	}
	saved.AllowTightSetup, saved.TightSetupFingerprint = true, key
	port := 9080
	unchanged, err := ApplyPreferencesPatch(saved, PreferencesPatch{Set: PreferenceValues{Port: &port}})
	if err != nil || !unchanged.AllowTightSetup || unchanged.TightSetupFingerprint != key {
		t.Fatalf("unrelated preference cleared consent: %+v %v", unchanged, err)
	}
	context := uint64(131072)
	changedSetup, err := ApplyPreferencesPatch(saved, PreferencesPatch{Set: PreferenceValues{Context: &context}})
	if err != nil || changedSetup.AllowTightSetup || changedSetup.TightSetupFingerprint != "" {
		t.Fatalf("changed saved setup retained consent: %+v %v", changedSetup, err)
	}
	in := actionTestInput(Stopped)
	in.Snapshot.Status = st
	in.Saved, in.SavedEstimate = saved, st.MemoryEstimate
	if view := new(ViewBuilder).Build(in); !view.Saved.AllowTightSetup || !view.Actions.Start.Allowed {
		t.Fatalf("matching consent was not active: %+v %+v", view.Saved, view.Actions.Start)
	}
	for _, change := range []struct {
		name string
		edit func(*Status)
	}{
		{"model", func(st *Status) { path := `C:\models\second\bubo-package.json`; st.PreparedDefault = &path }},
		{"context", func(st *Status) { st.Effective["context"] = json.RawMessage(`{"effective":131072}`) }},
		{"sessions", func(st *Status) { st.Effective["sessions"] = json.RawMessage(`{"effective":3}`) }},
		{"profile", func(st *Status) { st.Effective["profile"] = json.RawMessage(`{"effective":"subagents"}`) }},
		{"missing", func(st *Status) { delete(st.Effective, "profile") }},
	} {
		t.Run(change.name, func(t *testing.T) {
			changed := tightTestStatus(`C:\models\first\bubo-package.json`)
			change.edit(changed)
			in.Snapshot.Status, in.SavedEstimate = changed, changed.MemoryEstimate
			view := new(ViewBuilder).Build(in)
			if view.Saved.AllowTightSetup || view.Actions.Start.Allowed || view.Actions.Start.ReasonCode != "memory_tight" {
				t.Fatalf("stale consent remained active: %+v %+v", view.Saved, view.Actions.Start)
			}
		})
	}
}

func TestPatchTightConsentRequiresCurrentCLIFacts(t *testing.T) {
	st := tightTestStatus(`C:\models\first\bubo-package.json`)
	path := filepath.Join(t.TempDir(), "settings.json")
	b := &fakeBackend{view: Snapshot{Status: st, State: Stopped, Initialized: true, Prepared: true}, settingsPath: path}
	a := newTestApp(b, DefaultSettings(), nil, "")
	yes := true
	if result := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{AllowTightSetup: &yes}}); result.Error != nil || !a.settings.AllowTightSetup || a.settings.TightSetupFingerprint == "" {
		t.Fatalf("consent was not bound: %+v %+v", result, a.settings)
	}
	loaded, warning := LoadSettings(path)
	if warning != "" || loaded.TightSetupFingerprint != a.settings.TightSetupFingerprint || !new(ViewBuilder).Build(ViewInput{Saved: loaded, Snapshot: b.view, SavedEstimate: st.MemoryEstimate}).Saved.AllowTightSetup {
		t.Fatalf("consent did not survive an unchanged restart: %+v %s", loaded, warning)
	}
	delete(st.Effective, "profile")
	other := newTestApp(&fakeBackend{view: b.view}, DefaultSettings(), nil, "")
	if result := other.PatchPreferences(PreferencesPatch{Set: PreferenceValues{AllowTightSetup: &yes}}); result.Error == nil || other.settings.AllowTightSetup {
		t.Fatalf("consent granted without effective defaults: %+v %+v", result, other.settings)
	}
}

func TestDoesNotFitAnyContextRequiresCLIRow(t *testing.T) {
	no := false
	estimate := &MemoryEstimate{Fits: &no, Sessions: 2, ContextTokens: 65536}
	if got := buildFit(estimate); got.Class != "memory_does_not_fit" {
		t.Fatalf("missing table claimed no context fits: %+v", got)
	}
	estimate.RecommendedContext = &RecommendedContext{BySessions: []RecommendedContextRow{{Sessions: 2, Fits: false}}}
	if got := buildFit(estimate); got.Class != "memory_does_not_fit_any" {
		t.Fatalf("explicit no-fit row lost: %+v", got)
	}
}

func TestAtomicPatchCanClearKnownUnsetRegistryRow(t *testing.T) {
	registry := &Switches{Switches: []Switch{{Name: "--future", Kind: "flag", GUIControl: true}}}
	a := newTestApp(&registryBackend{reg: registry}, DefaultSettings(), nil, "")
	if result := a.PatchPreferences(PreferencesPatch{Clear: []string{"--future"}}); result.Error != nil {
		t.Fatalf("known unset clear should be idempotent: %+v", result)
	}
	if result := a.PatchPreferences(PreferencesPatch{Clear: []string{"--unknown"}}); result.Error == nil {
		t.Fatal("unknown clear was accepted")
	}
}
