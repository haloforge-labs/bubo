//go:build windows

package supervisor

import (
	"testing"
)

func TestF4RegistryScalarsUseCanonicalPreferenceWriter(t *testing.T) {
	for _, lan := range []bool{false, true} {
		patch, err := normalizePreferencePatch(PreferencesPatch{SwitchValues: []SwitchOverride{
			{Name: "--host", Value: requestedHost(Settings{LanAccess: lan})},
			{Name: "--port", Value: "19001"},
			{Name: "--profile", Value: "fixture-profile"},
		}})
		if err != nil {
			t.Fatal(err)
		}
		next, err := ApplyPreferencesPatch(Settings{Version: 1, CloseToTray: true}, patch)
		if err != nil || next.LanAccess != lan || next.Port != 19001 || next.Profile != "fixture-profile" || !next.CloseToTray || len(next.SwitchOverrides) != 0 {
			t.Fatal("registry scalars did not preserve the single preference writer", next, err)
		}
	}
	for _, value := range []SwitchOverride{
		{Name: "--port", Value: "7.5"}, {Name: "--port", Value: "null"},
		{Name: "--port", Value: ""}, {Name: "--port", Value: "not-a-number"},
		{Name: "--host", Value: "fixture.invalid"},
	} {
		if _, err := normalizePreferencePatch(PreferencesPatch{SwitchValues: []SwitchOverride{value}}); err == nil {
			t.Fatal("unsupported app transport reported success", value)
		}
	}
	// Registry rows now pass through normalization for validation by the same
	// atomic writer. An unregistered row must still fail before anything saves.
	patch, err := normalizePreferencePatch(PreferencesPatch{SwitchValues: []SwitchOverride{{Name: "--unknown", Value: "1"}}})
	if err != nil || len(patch.SwitchValues) != 1 {
		t.Fatal("registry value did not reach registry validation", patch, err)
	}
	if _, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: patch.SwitchValues}, &Switches{}); err == nil {
		t.Fatal("unregistered switch accepted")
	}
	port := 19002
	if _, err := normalizePreferencePatch(PreferencesPatch{Set: PreferenceValues{Port: &port}, SwitchValues: []SwitchOverride{{Name: "--port", Value: "19001"}}}); err == nil {
		t.Fatal("duplicate preference writers accepted")
	}
}

// The View no longer decides "not loadable" from the catalogue's prose.
func TestF4ViewRetainsTheDefaultAndNeverReadsLoaderProse(t *testing.T) {
	in := actionTestInput(Stopped)
	path, noLoader, missing := "D:/fixture/default.json", "This tree has no loader for this architecture.", "The model file is not on this computer."
	in.Snapshot.Status.PreparedDefault = &path
	in.Models = &ModelsState{Catalogue: []CatalogueEntry{
		{ID: "unloadable", Estimate: ModelEstimateInfo{State: "none_yet", Reason: &noLoader}},
		{ID: "downloadable", Estimate: ModelEstimateInfo{State: "none_yet", Reason: &missing}},
	}}
	v := new(ViewBuilder).Build(in)
	if v.PreparedDefault == nil || *v.PreparedDefault != path {
		t.Fatal("View lost the CLI default", v.PreparedDefault)
	}
	in.Snapshot.Status.Readiness.Model = false
	if action := new(ViewBuilder).Build(in).Actions.Start; action.Allowed || action.ReasonCode != "model_missing" || action.Reason == "" {
		t.Fatal("missing model must use the supervisor readiness reason", action)
	}
}

func TestF4PreferenceSwitchRouteKeepsAppPolicies(t *testing.T) {
	store := &memStore{m: map[string]string{}}
	a := lanApp(&fakeBackend{}, store)
	patch := PreferencesPatch{SwitchValues: []SwitchOverride{{Name: "--host", Value: requestedHost(Settings{LanAccess: true})}}}
	if result := a.PatchPreferences(patch); result.Error != nil || !a.settings.LanAccess || len(store.key()) != 64 {
		t.Fatal("registry route bypassed canonical LAN/key policy", result)
	}
	patch.SwitchValues[0].Value = requestedHost(Settings{})
	if result := a.PatchPreferences(patch); result.Error != nil || a.settings.LanAccess {
		t.Fatal("registry route did not preserve explicit local-only false", result)
	}
	backend := &registryBackend{reg: &Switches{Switches: []Switch{{Name: "--port", Class: "REMOVED"}}}}
	a = newTestApp(backend, DefaultSettings(), nil, "")
	result := a.PatchPreferences(PreferencesPatch{SwitchValues: []SwitchOverride{{Name: "--port", Value: "19001"}}})
	if result.Error == nil || result.Error.Code != "setting_removed" || a.settings.Port != 0 {
		t.Fatal("registry route bypassed the CLI removal verdict", result)
	}
}
