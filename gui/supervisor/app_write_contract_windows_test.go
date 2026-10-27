//go:build windows

package supervisor

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

func writePreferencePatch(t *testing.T, text string) PreferencesPatch {
	t.Helper()
	var patch PreferencesPatch
	if err := json.Unmarshal([]byte(text), &patch); err != nil {
		t.Fatal(err)
	}
	return patch
}

// The native behavior tests send preference fields through the public patch
// binding. Settings metadata, folders and switch requests are deliberately absent.
func patchTestPreferences(a *AppService, settings Settings) error {
	data, err := json.Marshal(settings)
	if err != nil {
		return err
	}
	var values PreferenceValues
	if err = json.Unmarshal(data, &values); err != nil {
		return err
	}
	return actionTestError(a.PatchPreferences(PreferencesPatch{Set: values}))
}

func actionTestError(result ActionResult) error {
	if result.Error != nil {
		return errors.New(result.Error.Remedy)
	}
	return nil
}

func TestPatchPreferencesPreservesUnchangedFieldsAndSwitchOverrides(t *testing.T) {
	b := &fakeBackend{}
	initial := DefaultSettings()
	initial.Profile, initial.Context = "future-profile", 991
	initial.SwitchOverrides = map[string]string{"future-env": "keep"}
	a := newTestApp(b, initial, nil, "")
	if result := a.PatchPreferences(writePreferencePatch(t, `{"set":{"model_path":"C:\\models\\new.gguf","start_server_with_gui":true},"clear":[]}`)); result.Error != nil {
		t.Fatal(result.Error)
	}
	if b.settings.Version != initial.Version || b.settings.Profile != initial.Profile || b.settings.Context != initial.Context || !b.settings.StartServerWithGUI || b.settings.ModelPath != `C:\models\new.gguf` || b.settings.SwitchOverrides["future-env"] != "keep" {
		t.Fatalf("partial preferences lost fields: %+v", b.settings)
	}
	if result := a.PatchPreferences(PreferencesPatch{Clear: []string{"profile", "context"}}); result.Error != nil {
		t.Fatal(result.Error)
	}
	if b.settings.Profile != "" || b.settings.Context != 0 || b.settings.ModelPath == "" {
		t.Fatal("clear did not compose")
	}
	if result := a.PatchPreferences(writePreferencePatch(t, `{"set":{"start_gui_at_login":true}}`)); result.Error != nil {
		t.Fatal(result.Error)
	}
	if b.settings.SwitchOverrides["future-env"] != "keep" {
		t.Fatal("preference patch undid switch overrides")
	}
}

func TestPatchPreferencesUsesTrayAndLANChecks(t *testing.T) {
	store := &memStore{m: map[string]string{}}
	b := &fakeBackend{}
	a := lanApp(b, store)
	a.trayAvailable = false
	if result := a.PatchPreferences(writePreferencePatch(t, `{"set":{"close_to_tray":true},"clear":[]}`)); result.Error != nil || a.settings.CloseToTray {
		t.Fatal("patch bypassed tray check", result)
	}
	if result := a.PatchPreferences(writePreferencePatch(t, `{"set":{"lan_access":true},"clear":[]}`)); result.Error != nil {
		t.Fatal(result.Error)
	}
	if !a.settings.LanAccess || len(store.key()) != 64 || !a.GetView().Key.Present {
		t.Fatal("patch bypassed key creation")
	}
	view, _ := json.Marshal(a.GetView())
	if strings.Contains(string(view), store.key()) {
		t.Fatal("key leaked through View")
	}
	if result := a.PatchPreferences(PreferencesPatch{Clear: []string{"lan_access"}}); result.Error != nil || a.settings.LanAccess {
		t.Fatal("LAN clear failed", result)
	}
	store.failWrite = true
	store.m = map[string]string{}
	a.keys = NewKeyManager(store)
	if result := a.PatchPreferences(writePreferencePatch(t, `{"set":{"lan_access":true},"clear":[]}`)); result.Error == nil || a.settings.LanAccess || b.settings.LanAccess {
		t.Fatal("key failure reported success", result)
	}
}

func TestConcurrentPreferencePatchesCompose(t *testing.T) {
	a := newTestApp(&fakeBackend{}, DefaultSettings(), nil, "")
	first := writePreferencePatch(t, `{"set":{"profile":"future-profile"},"clear":[]}`)
	second := writePreferencePatch(t, `{"set":{"context":899},"clear":[]}`)
	var wg sync.WaitGroup
	for _, patch := range []PreferencesPatch{first, second} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if result := a.PatchPreferences(patch); result.Error != nil {
				t.Error(result.Error)
			}
		}()
	}
	wg.Wait()
	if a.settings.Profile != "future-profile" || a.settings.Context != 899 {
		t.Fatal("concurrent patches lost a field")
	}
}

type overrideBackend struct {
	*fakeBackend
	registry *Switches
}

func (b *overrideBackend) Registry() *Switches { return b.registry }

func TestSetSwitchOverridesBindingAndTypedErrors(t *testing.T) {
	b := &overrideBackend{fakeBackend: &fakeBackend{}, registry: &Switches{Switches: []Switch{{Name: "FUTURE_ENV", Kind: "env", Echo: true, GUIControl: true}}}}
	a := newTestApp(b, DefaultSettings(), nil, "")
	patch := SwitchOverridesPatch{Set: []SwitchOverride{{Name: "FUTURE_ENV", Value: "opaque-cli-value"}}}
	if result := a.SetSwitchOverrides(patch); result.Error != nil {
		t.Fatal(result.Error)
	}
	if b.settings.SwitchOverrides["FUTURE_ENV"] != "opaque-cli-value" || a.GetView().Saved.SwitchOverrides["FUTURE_ENV"] != "opaque-cli-value" {
		t.Fatal("saved switch missing from View")
	}
	if result := a.SetSwitchOverrides(SwitchOverridesPatch{Set: []SwitchOverride{{Name: "--profile", Value: "bad"}}}); result.Error == nil || result.Error.Code != "setting_removed" {
		t.Fatal("unsafe write accepted", result)
	}
	b.saveErr = &CommandError{Event: Event{Value: &CLIError{Code: "invalid_arguments", Component: "serve", Stage: "startup", Remedy: "CLI remedy", Switch: "FUTURE_ENV"}}}
	if result := a.SetSwitchOverrides(patch); result.Error == nil || result.Error.Code != "invalid_arguments" || result.Error.Remedy != "CLI remedy" || result.Error.Switch != "FUTURE_ENV" {
		t.Fatal("typed error lost", result)
	}
	b.saveErr = nil
	if result := a.SetSwitchOverrides(SwitchOverridesPatch{Clear: []string{"FUTURE_ENV"}}); result.Error != nil || len(a.settings.SwitchOverrides) != 0 {
		t.Fatal("clear switch failed", result)
	}
}
