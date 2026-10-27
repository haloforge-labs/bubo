//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestPreferencesPatchRoundTripAndClearToDefault(t *testing.T) {
	initial := DefaultSettings()
	initial.Port, initial.Profile, initial.ModelPath = 19931, "future-profile", `C:\models\old.gguf`
	initial.SwitchOverrides = map[string]string{"test-env": "kept"}
	var patch PreferencesPatch
	if err := json.Unmarshal([]byte(`{"set":{"model_path":"C:\\models\\new.gguf","check_for_updates":false},"clear":[]}`), &patch); err != nil {
		t.Fatal(err)
	}
	next, err := ApplyPreferencesPatch(initial, patch)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := SaveSettings(path, next); err != nil {
		t.Fatal(err)
	}
	loaded, warning := LoadSettings(path)
	if warning != "" || loaded.Version != initial.Version || loaded.Port != initial.Port || loaded.Profile != initial.Profile || loaded.ModelPath != `C:\models\new.gguf` || loaded.UpdatesEnabled() || loaded.SwitchOverrides["test-env"] != "kept" {
		t.Fatalf("partial save lost data: %+v %s", loaded, warning)
	}
	next, err = ApplyPreferencesPatch(loaded, PreferencesPatch{Clear: []string{"port", "profile", "check_for_updates", "model_path"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveSettings(path, next); err != nil {
		t.Fatal(err)
	}
	loaded, warning = LoadSettings(path)
	if warning != "" || loaded.Port != 0 || loaded.Profile != "" || loaded.ModelPath != "" || loaded.CheckForUpdates != nil || !loaded.UpdatesEnabled() || len(startArgs(loaded)) != 0 {
		t.Fatalf("clear did not restore CLI defaults: %+v", loaded)
	}
	b, _ := os.ReadFile(path)
	var saved map[string]json.RawMessage
	_ = json.Unmarshal(b, &saved)
	for _, name := range []string{"port", "profile", "check_for_updates", "model_path"} {
		if saved[name] != nil {
			t.Fatal("cleared field persisted", name)
		}
	}
}

func TestPreferencesPatchRejectsMetadataAndConflicts(t *testing.T) {
	for _, body := range []string{`{"set":{},"clear":["version"]}`, `{"set":{"port":12},"clear":["port"]}`, `{"set":{},"clear":["switch_overrides"]}`} {
		var patch PreferencesPatch
		_ = json.Unmarshal([]byte(body), &patch)
		if _, err := ApplyPreferencesPatch(DefaultSettings(), patch); err == nil {
			t.Fatal("invalid patch accepted", body)
		}
	}
}

func writeTestRegistry(t *testing.T) *Switches {
	t.Helper()
	registry := viewFixture(t, "switches").Value.(*Switches)
	row := registry.Switches[0]
	row.Name, row.Kind, row.Echo, row.GUIControl, row.TakesValue = "--g4-value", "flag", true, true, true
	row.Allowed = json.RawMessage(`{"type":"string"}`)
	trigger := "--must-not-inject"
	row.InjectWith = &trigger
	registry.Switches = append(registry.Switches, row)
	row.Name, row.TakesValue = "--g4-bare", false
	registry.Switches = append(registry.Switches, row)
	row.Name, row.Kind, row.TakesValue = "G4_ENV_TEST", "env", true
	registry.Switches = append(registry.Switches, row)
	return registry
}

func writeContractBackend(t *testing.T, mode string) *NativeBackend {
	t.Helper()
	e := viewFixture(t, "switches")
	var err error
	e.Payload, err = json.Marshal(writeTestRegistry(t))
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseEvent(data); err != nil {
		t.Fatal("invalid test registry", err)
	}
	path := filepath.Join(t.TempDir(), "registry.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_REGISTRY", path)
	b := backendFor(t, mode)
	waitBackend(t, b, func(v Snapshot) bool { return v.RegistryLoaded })
	return b
}

func TestSwitchOverridesExcludeOwnedReadOnlyAndSecretRows(t *testing.T) {
	registry := writeTestRegistry(t)
	for _, owned := range serveArguments(Settings{}, "", 0) {
		t.Run(owned.name, func(t *testing.T) {
			// Even a future CLI row exposing an owned name cannot bypass the
			// builder's exclusion table.
			r := Switch{Name: owned.name, Kind: "flag", Echo: true, GUIControl: true}
			if switchWriteBlock(r) != SwitchSupervisorOwned {
				t.Fatal("owned row writable", r.Name)
			}
			if _, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: []SwitchOverride{{Name: owned.name, Value: "bad"}}}, registry); err == nil {
				t.Fatal("owned override accepted")
			}
		})
	}
	for _, row := range []Switch{
		{Name: "BUBO_API_KEY", Kind: "env", Echo: true, GUIControl: true},
		{Name: "future-credential", Kind: "env", Echo: true, GUIControl: true, Secret: true},
		{Name: "readonly", Kind: "env", Echo: true, GUIControl: false},
	} {
		if switchWriteBlock(row) == SwitchWritable {
			t.Fatal("unsafe row writable", row.Name)
		}
	}
	if _, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: []SwitchOverride{{Name: "unknown", Value: "x"}}}, registry); err == nil {
		t.Fatal("unknown row accepted")
	}
	if _, _, err := switchCommand(map[string]string{"--api-key": "must-never-launch"}, registry); err == nil {
		t.Fatal("persisted unsafe override launched")
	}
	cleared, err := ApplySwitchOverrides(map[string]string{"removed-from-registry": "x"}, SwitchOverridesPatch{Clear: []string{"removed-from-registry"}}, nil)
	if err != nil || len(cleared) != 0 {
		t.Fatal("cannot clear a stale override", err)
	}
}

func TestSwitchOverridesReachServeArgvAndEnvironment(t *testing.T) {
	b := writeContractBackend(t, "split")
	t.Setenv("g4_env_test", "inherited")
	t.Setenv("BUBO_FAKE_RECORD_ENV", "G4_ENV_TEST")
	s := b.settings
	s.Profile = "future-profile"
	s.SwitchOverrides = map[string]string{"--g4-value": "future-value", "--g4-bare": "true", "G4_ENV_TEST": "new=value"}
	if err := b.Save(s); err != nil {
		t.Fatal("Go validated an engine value", err)
	}
	reply(t, b.Action("start", s, ""))
	v := waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
	argv := fakeArgv(t, b.config)
	if count(argv, "--profile") != 1 || count(argv, "--api-key") != 0 || count(argv, "--g4-value") != 1 || argAfter(argv, "--g4-value") != "future-value" || count(argv, "--g4-bare") != 1 || count(argv, "--must-not-inject") != 0 {
		t.Fatal("wrong serve argv", argv)
	}
	var env map[string][]string
	data, err := os.ReadFile(filepath.Join(filepath.Dir(b.config.LogDir), "override-env.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &env); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(env["G4_ENV_TEST"], []string{"new=value"}) {
		t.Fatal("override did not replace inherited env", env)
	}
	if !reflect.DeepEqual(v.RunningSwitchOverrides, s.SwitchOverrides) {
		t.Fatal("running request history lost", v.RunningSwitchOverrides)
	}
	if value := os.Getenv("G4_ENV_TEST"); value != "inherited" {
		t.Fatal("parent env changed", value)
	}
	reply(t, b.Action("stop", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped })
}

func TestCLIOverrideRejectionRemainsTypedAndCanBeCleared(t *testing.T) {
	b := writeContractBackend(t, "reject-override")
	s := b.settings
	s.SwitchOverrides = map[string]string{"--g4-value": "reject", "G4_ENV_TEST": "keep"}
	if err := b.Save(s); err != nil {
		t.Fatal("CLI was bypassed", err)
	}
	reply(t, b.Action("start", s, ""))
	failed := waitBackend(t, b, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	if failed.LastError == nil || failed.LastError.Source != "cli" || failed.LastError.Code != "invalid_arguments" || failed.LastError.Component != "serve" || failed.LastError.Stage != "startup" || failed.LastError.Remedy != "Check the settings and model files." || failed.LastError.Switch != "--g4-value" {
		t.Fatalf("typed CLI rejection lost: %+v", failed.LastError)
	}
	var builder ViewBuilder
	view := builder.Build(ViewInput{Snapshot: failed, Registry: b.s.Registry(), Saved: s})
	if view.Failure == nil || view.Failure.Code != "invalid_arguments" || view.Failure.Switch != failed.LastError.Switch || view.Saved.SwitchOverrides[view.Failure.Switch] != "reject" {
		t.Fatal("failure or saved request lost in View")
	}
	var err error
	s.SwitchOverrides, err = ApplySwitchOverrides(s.SwitchOverrides, SwitchOverridesPatch{Clear: []string{view.Failure.Switch}}, b.s.Registry())
	if err != nil {
		t.Fatal(err)
	}
	if len(s.SwitchOverrides) != 1 || s.SwitchOverrides["G4_ENV_TEST"] != "keep" {
		t.Fatal("culprit clear changed unrelated overrides", s.SwitchOverrides)
	}
	if err := b.Save(s); err != nil {
		t.Fatal(err)
	}
	reply(t, b.Action("start", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
	if strings.Contains(strings.Join(fakeArgv(t, b.config), " "), "--g4-value") {
		t.Fatal("cleared override still sent")
	}
	reply(t, b.Action("stop", s, ""))
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Stopped })
}

func TestViewSwitchOverridePendingAndUnknown(t *testing.T) {
	registry := writeTestRegistry(t)
	st := &Status{RuntimeDetail: RuntimeDetail{EffectiveSwitches: []EffectiveSwitch{{Name: "--g4-value", Effective: json.RawMessage(`"old"`)}}}}
	in := ViewInput{Registry: registry, Saved: Settings{SwitchOverrides: map[string]string{"--g4-value": "new", "G4_ENV_TEST": "new-env", "--api-key": "never-expose"}}, Snapshot: Snapshot{Instance: &Instance{}, Status: st}}
	var builder ViewBuilder
	view := builder.Build(in)
	seen := map[string]SwitchOverrideView{}
	for _, row := range view.SwitchOverrides {
		seen[row.Name] = row
	}
	if seen["--g4-value"].RestartState != SwitchPending || seen["G4_ENV_TEST"].RestartState != SwitchUnknownUntilRestart || seen["--g4-value"].Effective.Text != "old" || *seen["--g4-value"].Saved != "new" {
		t.Fatal("incorrect pending projection", seen)
	}
	b, _ := json.Marshal(view)
	if strings.Contains(string(b), "never-expose") {
		t.Fatal("ineligible saved value leaked")
	}
	*seen["--g4-value"].Saved = "mutated"
	view.Saved.SwitchOverrides["--g4-value"] = "mutated"
	if in.Saved.SwitchOverrides["--g4-value"] != "new" {
		t.Fatal("View aliases saved input")
	}
	in.Saved.SwitchOverrides["--g4-value"] = "old"
	for _, row := range builder.Build(in).SwitchOverrides {
		if row.Name == "--g4-value" && row.RestartState != SwitchUnchanged {
			t.Fatal("equal effective value marked pending")
		}
	}
	in.Snapshot.RunningSwitchOverrides = map[string]string{"--g4-value": "old"}
	delete(in.Saved.SwitchOverrides, "--g4-value")
	for _, row := range builder.Build(in).SwitchOverrides {
		if row.Name == "--g4-value" && row.RestartState != SwitchUnknownUntilRestart {
			t.Fatal("clear invented a CLI default")
		}
	}
}

func TestSwitchEnvironmentReplacesEveryInheritedCase(t *testing.T) {
	got := switchEnvironment([]string{"A=1", "g4_env_test=old", "G4_ENV_TEST=older"}, map[string]string{"G4_ENV_TEST": "new"})
	if !reflect.DeepEqual(got, []string{"A=1", "G4_ENV_TEST=new"}) {
		t.Fatal(got)
	}
}

func TestSwitchBareFlagNeverSilentlyEnablesFalse(t *testing.T) {
	args, _, err := switchCommand(map[string]string{"--g4-bare": "false"}, writeTestRegistry(t))
	if err != nil || !reflect.DeepEqual(args, []string{"--g4-bare=false"}) {
		t.Fatal("explicit value discarded before CLI validation", args, err)
	}
}

func TestSwitchOverrideValuesAreValidatedOnlyByCLI(t *testing.T) {
	registry := writeTestRegistry(t)
	for i := range registry.Switches {
		if registry.Switches[i].Name == "--g4-value" {
			registry.Switches[i].Allowed = json.RawMessage(`{"type":"integer","minimum":1}`)
		}
	}
	got, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: []SwitchOverride{{Name: "--g4-value", Value: "ask-the-cli"}}}, registry)
	if err != nil || got["--g4-value"] != "ask-the-cli" {
		t.Fatal("Go duplicated allowed validation", got, err)
	}
	args, _, err := switchCommand(got, registry)
	if err != nil || !reflect.DeepEqual(args, []string{"--g4-value", "ask-the-cli"}) {
		t.Fatal("Go changed opaque engine value", args, err)
	}
}

func TestViewSwitchRestartMetadataAndNumericEquality(t *testing.T) {
	registry := writeTestRegistry(t)
	for _, required := range []bool{false, true} {
		t.Run(map[bool]string{false: "live", true: "restart"}[required], func(t *testing.T) {
			for i := range registry.Switches {
				if registry.Switches[i].Name == "--g4-value" {
					registry.Switches[i].RestartRequired = required
				}
			}
			in := ViewInput{Registry: registry, Saved: Settings{SwitchOverrides: map[string]string{"--g4-value": "1.0"}}, Snapshot: Snapshot{Instance: &Instance{}, Status: &Status{RuntimeDetail: RuntimeDetail{EffectiveSwitches: []EffectiveSwitch{{Name: "--g4-value", Effective: json.RawMessage(`1`)}}}}}}
			var builder ViewBuilder
			view := builder.Build(in)
			want := SwitchUnchanged
			if !required {
				want = SwitchRestartNotRequired
			}
			for _, row := range view.SwitchOverrides {
				if row.Name == "--g4-value" && row.RestartState != want {
					t.Fatal("wrong restart state", row)
				}
			}
			if len(view.Pending) != 0 {
				t.Fatal("equivalent numeric spelling marked pending")
			}
		})
	}
}
