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

func TestNPURuntimeContractAndViewPreserveEveryField(t *testing.T) {
	for _, name := range []string{"status-npu-on", "status-npu-off", "status-npu-tripped", "status-npu-faulted", "status-npu-unavailable", "ready-npu"} {
		t.Run(name, func(t *testing.T) {
			e := viewFixture(t, name)
			var runtime *NPUPrefillRuntime
			switch v := e.Value.(type) {
			case *Status:
				runtime = v.NPUPrefillRuntime
			case *Lifecycle:
				runtime = v.NPUPrefillRuntime
			default:
				t.Fatal("fixture has no runtime payload")
			}
			if runtime == nil {
				t.Fatal("NPU runtime lost")
			}
			var payload map[string]json.RawMessage
			if err := json.Unmarshal(e.Payload, &payload); err != nil {
				t.Fatal(err)
			}
			var want, got any
			if err := json.Unmarshal(payload["npu_prefill_runtime"], &want); err != nil {
				t.Fatal(err)
			}
			var builder ViewBuilder
			in := ViewInput{Snapshot: Snapshot{Status: &Status{RuntimeDetail: RuntimeDetail{NPUPrefillRuntime: runtime}}}}
			view := builder.Build(in)
			wire, err := json.Marshal(view.NPUPrefillRuntime)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(wire, &got); err != nil || !reflect.DeepEqual(got, want) {
				t.Fatal("CLI NPU fields lost or changed", string(wire), err)
			}
			// Snapshots and returned Views must own nested nullable values.
			owned := copySnapshot(in.Snapshot)
			view.NPUPrefillRuntime.State = "changed"
			view.NPUPrefillRuntime.Trips++
			if view.NPUPrefillRuntime.Reason != nil {
				*view.NPUPrefillRuntime.Reason = "changed"
			}
			if view.NPUPrefillRuntime.ChipPower.WMean != nil {
				*view.NPUPrefillRuntime.ChipPower.WMean = -1
			}
			if view.NPUPrefillRuntime.Memory != nil {
				view.NPUPrefillRuntime.Memory.MinAvailableMiB = -1
			}
			if view.NPUPrefillRuntime.Scheme != nil {
				view.NPUPrefillRuntime.Scheme.GUID = "changed"
			}
			fresh := builder.Build(in)
			if !reflect.DeepEqual(fresh.NPUPrefillRuntime, runtime) || !reflect.DeepEqual(owned.Status.NPUPrefillRuntime, runtime) {
				t.Fatal("View mutated CLI data")
			}
			owned.Status.NPUPrefillRuntime.State = "changed"
			if runtime.State == "changed" {
				t.Fatal("Snapshot aliases NPU runtime")
			}
		})
	}
	var builder ViewBuilder
	v := builder.Build(ViewInput{Snapshot: Snapshot{Status: viewFixture(t, "status-observed").Value.(*Status)}})
	if v.NPUPrefillRuntime != nil {
		t.Fatal("absent NPU observation invented")
	}
}

// The current candidate registry: the same rows as the canonical golden.
func TestCurrentRegistrySecretsAreExcludedFromOverrides(t *testing.T) {
	checkRealRegistrySecrets(t, "v12", true)
}

// The registry of the final package (engine 80a38da0), which the scenarios use.
func TestRealF80RegistrySecretsAreExcludedFromOverrides(t *testing.T) {
	checkRealRegistrySecrets(t, "f80", false)
}

func checkRealRegistrySecrets(t *testing.T, recording string, sameNamesAsCanonical bool) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", recording, "switches.json"))
	if err != nil {
		t.Fatal(err)
	}
	e, err := ParseEvent(b)
	if err != nil {
		t.Fatal(err)
	}
	registry := e.Value.(*Switches)
	canonical := viewFixture(t, "switches").Value.(*Switches)
	// The recorded CLI output is independent evidence of the registry's secret
	// contract; historical recordings remain unchanged.
	actualSecrets := map[string]bool{}
	for _, row := range registry.Switches {
		actualSecrets[row.Name] = row.Secret
	}
	if sameNamesAsCanonical && len(actualSecrets) != len(canonical.Switches) {
		t.Fatal("recorded registry has different switch names")
	}
	for _, row := range canonical.Switches {
		if secret, found := actualSecrets[row.Name]; sameNamesAsCanonical && (!found || secret != row.Secret) || found && secret != row.Secret {
			t.Fatal("recorded secret marker differs from canonical CLI listing", row.Name)
		}
	}
	var secrets int
	for _, row := range registry.Switches {
		if !row.Secret {
			continue
		}
		secrets++
		t.Run(row.Name, func(t *testing.T) {
			if switchWriteBlock(row) == SwitchWritable {
				t.Fatal("real secret row writable")
			}
			// Isolate the registry marker from ownership and control metadata.
			renamed := row
			renamed.Name, renamed.GUIControl = "future-secret", true
			if switchWriteBlock(renamed) != SwitchSecret {
				t.Fatal("exclusion did not follow the secret marker")
			}
			if _, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: []SwitchOverride{{Name: row.Name, Value: "G6-secret-sentinel"}}}, registry); err == nil {
				t.Fatal("secret save accepted")
			}
			if _, _, err := switchCommand(map[string]string{row.Name: "G6-secret-sentinel"}, registry); err == nil {
				t.Fatal("secret launched")
			}
			var builder ViewBuilder
			view := builder.Build(ViewInput{Registry: registry, Saved: Settings{SwitchOverrides: map[string]string{row.Name: "G6-secret-sentinel"}}})
			wire, _ := json.Marshal(view)
			if strings.Contains(string(wire), "G6-secret-sentinel") {
				t.Fatal("saved secret leaked")
			}
		})
	}
	if secrets == 0 {
		t.Fatal("real CLI supplied no secret metadata")
	}
	t.Logf("real CLI registry: %d rows, %d secret exclusions", len(registry.Switches), secrets)
}

func TestRejectedSwitchSurvivesStatusAndSavedFailure(t *testing.T) {
	e := viewFixture(t, "status")
	st := e.Value.(*Status)
	st.LastError.Switch = "--think-marker-bias"
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(e.Payload, &payload); err != nil {
		t.Fatal(err)
	}
	var err error
	payload["last_error"], err = json.Marshal(st.LastError)
	if err != nil {
		t.Fatal(err)
	}
	e.Payload, err = json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseEvent(wire)
	if err != nil || parsed.Value.(*Status).LastError.Switch != st.LastError.Switch {
		t.Fatal("status failure lost culprit", err)
	}
	saved := DefaultSettings()
	saved.LastError = &Failure{Code: "invalid_arguments", Source: "cli", Switch: st.LastError.Switch, Remedy: "sensitive value"}
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := SaveSettings(path, saved); err != nil {
		t.Fatal(err)
	}
	loaded, warning := LoadSettings(path)
	if warning != "" || loaded.LastError == nil || loaded.LastError.Switch != saved.LastError.Switch || loaded.LastError.Remedy == saved.LastError.Remedy {
		t.Fatal("saved failure lost culprit or retained diagnostic data", loaded.LastError, warning)
	}
}
