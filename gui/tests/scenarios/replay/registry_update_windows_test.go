//go:build windows

package replay

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"bubo/gui/supervisor"
)

func TestScenariosWithoutUpdaterRemainUnchecked(t *testing.T) {
	manifests, err := filepath.Glob(filepath.Join("..", "recordings", "*", "scenario.json"))
	if err != nil || len(manifests) == 0 {
		t.Fatalf("scenario manifests: %v (%d found)", err, len(manifests))
	}
	count := 0
	for _, path := range manifests {
		folder := filepath.Dir(path)
		manifest, _, err := Read(folder)
		if err != nil {
			t.Fatal(err)
		}
		if manifest.Update != nil {
			continue
		}
		count++
		t.Run(filepath.Base(folder), func(t *testing.T) {
			in, err := Inputs(folder)
			if err != nil {
				t.Fatal(err)
			}
			if in.Update.State != supervisor.UpdateNone || in.Update.Error != nil {
				t.Fatalf("no updater attached, but replay input says %+v", in.Update)
			}
			view, err := Build(folder)
			if err != nil {
				t.Fatal(err)
			}
			if view.Update.State != supervisor.UpdateNone || view.Update.Error != nil || view.Update.AvailableVersion != "" || view.Update.Progress != nil {
				t.Fatalf("no check performed, but View says %+v", view.Update)
			}
		})
	}
	if count == 0 {
		t.Fatal("no scenarios exercise an unattached updater")
	}
	t.Logf("unchecked updater scenarios: %d passed", count)
}

func TestCurrentRecordingPreservesRegistryHintsAndMetadata(t *testing.T) {
	folder := filepath.Join("..", "recordings", "v12")
	raw, err := os.ReadFile(filepath.Join(folder, "switches.json"))
	if err != nil {
		t.Fatal(err)
	}
	metadataRaw, err := os.ReadFile(filepath.Join(folder, "metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Revision   string `json:"revision"`
		Executable string `json:"executable_sha256"`
		Commands   []struct {
			File     string `json:"file"`
			ExitCode int    `json:"exit_code"`
			SHA256   string `json:"stdout_sha256"`
		} `json:"commands"`
	}
	if err := json.Unmarshal(metadataRaw, &metadata); err != nil {
		t.Fatal(err)
	}
	recorded := ""
	for _, c := range metadata.Commands {
		if c.File == "switches.json" && c.ExitCode == 0 {
			recorded = c.SHA256
		}
	}
	digest := sha256.Sum256(raw)
	if recorded == "" || hex.EncodeToString(digest[:]) != recorded || metadata.Executable == "" {
		t.Fatal("recording bytes no longer match capture provenance")
	}
	event, err := supervisor.ParseEvent(raw)
	if err != nil {
		t.Fatal(err)
	}
	registry, ok := event.Value.(*supervisor.Switches)
	if !ok {
		t.Fatal("capture is not a typed switches record")
	}
	golden, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "tests", "cli", "contract", "switches.golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(readJSONValue(t, event.Payload), readJSONValue(t, golden)) {
		t.Fatal("recorded CLI registry differs from the canonical B5 payload")
	}
	var builder supervisor.ViewBuilder
	view := builder.Build(supervisor.ViewInput{Registry: registry, Update: supervisor.Update{State: supervisor.UpdateNone}})
	if len(view.Switches) != len(registry.Switches) || len(view.Switches) == 0 {
		t.Fatal("CLI registry rows were dropped from the View")
	}
	suggested, secrets, excluded := 0, 0, 0
	for i, row := range registry.Switches {
		projected := view.Switches[i]
		if projected.Name != row.Name || projected.Secret != row.Secret || projected.GUIControl != row.GUIControl {
			t.Fatalf("registry metadata changed for %s", row.Name)
		}
		if row.Secret {
			secrets++
		}
		if !row.GUIControl {
			excluded++
		}
		var allowed map[string]json.RawMessage
		if err := json.Unmarshal(row.Allowed, &allowed); err != nil {
			t.Fatal(err)
		}
		if _, ok := allowed["suggested"]; ok {
			suggested++
			t.Run(row.Name, func(t *testing.T) {
				// Compare the full original allowed object: suggestions must
				// stay nested and must not acquire top-level limit fields.
				assertPreservedCLIValue(t, projected.Allowed, readJSONValue(t, row.Allowed))
			})
		}
	}
	// The totals come from the golden itself, parsed apart from the View.
	var counted struct {
		Switches []struct {
			Secret     bool `json:"secret"`
			GUIControl bool `json:"gui_control"`
			Allowed    struct {
				Suggested json.RawMessage `json:"suggested"`
			} `json:"allowed"`
		} `json:"switches"`
	}
	if err := json.Unmarshal(golden, &counted); err != nil {
		t.Fatal(err)
	}
	wantSuggested, wantSecrets, wantExcluded := 0, 0, 0
	for _, row := range counted.Switches {
		if len(row.Allowed.Suggested) != 0 {
			wantSuggested++
		}
		if row.Secret {
			wantSecrets++
		}
		if !row.GUIControl {
			wantExcluded++
		}
	}
	if suggested != wantSuggested || suggested == 0 || secrets != wantSecrets || excluded != wantExcluded {
		t.Fatalf("registry totals changed: suggested=%d secret=%d excluded=%d", suggested, secrets, excluded)
	}
	t.Logf("current registry: %d rows, %d preserved suggestions, %d secrets, %d excluded controls; capture hash and canonical payload match", len(view.Switches), suggested, secrets, excluded)
}

func TestScenarioRegistryReplayUsesCurrentCapture(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "recordings", "v12", "switches.json"))
	if err != nil {
		t.Fatal(err)
	}
	capture, err := supervisor.ParseEvent(raw)
	if err != nil {
		t.Fatal(err)
	}
	manifests, err := filepath.Glob(filepath.Join("..", "recordings", "*", "scenario.json"))
	if err != nil || len(manifests) == 0 {
		t.Fatalf("scenario manifests: %v (%d found)", err, len(manifests))
	}
	replayed := 0
	for _, path := range manifests {
		folder := filepath.Dir(path)
		t.Run(filepath.Base(folder), func(t *testing.T) {
			m, _, err := Read(folder)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := m.Records["switches"]; !ok {
				t.Skip("scenario has no switches recording")
			}
			events, err := Command(folder, "switches")
			if err != nil {
				t.Fatal(err)
			}
			if len(events) != 1 || events[0].Type != capture.Type || !bytes.Equal(events[0].Payload, capture.Payload) || events[0].OperationID != capture.OperationID || events[0].Time != capture.Time {
				t.Fatal("scenario-selected fake read differs from the current unmodified CLI capture")
			}
			replayed++
		})
	}
	t.Logf("current registry replay: %d scenarios passed", replayed)
}

func readJSONValue(t *testing.T, raw []byte) any {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value
}

func assertPreservedCLIValue(t *testing.T, got supervisor.Value, want any) {
	t.Helper()
	switch value := want.(type) {
	case map[string]any:
		if got.Kind != "object" || len(got.Properties) != len(value) {
			t.Fatalf("object changed: %+v, want %#v", got, value)
		}
		for key, child := range value {
			assertPreservedCLIValue(t, got.Properties[key], child)
		}
	case []any:
		if got.Kind != "array" || len(got.Items) != len(value) {
			t.Fatalf("array changed: %+v, want %#v", got, value)
		}
		for i, child := range value {
			assertPreservedCLIValue(t, got.Items[i], child)
		}
	case json.Number:
		if got.Kind != "number" || got.Text != value.String() {
			t.Fatalf("number changed: %+v, want %s", got, value)
		}
	case string:
		if got.Kind != "string" || got.Text != value {
			t.Fatalf("string changed: %+v, want %q", got, value)
		}
	case bool:
		if got.Kind != "boolean" || got.Boolean == nil || *got.Boolean != value {
			t.Fatalf("boolean changed: %+v, want %v", got, value)
		}
	case nil:
		if got.Kind != "null" {
			t.Fatalf("null changed: %+v", got)
		}
	default:
		t.Fatalf("unsupported decoded test value %T", want)
	}
}
