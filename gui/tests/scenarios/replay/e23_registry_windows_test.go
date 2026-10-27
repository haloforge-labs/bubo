//go:build windows

package replay

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"bubo/gui/supervisor"
)

// E23: a scenario standing in for the release GUI must project the candidate
// CLI registry, including changed defaults and non-control rows, unchanged.
func TestE23ScenarioRegistryFollowsCandidateCLI(t *testing.T) {
	canonical := filepath.Join("..", "..", "..", "..", "tests", "cli", "contract", "valid", "switches.json")
	raw, err := os.ReadFile(canonical)
	if err != nil {
		t.Fatal(err)
	}
	event, err := supervisor.ParseEvent(raw)
	if err != nil {
		t.Fatal(err)
	}
	registry := event.Value.(*supervisor.Switches)
	var builder supervisor.ViewBuilder
	want := builder.Build(supervisor.ViewInput{Registry: registry}).Switches

	folder := filepath.Join("..", "recordings", "ready-idle")
	got, err := Build(folder)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Switches, want) {
		t.Fatalf("GUI registry differs from candidate CLI: %d scenario rows, %d candidate rows", len(got.Switches), len(want))
	}
	t.Logf("candidate CLI registry: %d rows reached the GUI unchanged", len(want))
}
