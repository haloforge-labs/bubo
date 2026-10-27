//go:build windows

package supervisor

import (
	"encoding/json"
	"testing"
	"time"
)

func TestF5aPreparationHidesRetainedAddressesAndProjectsFileScope(t *testing.T) {
	in := actionTestInput(Preparing)
	in.Snapshot.Status.Endpoints = &Endpoints{OpenAIBaseURL: "https://example.test/stale"}
	in.Snapshot.Status.Connection = &Connection{Endpoint: "https://example.test/stale"}
	in.Snapshot.Progress = &Progress{Component: "runtime", FileID: "second"}
	in.Snapshot.PlanParts = []Plan{{Component: "runtime", Parts: 1, Paths: map[string]json.RawMessage{}, Sources: []Source{{FileID: "first"}, {FileID: "second"}}}}
	v := (&ViewBuilder{}).Build(in)
	if v.Endpoints != nil || v.Connection != nil {
		t.Fatal("preparing advertised old serving addresses")
	}
	for _, state := range []State{Stopped, Starting, Failed, NotPrepared, Stopping} {
		in.Snapshot.State = state
		if stopped := (&ViewBuilder{}).Build(in); stopped.Endpoints != nil || stopped.Connection != nil {
			t.Fatalf("%s advertised stale serving addresses", state)
		}
	}
	in.Snapshot.State = Preparing
	if v.PreparationFile == nil || *v.PreparationFile != (FilePosition{Index: 2, Total: 2}) {
		t.Fatalf("runtime plan scope missing: %+v", v.PreparationFile)
	}
	in.Snapshot.Progress.FileID = "unknown"
	if (&ViewBuilder{}).Build(in).PreparationFile != nil {
		t.Fatal("invented unknown file position")
	}
}

func TestF5aRejectedOverrideKeepsItsCauseUntilEdited(t *testing.T) {
	in := actionTestInput(Failed)
	in.Snapshot.LastError = &Failure{Code: "invalid_arguments", Switch: "--fixture-setting"}
	in.Snapshot.RunningSwitchOverrides = map[string]string{"--fixture-setting": "rejected"}
	in.Saved.SwitchOverrides = map[string]string{"--fixture-setting": "rejected"}
	in.SavedEstimateError = &CLIError{Code: "estimate_needs_restart", Remedy: "Fixture override is not estimated."}
	if got := lifecycleActions(in).Start; got.Allowed || got.ReasonCode != "setting_rejected" {
		t.Fatalf("override must retain its own cause: %+v", got)
	}
	in.Saved.SwitchOverrides["--fixture-setting"] = "revised"
	if got := lifecycleActions(in).Start; !got.Allowed {
		t.Fatalf("editing the rejected value must permit a new attempt: %+v", got)
	}
	delete(in.Saved.SwitchOverrides, "--fixture-setting")
	if got := lifecycleActions(in).Start; !got.Allowed {
		t.Fatalf("resetting the rejected setting must permit a new attempt: %+v", got)
	}
}

func TestF5aCatalogueFilePosition(t *testing.T) {
	catalogue := []CatalogueEntry{{ID: "fixture", Files: []ModelFile{{Name: "one.bin", URL: "https://example.test/one"}, {Name: "two.bin", URL: "https://example.test/two"}}}}
	plan := &ModelPlan{ID: "fixture", Files: []ModelPlanFile{{FileID: "second", Path: `D:\fixture\two.bin`, URL: "https://example.test/two"}}}
	for _, id := range []string{"fixture", ""} {
		plan.ID = id
		got := catalogueFilePosition("second", plan, catalogue)
		if got == nil || got.Index != 2 || got.Total != 2 {
			t.Fatalf("must use full catalogue, not one-file plan: %+v", got)
		}
	}
	for _, file := range []string{"", "missing"} {
		if catalogueFilePosition(file, plan, catalogue) != nil {
			t.Fatal("invented a total for unknown file")
		}
	}
	duplicate := append(append([]CatalogueEntry(nil), catalogue...), CatalogueEntry{ID: "another-fixture", Files: catalogue[0].Files})
	if catalogueFilePosition("second", plan, duplicate) != nil {
		t.Fatal("a shared URL must not identify one of several catalogues")
	}
	plan.Files[0].URL = "https://example.test/unknown"
	if catalogueFilePosition("second", plan, catalogue) != nil || catalogueFilePosition("second", nil, catalogue) != nil {
		t.Fatal("unknown catalogue identity must remain current-file only")
	}
}

func TestF5aCheckAgainExpiresSavedEstimate(t *testing.T) {
	m := newModelsApp(t, "")
	m.a.estimate.at = time.Now()
	if result := m.a.RefreshModels(); result.Error != nil {
		t.Fatal(result.Error)
	}
	if !m.a.estimate.at.IsZero() {
		t.Fatal("Check again kept the old memory-check cache")
	}
	a := newTestApp(&fakeBackend{}, DefaultSettings(), nil, "")
	if result := a.RefreshModels(); result.Error == nil || result.Error.Code != "internal_error" {
		t.Fatal("missing model service reported success")
	}
}
