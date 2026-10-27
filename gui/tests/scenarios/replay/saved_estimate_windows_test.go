//go:build windows

package replay

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"bubo/gui/supervisor"
)

func TestStoppedSavedEstimateKeepsNoServerAllocationNull(t *testing.T) {
	folder := filepath.Join("..", "recordings", "stopped-saved-estimate")
	manifest, records, err := Read(folder)
	if err != nil {
		t.Fatal(err)
	}
	status := records["status"][0].Value.(*supervisor.Status)
	recorded := records["saved_estimate"][0].Value.(*supervisor.ModelEstimate)
	if manifest.Source != "fixture" || status.MemoryEstimate != nil || status.MemoryEstimateReason == nil || len(status.Instances) != 0 || !status.Prepared || !status.Readiness.Runtime || !status.Readiness.Model {
		t.Fatal("fixture source or original no-server/readiness observations changed")
	}
	in, err := Inputs(folder)
	if err != nil {
		t.Fatal(err)
	}
	if in.Snapshot.Status.MemoryEstimate != nil || !reflect.DeepEqual(in.Snapshot.Status.MemoryEstimateReason, status.MemoryEstimateReason) {
		t.Fatal("saved estimate was injected into the observed status allocation")
	}
	events, err := Command(folder, "status")
	if err != nil || len(events) != 1 {
		t.Fatalf("fake status replay: %v (%d records)", err, len(events))
	}
	var replayed supervisor.Status
	if err := json.Unmarshal(events[0].Payload, &replayed); err != nil {
		t.Fatal(err)
	}
	if replayed.MemoryEstimate != nil || !reflect.DeepEqual(replayed.MemoryEstimateReason, status.MemoryEstimateReason) {
		t.Fatal("scenario-selected fake status leaked the saved setup allocation")
	}
	if in.SavedEstimate == nil || in.Saved.Context != uint64(recorded.MemoryEstimate.ContextTokens) || uint64(in.Saved.Sessions) != uint64(recorded.MemoryEstimate.Sessions) {
		t.Fatal("saved request does not match its CLI estimate recording")
	}
	// The estimate is the recorded one: its budgets and presets are the CLI's.
	if !reflect.DeepEqual(in.SavedEstimate, recorded.MemoryEstimate) {
		t.Fatal("saved estimate differs from the CLI estimate recording")
	}
	var builder supervisor.ViewBuilder
	view := builder.Build(in)
	estimate := view.SavedEstimate.MemoryEstimate
	if view.MemoryEstimate != nil || !reflect.DeepEqual(view.EstimateReason, status.MemoryEstimateReason) || estimate == nil || view.SavedEstimate.Error != nil {
		t.Fatal("View lost the separate saved estimate and no-server observation", view.SavedEstimate)
	}
	// The View fills nil lists (supervisor.Filled); compare the CLI presets after the same fill.
	if estimate.Fits == nil || !*estimate.Fits || !view.Actions.Start.Allowed || !reflect.DeepEqual(estimate.RecommendedContext, supervisor.Filled(in.SavedEstimate.RecommendedContext)) {
		t.Fatal("Start and setup did not share the saved CLI fit verdict and presets", view.Actions.Start)
	}
	presetFound := false
	for _, preset := range estimate.RecommendedContext.BySessions {
		// The CLI now returns the largest recommended context, which need not
		// equal the requested 200,000-token default. It must cover that setup.
		if preset.Sessions == estimate.Sessions && preset.ContextPerSession >= estimate.ContextTokens && preset.Fits && preset.Recommended != nil && *preset.Recommended {
			presetFound = true
		}
	}
	if !presetFound {
		t.Fatal("saved setup lacks a matching CLI context preset")
	}
	// With no status allocation to borrow, changing only the saved verdict
	// must also change Start admission and the estimate shown in the View.
	refused := *in.SavedEstimate
	fits := false
	reason := "Saved-estimate refusal fixture."
	refused.Fits, refused.FitReason = &fits, &reason
	in.SavedEstimate = &refused
	view = builder.Build(in)
	if view.Actions.Start.Allowed || view.Actions.Start.ReasonCode != "memory_does_not_fit" || view.Actions.Start.Reason != reason || view.SavedEstimate.MemoryEstimate.Fits == nil || *view.SavedEstimate.MemoryEstimate.Fits || view.MemoryEstimate != nil {
		t.Fatal("Start admission and setup diverged from the saved verdict", view.Actions.Start)
	}
	t.Log("pre-start saved estimate: original status allocation stays null; matching CLI preset and Start allowed/refused share one verdict")
}

func TestRecordedScenarioRejectsSavedEstimateReadinessOverlays(t *testing.T) {
	for name, manifest := range map[string]Manifest{
		"saved-estimate": {Source: "recorded", Records: map[string]string{"saved_estimate": "estimate.json"}},
		"saved-model":    {Source: "recorded", SavedModelID: "qwen38-27b-ud-q4-k-xl"},
	} {
		t.Run(name, func(t *testing.T) {
			folder := t.TempDir()
			raw, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(folder, "scenario.json"), raw, 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := Read(folder); err == nil {
				t.Fatal("synthetic saved-setup overlay accepted as an authentic recording")
			}
		})
	}
}
