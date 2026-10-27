//go:build windows

package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"bubo/gui/supervisor"
	recordings "bubo/gui/tests/scenarios/replay"
)



func TestMemoryWordingScenariosCarryTheRecordedCLIFields(t *testing.T) {
	for _, tc := range []struct {
		name, kind string
		fits       bool
		want       []supervisor.RecommendationFailure
	}{
		// Fits, but the GPU headroom and host floor rules prevent a recommendation.
		{"memory-gpu-headroom", "saved_estimate", true, []supervisor.RecommendationFailure{supervisor.RecommendationGPUHeadroom, supervisor.RecommendationHostFloor}},
		// The real grid gave no case with the host floor alone: both rules fail together.
		{"memory-default-setup", "estimate", false, []supervisor.RecommendationFailure{supervisor.RecommendationGPUHeadroom, supervisor.RecommendationHostFloor}},
		{"memory-smaller-context", "estimate", false, []supervisor.RecommendationFailure{supervisor.RecommendationGPUHeadroom, supervisor.RecommendationHostFloor}},
		{"no-fit", "saved_estimate", false, []supervisor.RecommendationFailure{supervisor.RecommendationGPUHeadroom, supervisor.RecommendationHostFloor}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			folder, err := filepath.Abs(filepath.Join("..", "scenarios", "recordings", tc.name))
			if err != nil {
				t.Fatal(err)
			}
			_, records, err := recordings.Read(folder)
			if err != nil {
				t.Fatal(err)
			}
			estimate := records[tc.kind][0].Value.(*supervisor.ModelEstimate).MemoryEstimate
			if estimate == nil || estimate.Fits == nil || *estimate.Fits != tc.fits || estimate.Recommended == nil || *estimate.Recommended || !reflect.DeepEqual(estimate.RecommendationFailures, tc.want) || estimate.HostFreeAfterSpillGiB == nil {
				t.Fatal("CLI admission or failure facts changed", estimate)
			}
			if estimate.RecommendedContext == nil || len(estimate.RecommendedContext.BySessions) == 0 {
				t.Fatal("the CLI's context rows are missing")
			}
		})
	}
}

// A scenario with no estimate record leaves `models estimate` to the fake, which
// answers from the recordings of the e21 folder.
func TestScenarioWithoutAnEstimateRecordLeavesEstimateToTheFake(t *testing.T) {
	previous := os.Args
	defer func() { os.Args = previous }()
	folder, err := filepath.Abs(filepath.Join("..", "scenarios", "recordings", "ready-busy"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_SCENARIO", folder)
	os.Args = []string{"fakecli", "models", "estimate", "--json"}
	if recordedScenario() {
		t.Fatal("a scenario with no estimate record intercepted the fake")
	}
}
