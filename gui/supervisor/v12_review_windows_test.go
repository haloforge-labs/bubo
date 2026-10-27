//go:build windows

package supervisor

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestV12ReadyRestartIgnoresOccupiedMemorySample(t *testing.T) {
	for _, fits := range []bool{true, false} {
		in := actionTestInput(Ready)
		no := false
		in.SavedEstimate = &MemoryEstimate{Fits: &fits, Recommended: &no, FitScope: "conservative_startup_estimate"}
		if got := lifecycleActions(in).Restart; !got.Allowed || got.ReasonCode != "fit_unchecked" {
			t.Fatalf("running allocation blocked restart: %+v", got)
		}
	}
}

func TestV12TightFingerprintPrefersEffectiveDefaults(t *testing.T) {
	saved := DefaultSettings()
	st := tightTestStatus(`C:\models\bubo-package.json`)
	registry := &Switches{Switches: []Switch{
		{Name: "--profile", DefaultText: "subagents", Allowed: json.RawMessage(`{"type":"enum","values":["subagents","vanilla"]}`)},
		{Name: "--context", DefaultText: "262144"},
		{Name: "--sessions", DefaultText: "1"},
	}}
	want, ok := tightSetupFingerprint(saved, st)
	got, known := tightSetupFingerprint(saved, st, registry)
	if !ok || !known || got != want {
		t.Fatalf("registry displaced effective setup: %s != %s", got, want)
	}
}

func TestV12MemoryFailureSaysWhichMemory(t *testing.T) {
	for _, fits := range []bool{true, false} {
		for _, tc := range []struct {
			failure  RecommendationFailure
			sentence string
		}{
			{RecommendationHostFloor, "System memory is low; close other apps."},
			{RecommendationGPUHeadroom, "GPU memory is low."},
		} {
			no := false
			estimate := &MemoryEstimate{Fits: &fits, Recommended: &no, FitScope: "conservative_startup_estimate", RecommendationFailures: []RecommendationFailure{tc.failure}}
			got := buildFit(estimate)
			if !strings.Contains(got.Sentence, tc.sentence) {
				t.Fatalf("memory-specific guidance missing: %+v", got)
			}
		}
	}
}
