//go:build windows

package supervisor

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMemoryFailureProjectionOwnsOptionalFields(t *testing.T) {
	var estimate MemoryEstimate
	if err := json.Unmarshal([]byte(`{"fits":true,"recommended":false,"recommendation_failures":["host_floor"],"host_free_after_spill_gib":1.8,"recommended_context":{"by_sessions":[{"recommendation_failures":["gpu_headroom"],"host_free_after_spill_gib":0}]}}`), &estimate); err != nil {
		t.Fatal(err)
	}
	projected := projectMemoryEstimate(estimate)
	encoded, err := json.Marshal(projected)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip MemoryEstimate
	if err = json.Unmarshal(encoded, &roundTrip); err != nil || !reflect.DeepEqual(roundTrip.RecommendationFailures, estimate.RecommendationFailures) || !reflect.DeepEqual(roundTrip.HostFreeAfterSpillGiB, estimate.HostFreeAfterSpillGiB) || !reflect.DeepEqual(roundTrip.RecommendedContext, estimate.RecommendedContext) || !reflect.DeepEqual(roundTrip.Fits, estimate.Fits) || !reflect.DeepEqual(roundTrip.Recommended, estimate.Recommended) {
		t.Fatal("View changed CLI memory facts", err, roundTrip)
	}
	projected.RecommendationFailures[0] = RecommendationUnconfirmed
	*projected.HostFreeAfterSpillGiB = 99
	projected.RecommendedContext.BySessions[0].RecommendationFailures[0] = RecommendationUnconfirmed
	*projected.RecommendedContext.BySessions[0].HostFreeAfterSpillGiB = 99
	if estimate.RecommendationFailures[0] != RecommendationHostFloor || *estimate.HostFreeAfterSpillGiB != 1.8 || estimate.RecommendedContext.BySessions[0].RecommendationFailures[0] != RecommendationGPUHeadroom || *estimate.RecommendedContext.BySessions[0].HostFreeAfterSpillGiB != 0 {
		t.Fatal("View aliases CLI failure or remaining-memory fields")
	}
	copy := copyViewMemoryEstimate(projectMemoryEstimate(estimate))
	copy.RecommendationFailures[0] = RecommendationUnconfirmed
	copy.RecommendedContext.BySessions[0].RecommendationFailures[0] = RecommendationUnconfirmed
	if estimate.RecommendationFailures[0] != RecommendationHostFloor || estimate.RecommendedContext.BySessions[0].RecommendationFailures[0] != RecommendationGPUHeadroom {
		t.Fatal("View copy aliases CLI failure fields")
	}
}
