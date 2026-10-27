package supervisor

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMemoryAdmissionAndRecommendationRemainDistinct(t *testing.T) {
	var estimate MemoryEstimate
	if err := json.Unmarshal([]byte(`{"fits":true,"recommended":false,"fit_scope":"conservative_startup_estimate","recommendation_reason":"headroom","per_session_vision":{"bytes":64,"gib":0,"evidence":"estimated"},"gpu":{"total":{"bytes":128,"gib":0,"evidence":"measured"},"dedicated_video_memory":null,"dedicated_video_memory_reason":"unavailable"},"cache_at_clamp_bytes":32,"host_spill_bytes":0,"free_after_growth_gib":1.5}`), &estimate); err != nil {
		t.Fatal(err)
	}
	if estimate.Fits == nil || !*estimate.Fits || estimate.Recommended == nil || *estimate.Recommended {
		t.Fatal("CLI admission and recommendation verdicts were lost", estimate)
	}
	if estimate.PerSessionVision == nil || estimate.PerSessionVision.Bytes != 64 {
		t.Fatal("separate vision term was lost")
	}
	if estimate.GPU == nil || estimate.GPU.Total == nil || estimate.GPU.Total.Bytes != 128 || estimate.GPU.DedicatedVideoMemory != nil || estimate.GPU.DedicatedVideoMemoryReason == nil || *estimate.GPU.DedicatedVideoMemoryReason != "unavailable" || estimate.CacheAtClampBytes == nil || *estimate.CacheAtClampBytes != 32 || estimate.HostSpillBytes == nil || *estimate.HostSpillBytes != 0 || estimate.FreeAfterGrowthGiB == nil || *estimate.FreeAfterGrowthGiB != 1.5 {
		t.Fatal("CLI measured readings or cache/spill terms were lost", estimate)
	}
	if estimate.FitScope != "conservative_startup_estimate" {
		t.Fatal("CLI fit scope was lost")
	}
	raw, err := json.Marshal(estimate)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip MemoryEstimate
	if err := json.Unmarshal(raw, &roundTrip); err != nil || roundTrip.Recommended == nil || *roundTrip.Recommended {
		t.Fatal("false recommendation must survive the View JSON round trip", err)
	}
	if roundTrip.GPU == nil || roundTrip.GPU.Total == nil || roundTrip.GPU.Total.Bytes != 128 || roundTrip.GPU.DedicatedVideoMemory != nil || roundTrip.HostSpillBytes == nil || *roundTrip.HostSpillBytes != 0 || roundTrip.CacheAtClampBytes == nil || *roundTrip.CacheAtClampBytes != 32 {
		t.Fatal("measured readings, unknown carve and zero spill must survive the View round trip", roundTrip)
	}
	if err := json.Unmarshal([]byte(`{"fits":null,"recommended":null}`), &estimate); err != nil || estimate.Fits != nil || estimate.Recommended != nil {
		t.Fatal("unknown budgets must preserve unknown verdicts", err)
	}
}

func TestOptionalRecommendationFailuresRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want []RecommendationFailure
	}{
		{"older CLI", `{}`, nil},
		{"recommended", `{"recommendation_failures":[]}`, []RecommendationFailure{}},
		{"host only", `{"recommendation_failures":["host_floor"]}`, []RecommendationFailure{RecommendationHostFloor}},
		{"GPU", `{"recommendation_failures":["gpu_headroom"]}`, []RecommendationFailure{RecommendationGPUHeadroom}},
		{"unconfirmed", `{"recommendation_failures":["unconfirmed"]}`, []RecommendationFailure{RecommendationUnconfirmed}},
		{"mixed", `{"recommendation_failures":["host_floor","gpu_headroom"]}`, []RecommendationFailure{RecommendationHostFloor, RecommendationGPUHeadroom}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var estimate MemoryEstimate
			var row RecommendedContextRow
			if err := json.Unmarshal([]byte(tc.raw), &estimate); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tc.raw), &row); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(estimate.RecommendationFailures, tc.want) || !reflect.DeepEqual(row.RecommendationFailures, tc.want) {
				t.Fatal("CLI failures changed while decoding", estimate.RecommendationFailures, row.RecommendationFailures)
			}
		})
	}
}
