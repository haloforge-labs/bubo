//go:build windows

package supervisor

import "testing"

func fitEstimate(topFree, topHost *float64, rowFree float64) *MemoryEstimate {
	fits, recommended := true, true
	return &MemoryEstimate{
		Fits: &fits, FitScope: "conservative_startup_estimate", Recommended: &recommended, ContextTokens: 262144, Sessions: 1,
		FreeAfterGrowthGiB: topFree, HostFreeAfterSpillGiB: topHost,
		RecommendedContext: &RecommendedContext{BySessions: []RecommendedContextRow{
			{Sessions: 1, ContextPerSession: 57344, Fits: true, FreeAfterGiB: rowFree},
		}},
	}
}

// E14 item 7: the sentence states the numbers of the context the user chose
// (the top level of the estimate), not the recommended row's startup headroom.
func TestFitSentenceUsesTheTopLevelAfterGrowthNumbers(t *testing.T) {
	top := 12.3
	got := buildFit(fitEstimate(&top, nil, 40.0))
	if got.Sentence != "Fits. 12 GiB of memory would be left." {
		t.Fatalf("sentence %q", got.Sentence)
	}
	// An older CLI without the top-level number keeps the row.
	if got := buildFit(fitEstimate(nil, nil, 40.0)); got.Sentence != "Fits. 40 GiB of memory would be left." {
		t.Fatalf("fallback sentence %q", got.Sentence)
	}
}

func TestFitSentenceHostFloorUsesTheTopLevelHostNumber(t *testing.T) {
	top, host := 12.3, 1.8
	est := fitEstimate(&top, &host, 40.0)
	est.RecommendationFailures = []RecommendationFailure{RecommendationHostFloor}
	got := buildFit(est)
	if got.Sentence != tightExplanation+" System memory is low; close other apps." || got.Class != "memory_tight" || got.HostFreeAfterSpillGiB == nil || *got.HostFreeAfterSpillGiB != host {
		t.Fatalf("sentence %q", got.Sentence)
	}
}

// Both recommendation terms can fail together; the sentence names both and
// never claims only one of them. The default setup has no exception: the CLI's
// verdict is shown for every setup.
func TestFitSentenceWithBothRecommendationFailuresNamesBoth(t *testing.T) {
	top, host := 1.2, 1.8
	est := fitEstimate(&top, &host, 40.0)
	est.RecommendationFailures = []RecommendationFailure{RecommendationHostFloor, RecommendationGPUHeadroom}
	// UX14 K14-03: plain words and a next step, no jargon.
	want := tightExplanation + " System memory is low; close other apps. GPU memory is low."
	got := buildFit(est)
	if got.Sentence != want || got.Class != "memory_tight" || got.FreeAfterGiB == nil || *got.FreeAfterGiB != top || got.HostFreeAfterSpillGiB == nil || *got.HostFreeAfterSpillGiB != host {
		t.Fatalf("%+v", got)
	}
}

// E23: a larger RAM cache can put the projected host remainder below zero.
// Say the shortfall as a positive amount in the same fit verdict.
func TestE23NegativeHostMemoryIsShownAsShortfall(t *testing.T) {
	for _, tc := range []struct {
		name     string
		failures []RecommendationFailure
		class    string
		want     string
	}{
		{
			name:     "gpu and host headroom",
			failures: []RecommendationFailure{RecommendationGPUHeadroom, RecommendationHostFloor},
			class:    "memory_tight",
			want:     tightExplanation + " System memory is low; close other apps. GPU memory is low.",
		},
		{
			name:     "host floor only",
			failures: []RecommendationFailure{RecommendationHostFloor},
			class:    "memory_tight",
			want:     tightExplanation + " System memory is low; close other apps.",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			free, host := 2.6, -1.4
			est := fitEstimate(&free, &host, 40)
			est.RecommendationFailures = tc.failures
			got := buildFit(est)
			if got.Class != tc.class || got.Sentence != tc.want || got.HostFreeAfterSpillGiB == nil || *got.HostFreeAfterSpillGiB != host {
				t.Fatalf("negative host remainder was not reported as a shortfall: %+v", got)
			}
		})
	}
}

// When the CLI cannot say whether it fits, the reason is shown, not "Checking
// memory…" for ever.
func TestFitUnknownShowsTheCLIReason(t *testing.T) {
	reason := "Host spill is not checked: DXGI DedicatedVideoMemory is not reported."
	est := &MemoryEstimate{Fits: nil, FitReason: &reason}
	got := buildFit(est)
	if got.Sentence != reason || got.Class != "fit_unknown" {
		t.Fatalf("fits:null gave %+v", got)
	}
	if got := buildFit(&MemoryEstimate{}); got.Sentence != "Checking memory…" || got.Class != "unavailable" {
		t.Fatalf("no reason must stay the check line: %+v", got)
	}
}

// UX12 N12-04: the sentence names both parts, as the refusal does.
func TestDoesNotFitSentenceNamesSessionsAndContext(t *testing.T) {
	for sessions, want := range map[uint32]string{
		1: doesNotFitExplanation,
		2: doesNotFitExplanation,
	} {
		fits := false
		est := &MemoryEstimate{Fits: &fits, ContextTokens: 65536, Sessions: Count(sessions),
			RecommendedContext: &RecommendedContext{BySessions: []RecommendedContextRow{{Sessions: Count(sessions), ContextPerSession: 4096, Fits: true}}}}
		if got := buildFit(est); got.Sentence != want || got.Class != "memory_does_not_fit" {
			t.Fatalf("sessions %d: %+v", sessions, got)
		}
	}
}
