//go:build windows

package supervisor

import "testing"

// The start check judges the context serve will run: the CLI's recommended row
// for the session count, never a row that is not recommended.
func TestRecommendedSessionContext(t *testing.T) {
	yes, no := true, false
	estimate := &MemoryEstimate{RecommendedContext: &RecommendedContext{BySessions: []RecommendedContextRow{
		{Sessions: 1, ContextPerSession: 200000, Recommended: &yes},
		{Sessions: 2, ContextPerSession: 163840, Recommended: &yes},
		{Sessions: 3, ContextPerSession: 100000, Recommended: &no},
		{Sessions: 4, ContextPerSession: 50000},
	}}}
	for _, test := range []struct {
		sessions uint32
		want     uint64
	}{{1, 200000}, {2, 163840}, {3, 0}, {4, 0}, {5, 0}} {
		if got := recommendedSessionContext(estimate, test.sessions); got != test.want {
			t.Errorf("sessions %d: got %d, want %d", test.sessions, got, test.want)
		}
	}
	if recommendedSessionContext(nil, 2) != 0 || recommendedSessionContext(&MemoryEstimate{}, 2) != 0 {
		t.Error("a missing table must give no context")
	}
}
