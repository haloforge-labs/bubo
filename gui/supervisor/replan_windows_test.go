//go:build windows

package supervisor

import (
	"testing"
	"time"
)

// E14 item 7: a new preflight starts a new plan for its component. Old plan
// parts must not pile up beside the new ones, or the user cannot approve.
func TestANewPreflightResetsThePlanPartsOfItsComponent(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Prepare())
	v := waitFor(t, s, "the first plan", 4*time.Second, func(v Snapshot) bool { return v.Approval != nil && len(v.PlanParts) > 0 })
	s.emit(update{v.Generation, "event", Event{Type: "preflight", Value: &Preflight{Component: "model", Phase: "preflight"}}})
	waitFor(t, s, "the reset", 4*time.Second, func(v Snapshot) bool { return len(v.PlanParts) == 0 })
	// The other component's plan stays.
	s.emit(update{v.Generation, "event", Event{Type: "plan", Value: &Plan{Component: "runtime", Parts: 1, Digest: "r"}}})
	waitFor(t, s, "the runtime plan", 4*time.Second, func(v Snapshot) bool { return len(v.PlanParts) == 1 })
	s.emit(update{v.Generation, "event", Event{Type: "preflight", Value: &Preflight{Component: "model", Phase: "preflight"}}})
	time.Sleep(200 * time.Millisecond)
	if n := len(s.Snapshot().PlanParts); n != 1 {
		t.Fatalf("a model preflight dropped the runtime plan: %d parts", n)
	}
}
