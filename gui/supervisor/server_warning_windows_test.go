//go:build windows

package supervisor

import (
	"testing"
	"time"
)

// E14 item 7: a lifecycle warning of the running server reaches the View, once
// per code (the CLI may repeat it), and ends with the run.
func TestLifecycleWarningsReachTheView(t *testing.T) {
	t.Setenv("BUBO_FAKE_WARNING", "queue_pressure")
	s := supervisorFor(t, "normal")
	reply(t, s.Start())
	waitState(t, s, Ready)
	v := waitFor(t, s, "the warning", 6*time.Second, func(v Snapshot) bool { return len(v.ServerWarnings) > 0 })
	time.Sleep(200 * time.Millisecond)
	v = s.Snapshot()
	if len(v.ServerWarnings) != 1 || v.ServerWarnings[0].Code != "queue_pressure" || v.ServerWarnings[0].Remedy == "" {
		t.Fatalf("warnings: %+v", v.ServerWarnings)
	}
	if view := (&ViewBuilder{}).Build(ViewInput{Snapshot: v}); len(view.ServerWarnings) != 1 {
		t.Fatal("the View lost the warning")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
	reply(t, s.Start())
	waitState(t, s, Ready)
}
