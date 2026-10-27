//go:build windows

package supervisor

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCPURecordedLifecycleViews(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", "json-exponent", "driver-serve.stdout.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var builder ViewBuilder
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		event, err := ParseEvent(line)
		if err != nil {
			t.Fatal(err)
		}
		life, ok := event.Value.(*Lifecycle)
		if !ok || event.Type != "starting" && event.Type != "loading" && event.Type != "ready" {
			continue
		}
		seen[event.Type] = true
		snap := Snapshot{Initialized: true, State: Starting, Stale: true, StatusFeed: StatusFeed{State: StatusWaiting}, Status: &Status{Phase: event.Type, RuntimeDetail: life.RuntimeDetail, Effective: life.Effective}}
		if event.Type == "ready" {
			snap.State, snap.Stale = Ready, false
			snap.StatusFeed.State = StatusUpdating
		}
		view := builder.Build(ViewInput{Snapshot: snap, Saved: DefaultSettings()})
		if view.Stale || view.Failure != nil {
			t.Fatalf("CPU %s: %+v", event.Type, view)
		}
		t.Logf("real CPU lifecycle %s: phase=%s label=%q stale=%t", event.Type, view.Phase, view.StatusLabel, view.Stale)
	}
	for _, phase := range []string{"starting", "loading", "ready"} {
		if !seen[phase] {
			t.Fatal("missing real CPU phase", phase)
		}
	}
}

// This process E2E uses the existing recorded fake CLI, since the unmodified
// CPU contract driver neither accepts --context nor implements model admission.
// The separate raw CPU recordings are replayed by TestJSONExponentRealStatusView.
func TestSupervisorStartWorkflowViews(t *testing.T) {
	s := streamSupervisor(t, "normal")
	t.Setenv("BUBO_FAKE_START_DELAY_MS", "700")
	t.Setenv("BUBO_FAKE_LOAD_MS", "1800")
	settings := DefaultSettings()
	settings.Context = 200000
	var builder ViewBuilder
	writeView := func(name string, snap Snapshot) View {
		t.Helper()
		view := builder.Build(ViewInput{Snapshot: snap, Saved: settings})
		t.Logf("%s: phase=%s label=%q serving=%t stale=%t feed=%s", name, view.Phase, view.StatusLabel, view.Serving, view.Stale, view.StatusFeed.State)
		if view.Stale || view.Failure != nil {
			t.Fatalf("normal start: %+v", view)
		}
		if dir := os.Getenv("BUBO_START_WORKFLOW_EVIDENCE"); dir != "" {
			data, err := json.MarshalIndent(view, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "workflow-"+name+".json"), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		return view
	}
	waitFor(t, s, "initial stopped", 5*time.Second, func(v Snapshot) bool { return v.Initialized && v.State == Stopped })
	writeView("stopped", s.Snapshot())
	started := s.Start("--context", "200000")
	waitFor(t, s, "starting", 5*time.Second, func(v Snapshot) bool { return v.State == Starting })
	writeView("starting", s.Snapshot())
	// The watch pauses while Start owns the lifecycle stream. Observe the
	// owned fake child's loading marker and the supervisor's Starting state.
	waitFor(t, s, "owned child loading", 5*time.Second, func(v Snapshot) bool {
		var instance struct {
			Phase string `json:"phase"`
		}
		data, err := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "instance.json"))
		return err == nil && json.Unmarshal(data, &instance) == nil && instance.Phase == "loading" && v.State == Starting
	})
	writeView("loading", s.Snapshot())
	reply(t, started)
	waitFor(t, s, "first Ready", 5*time.Second, func(v Snapshot) bool { return v.State == Ready })
	view := writeView("ready-first-sample", s.Snapshot())
	waitFeed(t, s, StatusUpdating)
	view = writeView("ready", s.Snapshot())
	if !view.Serving || view.Connection == nil || view.Connection.ContextTokens == nil || *view.Connection.ContextTokens != 200000 {
		t.Fatal("ready context lost", view.Connection)
	}
	// A feed that actually stops updating after Ready must still be signalled.
	streamFlag(t, "watch-stall")
	waitFeed(t, s, StatusNotUpdating)
	view = builder.Build(ViewInput{Snapshot: s.Snapshot(), Saved: settings})
	if !view.Stale || !view.Serving {
		t.Fatal("running feed failure was hidden")
	}
	reply(t, s.Stop())
	waitState(t, s, Stopped)
}
