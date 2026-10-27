//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func liveReplay(t *testing.T) map[string][]Slot {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "wails", "frontend", "src", "features", "requests", "progress-cli-facts.json"))
	if err != nil {
		t.Fatal(err)
	}
	var recording struct {
		Replay []struct {
			Name  string `json:"name"`
			Slots []Slot `json:"slots"`
		} `json:"replay"`
	}
	if err := json.Unmarshal(b, &recording); err != nil {
		t.Fatal(err)
	}
	out := map[string][]Slot{}
	for _, frame := range recording.Replay {
		out[frame.Name] = frame.Slots
	}
	return out
}

func TestLiveSlotRatesFollowCLIWatchReplay(t *testing.T) {
	frames := liveReplay(t)
	at := time.Now()
	var sampler liveSlotSampler
	step := func(name string, sec int) Snapshot {
		st := &Status{Identity: &Identity{ID: "recording-instance"}, Slots: frames[name]}
		var snap Snapshot
		sampler.observe(&snap, st, 1, at.Add(time.Duration(sec)*time.Second))
		return snap
	}
	first := step("long", 0)
	if first.LiveSlots[0].PrefillTPS != nil {
		t.Fatal("first observation invented a rate")
	}
	step("long", 1)
	moved := step("long-step", 2)
	if got := moved.LiveSlots[0].PrefillTPS; got == nil || *got != 2048 {
		t.Fatalf("prefill rate = %v, want 2048", got)
	}
	paused := step("long-step", 3)
	if got := paused.LiveSlots[0].PrefillTPS; got == nil || *got != 2048 {
		t.Fatalf("unmoved counter must retain averaged engine-step rate: %v", got)
	}
	averaged := step("long-step-two", 4)
	if got := averaged.LiveSlots[0].PrefillTPS; got == nil || *got != 2048 {
		t.Fatalf("step average = %v", got)
	}
	reset := step("backwards", 5)
	if reset.LiveSlots[0].PrefillTPS != nil || len(reset.LiveSamples) != 1 {
		t.Fatal("counter reset retained old request series")
	}
	step("cached", 6)
	reading := step("cached-finish", 7)
	if got := reading.LiveSlots[0].PrefillTPS; got == nil || *got != 28 {
		t.Fatalf("cached tokens counted as prefill work: %v", got)
	}
	writing := step("writing", 8)
	if writing.LiveSlots[0].DecodeTPS != nil {
		t.Fatal("phase transition invented decode speed")
	}
	writing = step("writing", 9)
	if got := writing.LiveSlots[0].DecodeTPS; got == nil || *got != 0 {
		t.Fatalf("CLI decode delta on unmoved token counter = %v, want 0", got)
	}
	step("idle", 10)
	var snap Snapshot
	sampler.observe(&snap, &Status{Identity: &Identity{ID: "recording-instance"}, Slots: frames["idle"]}, 1, at.Add(12*time.Second))
	if len(snap.LiveSamples) == 0 {
		t.Fatal("completion lost the live chart immediately")
	}
	sampler.observe(&snap, &Status{Identity: &Identity{ID: "recording-instance"}, Slots: frames["idle"]}, 1, at.Add(16*time.Second))
	if len(snap.LiveSamples) != 0 {
		t.Fatal("completed chart retained beyond five seconds")
	}
	sampler.observe(&snap, &Status{Identity: &Identity{ID: "next-instance"}, Slots: frames["writing"]}, 2, at.Add(17*time.Second))
	if len(snap.LiveSamples) != 1 || snap.LiveSlots[0].DecodeTPS != nil {
		t.Fatal("instance/generation inherited rates or history")
	}
	before := len(snap.LiveSamples)
	sampler.observe(&snap, &Status{Identity: &Identity{ID: "next-instance"}, Slots: frames["writing"]}, 2, at.Add(17*time.Second))
	if len(snap.LiveSamples) != before {
		t.Fatal("duplicate observation appended a point")
	}
}

func TestLiveSlotViewRecordingContract(t *testing.T) {
	st := &Status{Phase: "ready", Slots: liveReplay(t)["writing"]}
	v := new(ViewBuilder).Build(ViewInput{Snapshot: Snapshot{Initialized: true, State: Ready, Status: st}})
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatal(err)
	}
	slots := wire["slots"].([]any)
	for _, key := range []string{"prefill_tps", "decode_tps", "elapsed_ms"} {
		if _, ok := slots[0].(map[string]any)[key]; !ok {
			t.Errorf("writing session lacks %s", key)
		}
	}
	if _, ok := wire["live_samples"].([]any); !ok {
		t.Error("live_samples must always be an array")
	}
	if _, ok := wire["live_sample_time"].(string); !ok {
		t.Error("live_sample_time must expose the observed clock separately from the CLI clock")
	}
}

func TestLiveSlotRecordedStatusProjection(t *testing.T) {
	load := func(name string) *Status {
		b, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", "lrel8", name))
		if err != nil {
			t.Fatal(err)
		}
		e, err := ParseEvent(b)
		if err != nil {
			t.Fatal(err)
		}
		st, err := statusRecord(e)
		if err != nil {
			t.Fatal(err)
		}
		return st
	}
	var sampler liveSlotSampler
	var snap Snapshot
	at := time.Now().Add(-time.Second)
	var builder ViewBuilder
	export := func(name string, st *Status, now time.Time) View {
		sampler.observe(&snap, st, 1, now)
		snap.State, snap.Initialized, snap.Status = Ready, true, st
		snap.Prepared = st.Prepared
		view := builder.Build(ViewInput{Snapshot: snap})
		if dir := os.Getenv("BUBO_LIVE_REQUEST_EVIDENCE"); dir != "" {
			if err := os.MkdirAll(dir, 0700); err != nil {
				t.Fatal(err)
			}
			b, err := json.MarshalIndent(view, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name+".json"), append(b, '\n'), 0600); err != nil {
				t.Fatal(err)
			}
		}
		return view
	}
	writing := export("live-writing", load("status-0027-002858.110.json"), at)
	if len(writing.Slots) != 1 || writing.Slots[0].State != "generating" || writing.Slots[0].DecodeTPS != nil || len(writing.LiveSamples) != 1 {
		t.Fatal("real writing observation must preserve unknown speed", writing.Slots)
	}
	before := len(snap.LiveSamples)
	for i := 0; i < 3; i++ {
		_ = builder.Build(ViewInput{Snapshot: snap})
	}
	if len(snap.LiveSamples) != before {
		t.Fatal("View polls duplicated live points")
	}
	completed := export("live-completed", load("status-0029-002905.317.json"), at.Add(time.Second))
	if len(completed.LiveSamples) != 1 || completed.LastCompleted == nil || len(completed.Requests) == 0 {
		t.Fatal("first completed status must expose completed facts and retain live curve")
	}
	completed.LiveSamples[0].RequestID = 999
	if snap.LiveSamples[0].RequestID == 999 {
		t.Fatal("View aliases mutable retained history")
	}
	snap.LiveSamples[0].RetainUntil = time.Now().Add(-time.Second).Format(time.RFC3339Nano)
	if len(builder.Build(ViewInput{Snapshot: snap}).LiveSamples) != 0 {
		t.Fatal("View must expire a completed chart without another status sample")
	}
	frames := liveReplay(t)
	sampler = liveSlotSampler{}
	// The reading sequence is explicitly arranged. Preserve preparation and
	// package facts from the real snapshot instead of inventing those fields.
	st := load("status-0027-002858.110.json")
	st.Slots = frames["long"]
	export("live-reading-baseline", st, at.Add(-2*time.Second))
	st.Slots = frames["long-step"]
	reading := export("live-reading", st, at)
	if reading.Slots[0].PrefillTPS == nil || *reading.Slots[0].PrefillTPS != 2048 {
		t.Fatal("arranged reading steps lost rate")
	}
}
