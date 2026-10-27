//go:build windows

package supervisor

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These tests drive the real supervisor against the contract-shaped fake CLI.
// A test scripts the fake server's runtime numbers by writing status.json into
// the fake's state folder; the fake merges it into every status snapshot.

func scriptStatus(t *testing.T, over map[string]any) {
	t.Helper()
	dir := os.Getenv("BUBO_FAKE_DIR")
	if dir == "" {
		t.Error("BUBO_FAKE_DIR is empty: refusing to write status.json into the working folder")
		return
	}
	b, err := json.Marshal(over)
	if err != nil {
		t.Fatal(err)
	}
	tmp := filepath.Join(dir, "status.tmp")
	if err := os.WriteFile(tmp, b, 0600); err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 50; n++ {
		if err = os.Rename(tmp, filepath.Join(dir, "status.json")); err == nil {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal(err)
}
func clearScript(t *testing.T) {
	t.Helper()
	os.Remove(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "status.json"))
}
func slotJSON(id int, state string, used, prefilled, generated uint64) map[string]any {
	return map[string]any{"id": id, "processing": state != "idle", "state": state, "tokens_used": used, "speculative": false, "request_id": 1, "context_tokens": 262144, "prompt_tokens": prefilled, "cached_prompt_tokens": 0, "prefilled_tokens": prefilled, "generated_tokens": generated, "remaining_tokens": nil}
}
func countersJSON(prefill, decode uint64) map[string]any {
	return map[string]any{"prefill_tokens_total": prefill, "decode_tokens_total": decode, "cached_tokens_total": 0, "cache_disk_hits_total": 0, "cache_memory_hits_total": 0, "cache_misses_total": 1, "max_sequence_tokens": 0, "draft_rounds_total": 0, "draft_tokens_total": 0, "draft_accepted_tokens_total": 0, "device_lost_total": 0, "prefill_seconds_total": 0.5, "decode_seconds_total": 0.5}
}
func waitFor(t *testing.T, s *Supervisor, what string, d time.Duration, ok func(Snapshot) bool) Snapshot {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if v := s.Snapshot(); ok(v) {
			return v
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for %s; state %s stall=%v rates=%v err=%v label=%q", what, s.Snapshot().State, s.Snapshot().Stall, s.Snapshot().Rates, s.Snapshot().LastError, s.Snapshot().Label)
	return Snapshot{}
}
func stallSupervisor(t *testing.T, timeout time.Duration) *Supervisor {
	t.Helper()
	c := configFor(t, "normal")
	c.StallTimeout = timeout
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-s.finished:
			return
		default:
		}
		select {
		case <-s.Exit():
		case <-time.After(8 * time.Second):
			t.Error("Exit did not finish")
		}
	})
	reply(t, s.Start())
	waitState(t, s, Ready)
	return s
}
func pidAlive(t *testing.T, v Snapshot, c Config) {
	t.Helper()
	if v.Instance == nil {
		t.Fatal("no instance")
	}
	if err := VerifyIdentity(*v.Instance, c.Exe); err != nil {
		t.Fatalf("the server is no longer the identified live process: %v", err)
	}
}

// TestEndToEndPrepareStartStatusRestartNeededStop is the whole owned flow with
// the real contract: consent, ready, status updates (rates, slots, requests),
// a settings change that needs a restart, and a clean stop.
func TestEndToEndPrepareStartStatusRestartNeededStop(t *testing.T) {
	c := configFor(t, "ui") // ui mode: five plan parts, a question, unknown total first
	c.InitialPrepared = false
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	waitFor(t, s, "discovery", 5*time.Second, func(v Snapshot) bool { return v.Initialized })
	if v := s.Snapshot(); v.State != NotPrepared || v.Prepared {
		t.Fatalf("expected Not prepared %v err=%v label=%q", v.State, v.LastError, v.Label)
	}

	// Prepare with approval.
	reply(t, s.Prepare())
	v := waitFor(t, s, "approval", 6*time.Second, func(v Snapshot) bool { return v.Approval != nil })
	if v.Approval.Question != "" || len(v.Approval.Digest) != 64 || len(v.PlanParts) != 5 || len(v.Approval.Components) != 1 || v.Approval.Components[0] != "runtime" {
		t.Fatal("approval or plan parts", v.Approval, len(v.PlanParts))
	}
	if v.Approval.Digest == v.PlanParts[0].Digest {
		t.Fatal("approval digest must bind the runtime plan")
	}
	kinds := map[string]bool{}
	for _, p := range v.PlanParts {
		kinds[p.RecordKind] = true
	}
	if len(kinds) != 5 {
		t.Fatal("plan record kinds", kinds)
	}
	reply(t, s.Consent(true))
	v = waitFor(t, s, "unknown-total progress", 6*time.Second, func(v Snapshot) bool { return v.Progress != nil && v.Progress.Total == nil })
	waitFor(t, s, "runtime prepared", 8*time.Second, func(v Snapshot) bool {
		return v.State == Stopped && v.Status != nil && v.Status.Readiness.Runtime && !v.Status.Readiness.Model
	})
	// This supervisor contract then completes the model step before exercising
	// the owned server. The GUI's model consent uses ModelService separately.
	reply(t, s.Prepare("model"))
	v = waitFor(t, s, "model approval", 6*time.Second, func(v Snapshot) bool { return v.Approval != nil })
	if len(v.Approval.Components) != 1 || v.Approval.Components[0] != "model" {
		t.Fatal("model step was not separate", v.Approval)
	}
	reply(t, s.Consent(true))
	waitFor(t, s, "model prepared", 8*time.Second, func(v Snapshot) bool { return v.State == Stopped && v.Prepared })

	// Start. From here the fake is plain (no ui numbers) so rates come from counters.
	t.Setenv("BUBO_FAKE_MODE", "normal")
	reply(t, s.Start("--profile", "default", "--context", "262144", "--sessions", "1"))
	v = waitState(t, s, Ready)
	if !regexp.MustCompile(`^[a-f0-9]{32}$`).MatchString(v.Instance.ID) || v.Instance.PID == 0 || v.Instance.CreationTime == "" {
		t.Fatal("identity fields", v.Instance)
	}
	if v.Instance.ExePath == "" || v.Instance.Endpoint == nil || !strings.HasPrefix(*v.Instance.Endpoint, "http://127.0.0.1:19091/") {
		t.Fatal("endpoint", v.Instance)
	}
	waitFor(t, s, "registry", 6*time.Second, func(v Snapshot) bool { return v.RegistryLoaded })
	if reg := s.Registry(); reg == nil || len(reg.ByAudience("user")) == 0 || len(reg.ByAudience("advanced")) == 0 || len(reg.ByAudience("debug")) == 0 {
		t.Fatal("registry rows by audience")
	}

	// Status updates: counters move, a slot generates, a request completes.
	var prefill, decode atomic.Uint64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		tick := time.NewTicker(40 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				p, d := prefill.Add(40), decode.Add(8)
				scriptStatus(t, map[string]any{
					"counters": countersJSON(p, d), "sessions": 1, "queue": 0,
					"slots":    []any{slotJSON(0, "generating", p+d, p, d)},
					"requests": []any{map[string]any{"id": "r1", "time": "2026-10-14T05:00:00Z", "slot": 0, "prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0, "ttft_ms": 20, "prefill_tps": 2000, "decode_tps": 50, "cache_kind": "miss", "finish_reason": "stop", "outcome": "completed"}},
				})
			}
		}
	}()
	v = waitFor(t, s, "rates, slots and requests", 8*time.Second, func(v Snapshot) bool {
		return v.Rates != nil && v.Rates.DecodeTPS != nil && v.Status != nil && len(v.Status.Slots) == 1 && len(v.Status.Requests) == 1
	})
	close(stop)
	wg.Wait()
	if v.Rates.Source != "client_delta" || *v.Rates.DecodeTPS <= 0 || *v.Rates.PrefillTPS <= 0 || v.Rates.Label != RateLabel || v.Rates.Unit != RateUnit {
		t.Fatal("rates", v.Rates)
	}
	if v.Status.Slots[0].State != "generating" || v.Status.Requests[0].ID != "r1" || v.Status.Requests[0].Outcome != "completed" {
		t.Fatal("slots or requests", v.Status.Slots, v.Status.Requests)
	}
	if busy, known := inFlight(v.Status); !known || !busy {
		t.Fatal("CLI activity missing", v.Status)
	}
	if v.Stall != nil {
		t.Fatal("tokens moved but a stall was raised", v.Stall)
	}
	if len(v.CounterDelta) == 0 || v.CountersReset {
		t.Fatal("counter delta", v.CounterDelta, v.CountersReset)
	}

	// A settings change needs a restart; the restart clears it.
	saved := DefaultSettings()
	saved.Context = 131072
	saved.Port = 19091
	changes := RestartNeeded(saved, v.Instance, v.Status.Effective, fixtureEvent(t, "switches").Value.(*Switches))
	if len(changes) != 1 || changes[0].Key != "context" || changes[0].Running != "262144" || changes[0].Next != "131072" {
		t.Fatalf("restart-needed: %+v", changes)
	}
	same := DefaultSettings()
	same.Port = 19091
	if got := RestartNeeded(same, v.Instance, v.Status.Effective, fixtureEvent(t, "switches").Value.(*Switches)); len(got) != 0 {
		t.Fatalf("no change must not need a restart: %+v", got)
	}
	clearScript(t)
	first := v.Instance.ID
	reply(t, s.Restart("--profile", "default", "--context", "131072", "--sessions", "1"))
	v = waitFor(t, s, "second instance ready", 10*time.Second, func(v Snapshot) bool {
		return v.State == Ready && v.Instance != nil && v.Instance.ID != first && v.Status != nil && v.Status.Identity != nil && v.Status.Identity.ID == v.Instance.ID
	})
	if got := RestartNeeded(saved, v.Instance, v.Status.Effective, fixtureEvent(t, "switches").Value.(*Switches)); len(got) != 0 {
		t.Fatalf("restart applied but still pending: %+v", got)
	}
	if v.Stall != nil {
		t.Fatal("idle server raised a stall")
	}

	// Stop.
	reply(t, s.Stop())
	v = waitState(t, s, Stopped)
	if v.Forced || v.Instance != nil || v.Rates != nil || v.Stall != nil {
		t.Fatal("clean stop left state", v)
	}
}

func TestApprovalSpaceCheckReadsComponentParts(t *testing.T) {
	c := configFor(t, "ui")
	c.InitialPrepared = false
	if err := os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "volume-short"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	waitFor(t, s, "discovery", 5*time.Second, func(v Snapshot) bool { return v.Initialized })
	reply(t, s.Prepare())
	v := waitFor(t, s, "approval", 6*time.Second, func(v Snapshot) bool { return v.Approval != nil })
	if len(v.PlanParts) != 5 || len(v.Approval.Components) != 1 || v.Approval.Components[0] != "runtime" || v.Approval.Digest == v.PlanParts[0].Digest {
		t.Fatalf("fake CLI did not offer component-bound approval: %+v", v.Approval)
	}
	result := <-s.Consent(true)
	var coded *AppError
	if !errors.As(result.Err, &coded) || coded.Code != "insufficient_space" {
		t.Fatalf("short plan was accepted: %v", result.Err)
	}
}

func TestStalledButHealthyServerWarnsAndNeverKills(t *testing.T) {
	s := stallSupervisor(t, time.Second)
	c := s.config
	// Work is in flight; /health-style readiness is fine; no token moves.
	scriptStatus(t, map[string]any{"counters": countersJSON(100, 50), "sessions": 1, "queue": 0, "slots": []any{slotJSON(0, "generating", 150, 100, 50)}})
	v := waitFor(t, s, "stall warning", 8*time.Second, func(v Snapshot) bool { return v.Stall != nil })
	if !strings.HasPrefix(v.Stall.Message, "No token progress for ") || !strings.HasSuffix(v.Stall.Message, " s") || v.Stall.Seconds < 1 || v.Stall.ThresholdSecs != 1 {
		t.Fatal("stall message", v.Stall)
	}
	if v.Label != v.Stall.Message {
		t.Fatal("the stall line must reach the label that the tray reads", v.Label)
	}
	if v.State != Ready || v.Status == nil || !v.Status.Readiness.Runtime || v.Status.Readiness.Server == nil || !*v.Status.Readiness.Server || v.Status.Phase != "ready" {
		t.Fatal("the server is healthy by every flag, yet", v.State)
	}
	// It warns and offers Restart; it does not stop or kill by itself.
	time.Sleep(1200 * time.Millisecond)
	v = s.Snapshot()
	if v.State != Ready || v.StopPhase != "" || v.Stall == nil || v.Stall.Seconds < 2 {
		t.Fatal("watchdog acted or lost the warning", v.State, v.StopPhase, v.Stall)
	}
	pidAlive(t, v, c)
	if _, err := os.Stat(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "stop")); err == nil {
		t.Fatal("a stop was requested by the watchdog")
	}
	// Restart is the offered action and works while stalled.
	clearScript(t)
	first := v.Instance.ID
	reply(t, s.Restart())
	v = waitFor(t, s, "restart after stall", 10*time.Second, func(v Snapshot) bool {
		return v.State == Ready && v.Instance != nil && v.Instance.ID != first
	})
	if v.Stall != nil {
		t.Fatal("warning survived the restart", v.Stall)
	}
}

func TestStallRecoversWhenTokensAdvance(t *testing.T) {
	s := stallSupervisor(t, time.Second)
	scriptStatus(t, map[string]any{"counters": countersJSON(10, 10), "sessions": 1, "queue": 0, "slots": []any{slotJSON(0, "generating", 20, 10, 10)}})
	waitFor(t, s, "stall warning", 8*time.Second, func(v Snapshot) bool { return v.Stall != nil })
	// The counters move again: the warning clears at the next sample.
	scriptStatus(t, map[string]any{"counters": countersJSON(11, 12), "sessions": 1, "queue": 0, "slots": []any{slotJSON(0, "generating", 23, 11, 12)}})
	v := waitFor(t, s, "recovery", 5*time.Second, func(v Snapshot) bool { return v.Stall == nil })
	if v.State != Ready {
		t.Fatal(v.State)
	}
	// Frozen again: the timer starts from the last movement, not from the first stall.
	waitFor(t, s, "second stall", 8*time.Second, func(v Snapshot) bool { return v.Stall != nil })
}

func TestIdleServerNeverWarns(t *testing.T) {
	s := stallSupervisor(t, time.Second)
	scriptStatus(t, map[string]any{"counters": countersJSON(5, 5), "sessions": 0, "queue": 0, "slots": []any{slotJSON(0, "idle", 0, 0, 0)}})
	deadline := time.Now().Add(2600 * time.Millisecond) // more than twice the threshold
	for time.Now().Before(deadline) {
		if v := s.Snapshot(); v.Stall != nil {
			t.Fatal("idle server raised a stall", v.Stall)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestStallInputsAndSwitches(t *testing.T) {
	t.Run("queue only counts as work in flight", func(t *testing.T) {
		s := stallSupervisor(t, time.Second)
		scriptStatus(t, map[string]any{"counters": countersJSON(5, 5), "sessions": 0, "queue": 3, "slots": []any{slotJSON(0, "idle", 0, 0, 0)}})
		waitFor(t, s, "stall with a queue", 8*time.Second, func(v Snapshot) bool { return v.Stall != nil })
	})
	t.Run("moving slot tokens count as progress even if totals do not move", func(t *testing.T) {
		s := stallSupervisor(t, time.Second)
		done := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			var n uint64
			for {
				select {
				case <-done:
					return
				case <-time.After(60 * time.Millisecond):
					n++
					scriptStatus(t, map[string]any{"counters": countersJSON(5, 5), "sessions": 1, "queue": 0, "slots": []any{slotJSON(0, "generating", 100+n, 50, n)}})
				}
			}
		}()
		deadline := time.Now().Add(2600 * time.Millisecond)
		for time.Now().Before(deadline) {
			if v := s.Snapshot(); v.Stall != nil {
				t.Fatal("a generating slot with moving tokens raised a stall", v.Stall)
			}
			time.Sleep(20 * time.Millisecond)
		}
		close(done)
		wg.Wait()
	})
	t.Run("unavailable slots and queue are not judged", func(t *testing.T) {
		s := stallSupervisor(t, time.Second)
		scriptStatus(t, map[string]any{"counters": nil, "sessions": nil, "queue": nil, "slots": nil})
		waitFor(t, s, "null slots reach the window", 5*time.Second, func(v Snapshot) bool { return v.Status != nil && v.Status.Slots == nil && v.Status.Counters == nil })
		deadline := time.Now().Add(2600 * time.Millisecond)
		for time.Now().Before(deadline) {
			v := s.Snapshot()
			if v.Stall != nil {
				t.Fatal("unavailable data raised a stall", v.Stall)
			}
			if v.Status != nil && v.Status.Slots != nil {
				t.Fatal("null slots became a list")
			}
			time.Sleep(20 * time.Millisecond)
		}
		if v := s.Snapshot(); v.Status == nil {
			t.Fatal("status unavailable")
		} else if _, known := inFlight(v.Status); known {
			t.Fatal("absent CLI activity became known", v.Status)
		}
	})
	t.Run("a negative timeout turns the watchdog off; SetStallTimeout applies at once", func(t *testing.T) {
		s := stallSupervisor(t, -1)
		scriptStatus(t, map[string]any{"counters": countersJSON(1, 1), "sessions": 1, "queue": 0, "slots": []any{slotJSON(0, "generating", 2, 1, 1)}})
		time.Sleep(1500 * time.Millisecond)
		if s.Snapshot().Stall != nil {
			t.Fatal("disabled watchdog warned")
		}
		s.SetStallTimeout(time.Second)
		waitFor(t, s, "stall after enabling", 8*time.Second, func(v Snapshot) bool { return v.Stall != nil })
	})
}

func TestStallDefaultIsOneHundredTwentySeconds(t *testing.T) {
	c := configFor(t, "normal")
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { reply(t, s.Exit()) }()
	if s.StallTimeout() != 120*time.Second || DefaultSettings().StallTimeout() != 120*time.Second {
		t.Fatal(s.StallTimeout())
	}
	st := DefaultSettings()
	st.StallSeconds = 45
	if st.StallTimeout() != 45*time.Second {
		t.Fatal("setting ignored")
	}
}

func TestExternalServerStatusIsReadOnlyAndStallAware(t *testing.T) {
	c := configFor(t, "normal")
	c.StallTimeout = time.Second
	p, i := startExternal(t, c)
	scriptStatus(t, map[string]any{"counters": countersJSON(7, 7), "sessions": 1, "queue": 0, "slots": []any{slotJSON(0, "generating", 14, 7, 7)}})
	s, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reply(t, s.Exit()) })
	reply(t, s.Discover())
	v := waitState(t, s, AttachedExternal)
	if v.Managed || v.Label != "Started outside this window" || v.Instance.ID != i.ID {
		t.Fatal(v)
	}
	v = waitFor(t, s, "external slots and stall", 8*time.Second, func(v Snapshot) bool {
		return v.Status != nil && len(v.Status.Slots) == 1 && v.Stall != nil
	})
	if v.Status.Slots[0].State != "generating" || v.State != AttachedExternal {
		t.Fatal(v.Status.Slots)
	}
	// Read only: Stop is refused, and closing this window leaves the server alive.
	if r := <-s.Stop(); r.Err == nil {
		t.Fatal("read-only stop accepted")
	}
	if r := <-s.Restart(); r.Err == nil {
		t.Fatal("read-only restart accepted")
	}
	reply(t, s.Exit())
	if err := VerifyIdentity(i, c.Exe); err != nil {
		t.Fatal("closing the window touched the external server", err)
	}
	select {
	case <-p.Done:
		t.Fatal("external server exited")
	default:
	}
}

func TestFailureKeepsTheCLIErrorAndRetryIsExplicit(t *testing.T) {
	s := supervisorFor(t, "ui-fail")
	t.Setenv("BUBO_FAKE_MODE", "ui-fail")
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "prepared"), []byte("true"), 0600)
	reply(t, s.Start())
	waitState(t, s, Ready)
	os.WriteFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "fail"), []byte("fail"), 0600)
	v := waitFor(t, s, "failure", 8*time.Second, func(v Snapshot) bool { return v.State == Failed && v.Instance == nil })
	e := v.LastError
	if e == nil || e.Code != "device_lost" || e.Source != "cli" || e.Component != "serve" || e.Stage != "decode" || e.Remedy != "Restart on request." {
		t.Fatalf("typed CLI error lost: %+v", e)
	}
	if v.Rates != nil || v.Stall != nil {
		t.Fatal("a dead server kept live numbers")
	}
}

func TestMajorVersionMismatchShowsAProtocolErrorInTheWindow(t *testing.T) {
	s := supervisorFor(t, "major")
	v := waitFor(t, s, "protocol error", 6*time.Second, func(v Snapshot) bool { return v.Initialized })
	if v.State != Failed || v.LastError == nil || v.LastError.Code != "protocol_mismatch" {
		t.Fatal("version mismatch must fail visibly", v.State, v.LastError)
	}
	if !strings.HasPrefix(v.Label, "Protocol error: unsupported schema_version 2") {
		t.Fatalf("label does not name the protocol error: %q", v.Label)
	}
	// Start must not run a server under an unreadable contract.
	if r := <-s.Start(); r.Err == nil {
		v := waitNoOperation(t, s)
		if v.State != Failed || v.LastError.Code != "protocol_mismatch" {
			t.Fatal("start under a major mismatch", v.State, v.LastError)
		}
	}
}

func TestLifecycleIdentityOfAnotherProcessIsRejected(t *testing.T) {
	s := supervisorFor(t, "spoof-pid")
	reply(t, s.Start())
	v := waitNoOperation(t, s)
	if v.State != Failed || v.LastError == nil || v.LastError.Code != "protocol_mismatch" {
		t.Fatal("an event naming another PID was accepted", v.State, v.LastError)
	}
}

func TestRequestCompletedEventsKeepTheServerReady(t *testing.T) {
	s := supervisorFor(t, "normal")
	reply(t, s.Start())
	waitState(t, s, Ready)
	dir := os.Getenv("BUBO_FAKE_DIR")
	row := `{"id":"r9","time":"2026-10-14T05:00:00Z","slot":null,"prompt_tokens":10,"completion_tokens":2,"cached_tokens":0,"ttft_ms":null,"prefill_tps":null,"decode_tps":null,"cache_kind":"miss","finish_reason":"stop","outcome":"completed"}`
	os.WriteFile(filepath.Join(dir, "emit-request"), []byte(row), 0600)
	time.Sleep(500 * time.Millisecond)
	if v := s.Snapshot(); v.State != Ready || v.LastError != nil {
		t.Fatal("a valid request_completed event changed the lifecycle", v.State, v.LastError)
	}
	// A row that breaks the contract is a Protocol error, not an ignored line.
	os.WriteFile(filepath.Join(dir, "emit-request"), []byte(`{"id":"r10"}`), 0600)
	v := waitFor(t, s, "protocol error", 8*time.Second, func(v Snapshot) bool { return v.State == Failed })
	if v.LastError.Code != "protocol_mismatch" {
		t.Fatal(v.LastError)
	}
}

func TestStopResultNeedsConfirmedExit(t *testing.T) {
	id := Identity{ID: strings.Repeat("a", 32), PID: 77, CreationTime: "134000000000000000", ExePath: "C:/Bubo/bubo.exe"}
	inst := Instance{ID: id.ID, PID: id.PID, CreationTime: id.CreationTime, ExePath: id.ExePath}
	mk := func(kind string, who Identity, ack, exited *bool) Event {
		return Event{Type: kind, Value: &Lifecycle{Identity: who, Acknowledged: ack, ProcessExited: exited}}
	}
	yes, no := true, false
	if err := checkStop([]Event{mk("stopping", id, &yes, &no), mk("stopped", id, nil, &yes)}, inst); err != nil {
		t.Fatal(err)
	}
	other := id
	other.PID = 78
	for name, events := range map[string][]Event{
		"no records":         nil,
		"ack only":           {mk("stopping", id, &yes, &no)},
		"not acknowledged":   {mk("stopping", id, &no, &no), mk("stopped", id, nil, &yes)},
		"stopped not exited": {mk("stopping", id, &yes, &no), mk("stopped", id, nil, &no)},
		"exit flag missing":  {mk("stopped", id, nil, nil)},
		"another identity":   {mk("stopping", other, &yes, &no), mk("stopped", other, nil, &yes)},
		"wrong event type":   {mk("ready", id, nil, nil)},
	} {
		if checkStop(events, inst) == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
