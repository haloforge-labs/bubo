//go:build windows

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bubo/gui/supervisor"
)

func mvpConfig(t *testing.T, fault string) Config {
	t.Helper()
	recording, err := filepath.Abs("../scenarios/recordings/b2/owner-budget-status.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	c := fakeConfig(t, "normal", "mvp", map[string]string{"BUBO_FAKE_MVP_RECORDING": recording, "BUBO_FAKE_MVP_FAULT": fault, "BUBO_FAKE_WATCH_MS": "500", "BUBO_FAKE_REQUEST_MS": "2800"})
	t.Setenv("BUBO_FAKE_PACKAGES", "")
	c.MVP, c.Fake = true, true
	c.PollInterval = 250 * time.Millisecond
	c.QueryTimeout = 4 * time.Second
	return c
}

func TestMVPFakePass(t *testing.T) {
	c := mvpConfig(t, "")
	var console bytes.Buffer
	if code := run(context.Background(), c, &console); code != ExitPass {
		t.Fatalf("code=%d\n%s", code, console.String())
	}
	r := loadReport(t, c)
	if r.Overall != "DRY RUN PASS" || r.Mode != "mvp-fake" || len(r.Steps) != 6 || len(r.PIDs) != 2 {
		t.Fatalf("bad result: %+v", r)
	}
	for _, s := range r.Steps {
		if s.Status != Pass {
			t.Error(s.Verdict)
		}
	}
	for _, p := range r.PIDs {
		if p.AliveAtEnd == nil || *p.AliveAtEnd || isLive(p.PID, p.CreationTime) {
			t.Errorf("live identity: %+v", p)
		}
	}
	if err := portFree(c.Port); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(c.Out, "views.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	count, sawTwo := 0, false
	for scanner.Scan() {
		var envelope struct {
			View supervisor.View `json:"view"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &envelope); err != nil {
			t.Fatal(err)
		}
		v := envelope.View
		if v.Source != "fixture" {
			t.Fatalf("fake mislabelled: %s", v.Source)
		}
		sawTwo = sawTwo || busySlots(v) == 2
		count++
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if !sawTwo || float64(count) != dataOf(r, 6)["view_count"].(float64) {
		t.Fatalf("saved Views=%d sawTwo=%t", count, sawTwo)
	}
	args, err := os.ReadFile(filepath.Join(os.Getenv("BUBO_FAKE_DIR"), "mvp-serve-args.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(args, []byte(`"--context"`)) || bytes.Contains(args, []byte(`"--profile"`)) {
		t.Fatalf("injected default: %s", args)
	}
	if !strings.Contains(console.String(), "non-null assertion NOT RUN") {
		t.Fatal("fake bypass not explicit")
	}
}

func TestMVPFailuresStayFailures(t *testing.T) {
	for _, test := range []struct{ fault, check string }{{"missing-endpoints", "CLI OpenAI and Anthropic endpoints"}, {"wrong-context", "CLI resolved context without override"}, {"watch-silence", "status cadence every gap <=2s"}} {
		t.Run(test.fault, func(t *testing.T) {
			c := mvpConfig(t, test.fault)
			var console bytes.Buffer
			if code := run(context.Background(), c, &console); code != ExitFail {
				t.Fatalf("code=%d\n%s", code, console.String())
			}
			r := loadReport(t, c)
			found := false
			for _, s := range r.Steps {
				for _, check := range s.Checks {
					found = found || (check.Name == test.check && !check.OK)
				}
			}
			if !found {
				t.Fatalf("missing failed %s\n%s", test.check, console.String())
			}
			for _, p := range r.PIDs {
				if isLive(p.PID, p.CreationTime) {
					t.Errorf("left live PID %d", p.PID)
				}
			}
		})
	}
}

func TestContextResolutionOK(t *testing.T) {
	const step, minTokens, maxTokens = 8192, 4096, 262144
	for _, c := range []struct {
		name                      string
		effective, snapshot, serv uint64
		haveServed, want          bool
	}{
		{"equal", 131072, 131072, 0, false, true},
		{"one step apart", 139264, 131072, 0, false, true},
		{"one step apart, below", 122880, 131072, 0, false, true},
		{"two steps apart", 147456, 131072, 0, false, false},
		{"serve line differs from snapshot, equals effective", 139264, 131072, 139264, true, true},
		{"serve line differs from effective", 131072, 131072, 139264, true, false},
		{"serve line equal but far from snapshot", 196608, 131072, 196608, true, true},
		{"not a multiple of the step", 131073, 131073, 0, false, false},
		{"above the range", 270336, 270336, 0, false, false},
		{"zero", 0, 0, 0, false, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := contextResolutionOK(c.effective, c.snapshot, step, minTokens, maxTokens, c.serv, c.haveServed); got != c.want {
				t.Fatalf("effective=%d snapshot=%d served=%d/%v: got %v want %v", c.effective, c.snapshot, c.serv, c.haveServed, got, c.want)
			}
		})
	}
	if contextResolutionOK(131072, 131072, 0, 0, 0, 0, false) {
		t.Fatal("a table without step_tokens must not pass")
	}
}

func TestResolvedContextFromLogs(t *testing.T) {
	dir := t.TempDir()
	if _, ok := resolvedContextFromLogs(dir); ok {
		t.Fatal("empty log folder reported a number")
	}
	if err := os.WriteFile(filepath.Join(dir, "stderr.0.log"), []byte("x\nResolved 131072 tokens per session from CLI recommended_context.by_sessions for 2 sessions\nResolved 139264 tokens per session from CLI recommended_context.by_sessions for 2 sessions\ny\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if n, ok := resolvedContextFromLogs(dir); !ok || n != 139264 {
		t.Fatalf("got %d %v, want the last line 139264", n, ok)
	}
}

func TestMVPStrictRatesCadenceAndIsolation(t *testing.T) {
	zero := 0.0
	v := supervisor.View{Rates: &supervisor.Rates{Source: "cli", PrefillTPS: &zero, DecodeTPS: &zero}}
	if !hasLiveRates(v) {
		t.Fatal("measured zero is available")
	}
	v.Rates.DecodeTPS = nil
	if hasLiveRates(v) {
		t.Fatal("null accepted for real run")
	}
	v.Rates.DecodeTPS = &zero
	v.Rates.Source = "client_delta"
	if hasLiveRates(v) {
		t.Fatal("client fallback qualified CLI rate")
	}
	start := time.Now()
	cad := cadence{last: start}
	cad.observe(start.Add(2100*time.Millisecond), start.Add(2100*time.Millisecond))
	cad.observe(start.Add(2200*time.Millisecond), start.Add(2200*time.Millisecond))
	if !cad.bad || cad.records != 2 {
		t.Fatal("slow gap averaged away")
	}
	for _, ep := range []string{"http://127.0.0.1:8086/v1", "https://example.com:18120/v1", "http://127.0.0.1:18120/v1?q=1", "http://user@127.0.0.1:18120/v1"} {
		if smokeEndpoint(ep, 18120) {
			t.Fatalf("unsafe endpoint %s", ep)
		}
	}
	if !smokeEndpoint("http://127.0.0.1:18120/arbitrary-cli-path", 18120) {
		t.Fatal("CLI path must remain opaque")
	}
	c := mvpConfig(t, "")
	c.Fake = false
	if c.validate() == nil {
		t.Fatal("real mode without package accepted")
	}
}

func TestMVPInterruptedStillCleansUp(t *testing.T) {
	c := mvpConfig(t, "")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	timer := time.AfterFunc(1500*time.Millisecond, cancel)
	defer timer.Stop()
	var console bytes.Buffer
	if code := run(ctx, c, &console); code != ExitInterrupted {
		t.Fatalf("code=%d\n%s", code, console.String())
	}
	r := loadReport(t, c)
	if status(r, 6) != Pass || !r.Interrupt || len(r.PIDs) == 0 {
		t.Fatalf("no interrupt cleanup proof\n%s", console.String())
	}
	for _, p := range r.PIDs {
		if isLive(p.PID, p.CreationTime) {
			t.Errorf("orphan %d", p.PID)
		}
	}
}
