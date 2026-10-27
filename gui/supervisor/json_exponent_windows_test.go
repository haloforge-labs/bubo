//go:build windows

package supervisor

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestJSONExponentRealStatusView(t *testing.T) {
	root := filepath.Join("..", "tests", "scenarios", "recordings", "json-exponent")
	for _, file := range []string{"status-running.stdout.json", "status-watch.stdout.jsonl", "status-context-200000.derived.json"} {
		t.Run(file, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(root, file))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(raw, []byte("8.779e+10")) {
				t.Fatal("real exponent memory value missing")
			}
			for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
				event, err := ParseEvent(line)
				if err != nil {
					t.Fatal(err)
				}
				if event.Type != "status" {
					continue
				}
				st := event.Value.(*Status)
				want := uint64(4096)
				if strings.Contains(file, "derived") {
					want = 200000
				}
				if st.PiSnippet.Providers.Bubo.Models[0].ContextWindow != want {
					t.Fatal("context lost", st.PiSnippet)
				}
				var builder ViewBuilder
				view := builder.Build(ViewInput{Snapshot: Snapshot{Initialized: true, State: Ready, Prepared: true, Instance: &st.Instances[0], Status: st}, Saved: DefaultSettings()})
				if !view.Serving || view.Failure != nil || view.Connection == nil || view.Connection.ContextTokens == nil || *view.Connection.ContextTokens != want {
					t.Fatalf("running view lost: %+v", view)
				}
				if strings.Contains(view.PiSnippet, "2e+05") || !strings.Contains(view.PiSnippet, `"contextWindow": `+fmtUint(want)) {
					t.Fatal("Pi copy text", view.PiSnippet)
				}
				if dir := os.Getenv("BUBO_START_WORKFLOW_EVIDENCE"); dir != "" {
					data, _ := json.MarshalIndent(view, "", "  ")
					if err := os.WriteFile(filepath.Join(dir, file+".view.json"), data, 0600); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
}

func fmtUint(n uint64) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestServerStartupFreshness(t *testing.T) {
	for _, phase := range []string{"starting", "loading"} {
		t.Run(phase, func(t *testing.T) {
			var builder ViewBuilder
			view := builder.Build(ViewInput{Snapshot: Snapshot{Initialized: true, State: Starting, Status: &Status{Phase: phase, SampleTime: time.Now().Add(-time.Minute).UTC().Format(time.RFC3339)}, Stale: true, StatusFeed: StatusFeed{State: StatusReconnecting}, StatusRecordAt: time.Now().Add(-time.Minute), StatusStaleTimeout: 5 * time.Second}, Saved: DefaultSettings()})
			if view.Stale {
				t.Fatal("normal startup must not present stale status")
			}
			if view.StatusFeed.State != StatusWaiting {
				t.Fatal("startup must be waiting", view.StatusFeed)
			}
		})
	}
	// An actual running server whose last accepted sample stopped arriving must
	// retain the stale signal; suppressing startup must not hide feed failures.
	var builder ViewBuilder
	view := builder.Build(ViewInput{Snapshot: Snapshot{Initialized: true, State: Ready, Stale: true}, Saved: DefaultSettings()})
	if !view.Stale {
		t.Fatal("running stale status was hidden")
	}
}

func TestJSONExponentAllEventFamilies(t *testing.T) {
	files, err := filepath.Glob(filepath.Join(contractDir(), "valid", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			raw, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			var doc map[string]any
			decoder := json.NewDecoder(bytes.NewReader(raw))
			decoder.UseNumber()
			if err := decoder.Decode(&doc); err != nil {
				t.Fatal(err)
			}
			if doc["type"] == nil {
				return
			}
			if _, err := ParseEvent(raw); err != nil {
				t.Fatalf("base fixture: %v", err)
			}
			var exponentNumbers func(any) any
			exponentNumbers = func(v any) any {
				switch x := v.(type) {
				case json.Number:
					if !strings.ContainsAny(x.String(), ".eE") {
						return json.RawMessage(x.String() + "e+00")
					}
				case map[string]any:
					for k, item := range x {
						x[k] = exponentNumbers(item)
					}
				case []any:
					for i, item := range x {
						x[i] = exponentNumbers(item)
					}
				}
				return v
			}
			raw, err = json.Marshal(exponentNumbers(doc))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ParseEvent(raw); err != nil {
				t.Fatal(err)
			}
		})
	}
}
