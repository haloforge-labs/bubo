package supervisor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestContractFixtures(t *testing.T) {
	root := filepath.Join("..", "..", "tests", "cli", "contract")
	for _, dir := range []string{"valid", "invalid"} {
		files, err := filepath.Glob(filepath.Join(root, dir, "*.json"))
		if err != nil || len(files) == 0 {
			t.Fatal("no contract fixtures", err)
		}
		for _, file := range files {
			t.Run(dir+"/"+filepath.Base(file), func(t *testing.T) {
				b, err := os.ReadFile(file)
				if err != nil {
					t.Fatal(err)
				}
				_, err = ParseEvent(b)
				if (err == nil) != (dir == "valid") {
					t.Fatalf("%s: %v", dir, err)
				}
			})
		}
	}
}
func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "tests", "cli", "contract", "valid", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func TestStrictParser(t *testing.T) {
	for _, mutation := range []string{"major", "missing", "null", "range", "time", "utf8", "extra", "duplicate"} {
		t.Run(mutation, func(t *testing.T) {
			b := fixture(t, "progress")
			var o map[string]any
			json.Unmarshal(b, &o)
			p := o["payload"].(map[string]any)
			switch mutation {
			case "major":
				o["schema_version"] = 2
			case "missing":
				delete(o, "seq")
			case "null":
				p["bytes_done"] = nil
			case "range":
				p["bytes_done"] = 1e20
			case "time":
				o["time"] = "2026-02-30T00:00:00Z"
			}
			b, _ = json.Marshal(o)
			switch mutation {
			case "utf8":
				b = append(b, 0xff)
			case "extra":
				b = append(b, []byte(" {}")...)
			case "duplicate":
				b = bytes.Replace(b, []byte(`"schema_version":1`), []byte(`"schema_version":2,"schema_version":1`), 1)
			}
			if _, err := ParseEvent(b); err == nil {
				t.Fatal("invalid record accepted")
			}
		})
	}
}
func TestAdditiveAndNull(t *testing.T) {
	b := fixture(t, "progress-unknown")
	e, err := ParseEvent(b)
	if err != nil {
		t.Fatal(err)
	}
	if e.Value.(*Progress).Total != nil {
		t.Fatal("null became zero")
	}
	var o map[string]any
	json.Unmarshal(fixture(t, "ready"), &o)
	o["future"] = true
	o["type"] = "future-event"
	b, _ = json.Marshal(o)
	e, err = ParseEvent(b)
	if err != nil || e.Value != nil {
		t.Fatal("additive event", err)
	}
}
func TestCounterReset(t *testing.T) {
	b := CounterBaseline{}
	for n, tc := range []struct {
		id    string
		v     uint64
		reset bool
		delta uint64
	}{{"a", 10, true, 0}, {"a", 12, false, 2}, {"a", 1, true, 0}, {"b", 100, true, 0}} {
		d, r := b.Delta(tc.id, map[string]uint64{"tokens": tc.v})
		if r != tc.reset || d["tokens"] != tc.delta {
			t.Fatal(n, d, r)
		}
	}
}

func TestNullCountersStayUnavailable(t *testing.T) {
	var c Counters
	if err := json.Unmarshal([]byte(`{"prefill_tokens_total":null,"decode_tokens_total":0,"prefill_seconds_total":null}`), &c); err != nil {
		t.Fatal(err)
	}
	if c.PrefillTokens != nil || c.DecodeTokens == nil || *c.DecodeTokens != 0 || c.PrefillSeconds != nil {
		t.Fatal(c)
	}
}
func TestTransitionTable(t *testing.T) {
	seen := map[string]bool{}
	for _, edge := range Transitions() {
		name := fmt.Sprintf("%s/%s/%s", edge.From, edge.Action, edge.To)
		if seen[name] {
			t.Fatal("duplicate transition")
		}
		seen[name] = true
		t.Run(name, func(t *testing.T) {
			s := Snapshot{State: edge.From, LastError: &Failure{Code: "device_lost"}}
			if err := s.transition(edge.Action); err != nil || s.State != edge.To {
				t.Fatal(s, err)
			}
			if s.LastError == nil {
				t.Fatal("transition cleared retained error")
			}
			t.Logf("TRANSITION %s -> %s via %s", edge.From, edge.To, edge.Action)
		})
	}
	s := Snapshot{State: NotPrepared}
	if s.transition("start") == nil {
		t.Fatal("unprepared start permitted")
	}
	t.Logf("TRANSITION_COUNT=%d", len(seen))
}
func TestDrains(t *testing.T) {
	for _, input := range []string{"é水\r\nnext\n", "a\nb\nc", "012345678901\nok\n"} {
		var lines []string
		var complete []bool
		r := &byteReader{b: []byte(input)}
		drain(r, 10, func(b []byte, c bool) { lines = append(lines, string(b)); complete = append(complete, c) })
		switch {
		case strings.HasPrefix(input, "é"):
			if len(lines) != 2 || lines[0] != "é水" {
				t.Fatal(lines)
			}
		case strings.HasPrefix(input, "a"):
			if len(lines) != 3 || complete[2] {
				t.Fatal(lines, complete)
			}
		default:
			if len(lines) != 2 || complete[0] || lines[1] != "ok" {
				t.Fatal(lines, complete)
			}
		}
	}
}

type byteReader struct{ b []byte }

func (r *byteReader) Read(b []byte) (int, error) {
	if len(r.b) == 0 {
		return 0, os.ErrClosed
	}
	b[0] = r.b[0]
	r.b = r.b[1:]
	return 1, nil
}
func TestPortPolicy(t *testing.T) {
	for _, p := range []int{-1, 0, 65536} {
		if ValidatePort(p) == nil {
			t.Fatal(p)
		}
	}
	for _, p := range []int{1, 8081, 8082, 8083, 19091, 65535} {
		if ValidatePort(p) != nil {
			t.Fatal("valid network port rejected", p)
		}
	}
}
