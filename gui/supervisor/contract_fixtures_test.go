package supervisor

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func contractRoot() string { return filepath.Join("..", "..", "tests", "cli", "contract") }

func readFixtureFile(t *testing.T, rel string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(contractRoot(), filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func parseFixture(t *testing.T, rel string) Event {
	t.Helper()
	e, err := ParseEvent(readFixtureFile(t, rel))
	if err != nil {
		t.Fatalf("%s: %v", rel, err)
	}
	return e
}

// TestEmbeddedSchemaMatchesContract fails when tests/cli/contract/event.schema.json
// changes and the supervisor copy has not been refreshed.
func TestEmbeddedSchemaMatchesContract(t *testing.T) {
	b := readFixtureFile(t, "event.schema.json")
	if string(b) != string(contractSchema) {
		t.Fatal("embedded schema drift: run go generate in gui/supervisor")
	}
}

// TestEveryFixtureInTheIndex parses every indexed event fixture. Valid
// fixtures must parse into their typed value; invalid fixtures must be
// rejected with a Protocol error. The table is logged as FIXTURE lines, and
// written to J9_FIXTURE_TABLE when that variable is set.
func TestEveryFixtureInTheIndex(t *testing.T) {
	var index struct {
		SchemaVersion int               `json:"schema_version"`
		Fixtures      map[string]string `json:"fixtures"`
	}
	if err := json.Unmarshal(readFixtureFile(t, "fixture-index.json"), &index); err != nil || index.SchemaVersion != 1 {
		t.Fatal("fixture index", err)
	}
	var names []string
	for name := range index.Fixtures {
		names = append(names, name)
	}
	sort.Strings(names)
	var table strings.Builder
	counts := map[string]int{}
	for _, name := range names {
		dir := strings.SplitN(name, "/", 2)[0]
		if dir != "valid" && dir != "invalid" && dir != "v1-legacy" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			e, err := ParseEvent(readFixtureFile(t, name))
			var protocolErr *ProtocolError
			switch dir {
			case "valid":
				if err != nil {
					t.Fatalf("valid fixture rejected: %v", err)
				}
				if e.Type != "switches" && e.Type != "request_completed" && e.Type != "support_bundle" && e.Value == nil {
					t.Fatalf("%s has no typed value", e.Type)
				}
				fmt.Fprintf(&table, "%s\tparsed ok\t%s\n", name, e.Type)
				counts["valid ok"]++
			case "invalid":
				if err == nil || !errors.As(err, &protocolErr) {
					t.Fatalf("invalid fixture accepted or not a Protocol error: %v", err)
				}
				fmt.Fprintf(&table, "%s\trejected as expected\t%s\n", name, err)
				counts["invalid rejected"]++
			default:
				// Every retained legacy fixture parsed before the C2 validator
				// migration; keep that classification instead of silently
				// treating a newly rejected fixture as an expected older shape.
				if err != nil {
					t.Fatalf("legacy fixture classification changed: %v", err)
				}
				fmt.Fprintf(&table, "%s\tparsed ok\t%s\n", name, e.Type)
				counts["legacy ok"]++
			}
		})
	}
	t.Logf("FIXTURE_COUNTS %v", counts)
	for _, line := range strings.Split(strings.TrimSpace(table.String()), "\n") {
		t.Log("FIXTURE " + line)
	}
	if path := os.Getenv("J9_FIXTURE_TABLE"); path != "" {
		if err := os.WriteFile(path, []byte(table.String()), 0600); err != nil {
			t.Fatal(err)
		}
	}
	// Every file on disk is indexed (nothing is silently skipped).
	for _, dir := range []string{"valid", "invalid"} {
		files, _ := filepath.Glob(filepath.Join(contractRoot(), dir, "*.json"))
		for _, f := range files {
			if _, ok := index.Fixtures[dir+"/"+filepath.Base(f)]; !ok {
				t.Errorf("%s/%s is not in fixture-index.json", dir, filepath.Base(f))
			}
		}
	}
}

func TestInputFixturesMatchSupervisorLines(t *testing.T) {
	schema, err := compileSchema(readFixtureFile(t, "input.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	check := func(b []byte) error {
		var v any
		dec := json.NewDecoder(strings.NewReader(string(b)))
		dec.UseNumber()
		if err := dec.Decode(&v); err != nil {
			return err
		}
		return schema.Validate(v)
	}
	for _, dir := range []string{"input-valid", "input-invalid"} {
		files, _ := filepath.Glob(filepath.Join(contractRoot(), dir, "*.json"))
		if len(files) == 0 {
			t.Fatal("no input fixtures in", dir)
		}
		for _, f := range files {
			b, _ := os.ReadFile(f)
			err := check(b)
			if (err == nil) != (dir == "input-valid") {
				t.Errorf("%s/%s: %v", dir, filepath.Base(f), err)
			}
		}
	}
	// The exact lines the supervisor sends validate against the input schema.
	op, digest := strings.Repeat("1", 32), strings.Repeat("c", 64)
	for _, line := range [][]byte{ApprovalLine("approve", op, digest), ApprovalLine("decline", op, digest), CancelLine(op)} {
		if err := check(line); err != nil {
			t.Errorf("%s: %v", line, err)
		}
	}
}

func TestUnknownAdditiveKeysAreIgnored(t *testing.T) {
	for _, name := range []string{"status-observed", "ready-observed", "status", "progress", "plan", "approval_required", "starting", "request_completed"} {
		t.Run(name, func(t *testing.T) {
			var o map[string]any
			if err := json.Unmarshal(readFixtureFile(t, "valid/"+name+".json"), &o); err != nil {
				t.Fatal(err)
			}
			base, err := ParseEvent(readFixtureFile(t, "valid/"+name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			o["future_envelope_key"] = map[string]any{"a": 1}
			p := o["payload"].(map[string]any)
			p["future_payload_key"] = []any{1, 2}
			for _, k := range []string{"counters", "connection", "identity"} {
				if m, ok := p[k].(map[string]any); ok {
					m["future_nested_key"] = true
				}
			}
			if list, ok := p["instances"].([]any); ok && len(list) > 0 {
				list[0].(map[string]any)["future_instance_key"] = "x"
			}
			if list, ok := p["slots"].([]any); ok && len(list) > 0 {
				list[0].(map[string]any)["future_slot_key"] = "x"
			}
			b, _ := json.Marshal(o)
			got, err := ParseEvent(b)
			if err != nil {
				t.Fatalf("additive keys rejected: %v", err)
			}
			canon := func(v any) string {
				b, _ := json.Marshal(v)
				var generic any
				json.Unmarshal(b, &generic)
				b, _ = json.Marshal(generic) // sorted keys: RawMessage order is irrelevant
				return string(b)
			}
			a, c := canon(base.Value), canon(got.Value)
			if a != c {
				t.Fatalf("additive keys changed the typed value\n%s\n%s", a, c)
			}
		})
	}
}

func TestMajorVersionMismatchIsAVisibleProtocolError(t *testing.T) {
	for _, v := range []any{2, 0, 10, 1.5, "1"} {
		var o map[string]any
		json.Unmarshal(readFixtureFile(t, "valid/status-observed.json"), &o)
		o["schema_version"] = v
		b, _ := json.Marshal(o)
		_, err := ParseEvent(b)
		var p *ProtocolError
		if !errors.As(err, &p) || !strings.HasPrefix(err.Error(), "Protocol error: ") {
			t.Fatalf("schema_version %v: %v", v, err)
		}
		if n, ok := v.(int); ok && n != 0 && !strings.Contains(err.Error(), "unsupported schema_version") {
			t.Fatalf("schema_version %v: message does not name the version: %v", v, err)
		}
	}
}

func TestRequiredFieldsMissingAreRejected(t *testing.T) {
	cases := map[string]func(p map[string]any){
		"sample_time":     func(p map[string]any) { delete(p, "sample_time") },
		"readiness":       func(p map[string]any) { delete(p, "readiness") },
		"live_throughput": func(p map[string]any) { delete(p, "live_throughput") },
		"counter":         func(p map[string]any) { delete(p["counters"].(map[string]any), "decode_tokens_total") },
		"instance flat":   func(p map[string]any) { delete(p["instances"].([]any)[0].(map[string]any), "pid") },
		"request row":     func(p map[string]any) { delete(p["requests"].([]any)[0].(map[string]any), "outcome") },
		"memory unit":     func(p map[string]any) { delete(p["memory"].(map[string]any), "unit") },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			var o map[string]any
			json.Unmarshal(readFixtureFile(t, "valid/status-observed.json"), &o)
			mutate(o["payload"].(map[string]any))
			b, _ := json.Marshal(o)
			if _, err := ParseEvent(b); err == nil {
				t.Fatal("accepted")
			}
		})
	}
}

func TestLifecycleIdentityMustAgree(t *testing.T) {
	for _, name := range []string{"starting", "loading", "ready", "warning", "stopping", "stopped"} {
		e := parseFixture(t, "valid/"+name+".json")
		id, ok := e.PayloadIdentity()
		if !ok || id.ID != e.InstanceID || id.PID != e.PID || id.CreationTime != e.CreationTime || id.ExePath != e.ExePath {
			t.Fatalf("%s: %v %v", name, id, e)
		}
		var o map[string]any
		json.Unmarshal(readFixtureFile(t, "valid/"+name+".json"), &o)
		o["pid"] = 4321
		b, _ := json.Marshal(o)
		if _, err := ParseEvent(b); err == nil {
			t.Fatalf("%s: top-level and payload identity disagree but parsed", name)
		}
	}
}

func TestTypedStatusObserved(t *testing.T) {
	st := parseFixture(t, "valid/status-observed.json").Value.(*Status)
	if st.Phase != "ready" || !st.Readiness.Runtime || st.Readiness.Server == nil || !*st.Readiness.Server {
		t.Fatal(st.Phase, st.Readiness)
	}
	if len(st.Instances) != 1 || st.Instances[0].Identity == nil || st.Instances[0].PID != 8664 || st.Identity == nil || st.Identity.ID != st.Instances[0].ID {
		t.Fatal("identity", st.Instances)
	}
	if st.Counters == nil || *st.Counters.PrefillTokens != 8 || *st.Counters.DecodeTokens != 2 || *st.Counters.PrefillSeconds != 0.002 {
		t.Fatal("counters")
	}
	if len(st.Slots) != 1 || st.Slots[0].State != "idle" || st.Slots[0].RequestID != nil || st.Slots[0].RemainingTokens != nil {
		t.Fatal("slots (null stays null)", st.Slots)
	}
	if len(st.Requests) != 1 || st.Requests[0].ID != "r2" || st.Requests[0].TTFTMS != nil || st.Requests[0].DecodeTPS == nil || *st.Requests[0].DecodeTPS != 500 {
		t.Fatal("requests", st.Requests)
	}
	if u, ok := st.MemoryUse(); !ok || u.RSSBytes == nil || *u.RSSBytes != 10661888 || u.Unit != "bytes" {
		t.Fatal("memory", u)
	}
	if st.MemoryEstimate != nil || st.MemoryEstimateReason == nil || st.InferenceDefaults != nil || st.PiSnippet != nil || st.PreparedModels != nil {
		t.Fatal("unavailable objects became values")
	}
	if st.Connection == nil || st.Connection.ModelID != "bubo-test-only-fake" || st.Connection.ContextTokens == nil || *st.Connection.ContextTokens != 4096 || st.Connection.Authenticated {
		t.Fatal("connection", st.Connection)
	}
	if st.LiveThroughput == nil || st.LiveThroughput.Label != RateLabel || st.LiveThroughput.PrefillTPS != nil {
		t.Fatal("live throughput")
	}
	if st.ModelSelection == nil || st.ModelSelection.Source != "standard_location" {
		t.Fatal("model selection")
	}
	if len(st.StateHistory) != 3 || st.StateHistory[2].Type != "ready" {
		t.Fatal("state history")
	}
	if row, ok := SettingRow(st.Effective, "context"); !ok || string(row.Effective) != "4096" || row.Source != "test_fixture" || row.Reason == nil {
		t.Fatal("setting row", row)
	}
}

func TestTypedSettingsMemoryFixture(t *testing.T) {
	e := parseFixture(t, "valid/settings-memory-v1.json")
	var d *RuntimeDetail
	switch v := e.Value.(type) {
	case *Status:
		d = &v.RuntimeDetail
	case *Lifecycle:
		d = &v.RuntimeDetail
	default:
		t.Fatalf("%T", v)
	}
	me := d.MemoryEstimate
	if me == nil || me.TableVersion != 1 || me.FitVerified || me.ByteUnit != "bytes" || me.DisplayUnit != "GiB" || me.Evidence != "estimated" {
		t.Fatal("memory estimate", me)
	}
	if me.TotalRange.Min.Bytes == 0 || me.TotalRange.Max.Bytes < me.TotalRange.Min.Bytes || me.TotalRange.Min.GiB <= 0 || me.Weights.Bytes == 0 {
		t.Fatal("memory range with units", me.TotalRange)
	}
	if me.CarveBudget != nil || me.HostBudget != nil {
		t.Fatal("budgets are null until observed")
	}
	if d.InferenceDefaults == nil || d.InferenceDefaults.Temperature.Value == nil || d.InferenceDefaults.Temperature.Source == "" || d.InferenceDefaults.MaxTokens.Source == "" {
		t.Fatal("inference defaults", d.InferenceDefaults)
	}
	if d.ThinkMarkerBias == nil || d.LoopBreaker == nil || d.EntropyGate == nil || d.EntropyGate.Available {
		t.Fatal("reasoning controls")
	}
	if len(d.PreparedModels) == 0 || d.PreparedModels[0].DisplayName == "" || d.PreparedModels[0].SizeBytes == 0 {
		t.Fatal("prepared models")
	}
	audiences := map[string]int{}
	for _, row := range d.EffectiveSwitches {
		audiences[row.Audience]++
		if row.Source == "" {
			t.Fatal("switch without source", row.Name)
		}
	}
	if len(d.EffectiveSwitches) == 0 || audiences["user"] == 0 || audiences["advanced"] == 0 || audiences["debug"] == 0 {
		t.Fatal("effective switches by audience", audiences)
	}
}

func TestTypedSwitchesRegistry(t *testing.T) {
	e := parseFixture(t, "valid/switches.json")
	sw := e.Value.(*Switches)
	var raw struct {
		Payload struct {
			Switches []map[string]any `json:"switches"`
		} `json:"payload"`
	}
	json.Unmarshal(readFixtureFile(t, "valid/switches.json"), &raw)
	if len(sw.Switches) == 0 || len(sw.Switches) != len(raw.Payload.Switches) {
		t.Fatal("rows", len(sw.Switches))
	}
	want := map[string]int{}
	for _, r := range raw.Payload.Switches {
		want[r["audience"].(string)]++
	}
	for _, a := range []string{"user", "advanced", "debug"} {
		if got := len(sw.ByAudience(a)); got != want[a] || got == 0 {
			t.Fatalf("audience %s: %d rows, want %d", a, got, want[a])
		}
	}
	off := 0
	for _, r := range sw.Switches {
		if !r.GUIControl {
			off++
		}
	}
	t.Logf("registry rows=%d user=%d advanced=%d debug=%d gui_control=false:%d", len(sw.Switches), want["user"], want["advanced"], want["debug"], off)
}

func TestTypedSupportBundleRequestAndMachine(t *testing.T) {
	b := parseFixture(t, "valid/support_bundle.json").Value.(*SupportBundle)
	if b.Saved || len(b.Files) != 7 || b.TextPreview["versions.json"] == "" || b.SHA256["manifest.json"] == "" || len(b.BundleSHA256) != 64 {
		t.Fatal("bundle", b)
	}
	r := parseFixture(t, "valid/request_completed.json").Value.(*RequestRow)
	if r.ID != "r1" || r.Slot != nil || *r.PromptTokens != 100 || *r.CacheKind != "miss" || r.Outcome != "completed" {
		t.Fatal("request", r)
	}
	m := parseFixture(t, "valid/status-machine.json").Value.(*Status)
	if m.Machine == nil || m.Machine["windows"] == nil || m.Machine["gpu"] == nil {
		t.Fatal("machine is optional but present here", m.Machine)
	}
	u := parseFixture(t, "valid/status-machine-unavailable.json").Value.(*Status)
	if u.Machine == nil {
		t.Fatal("unavailable machine object missing")
	}
	// A status without a machine object stays valid (J5 keys are optional).
	if s := parseFixture(t, "valid/status-observed.json").Value.(*Status); s.Machine != nil {
		t.Fatal("machine invented", s.Machine)
	}
}

func TestApprovalQuestionAndPlanParts(t *testing.T) {
	var o map[string]any
	json.Unmarshal(readFixtureFile(t, "valid/approval_required.json"), &o)
	o["payload"].(map[string]any)["question"] = "Download and install the listed files?"
	b, _ := json.Marshal(o)
	e, err := ParseEvent(b)
	if err != nil || e.Value.(*Approval).Question != "Download and install the listed files?" || e.Value.(*Approval).Digest == "" {
		t.Fatal(err)
	}
	parts := map[string]bool{}
	for _, n := range []string{"plan", "plan-source-part", "plan-license-part", "plan-path-part", "plan-volume-part"} {
		p := parseFixture(t, "valid/"+n+".json").Value.(*Plan)
		parts[p.RecordKind] = true
		if p.Parts == 0 || p.Digest == "" {
			t.Fatal(n)
		}
	}
	if len(parts) != 5 {
		t.Fatal("record kinds", parts)
	}
}

// No fixture carries a non-null pi_snippet, so the typed shape is exercised with
// a snippet built from the documented example. A real key must be rejected.
func TestPiSnippetTypedAndKeyRestricted(t *testing.T) {
	status := func(key string) []byte {
		var o map[string]any
		json.Unmarshal(readFixtureFile(t, "valid/status-observed.json"), &o)
		o["payload"].(map[string]any)["pi_snippet"] = map[string]any{"providers": map[string]any{"bubo": map[string]any{
			"baseUrl": "http://127.0.0.1:19094/v1", "api": "openai-completions", "apiKey": key,
			"models": []any{map[string]any{"id": "m", "name": "m", "contextWindow": 4096}}}}}
		b, _ := json.Marshal(o)
		return b
	}
	for _, key := range []string{"bubo-loopback", "BUBO_API_KEY"} {
		e, err := ParseEvent(status(key))
		if err != nil {
			t.Fatal(key, err)
		}
		st := e.Value.(*Status)
		if st.PiSnippet == nil || st.PiSnippet.Providers.Bubo.APIKey != key || !strings.Contains(PiSnippetText(st.PiSnippet), "openai-completions") {
			t.Fatal("pi snippet", st.PiSnippet)
		}
	}
	if _, err := ParseEvent(status("sk-real-secret-123")); err == nil {
		t.Fatal("a real-looking key was accepted in pi_snippet")
	}
}
