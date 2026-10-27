//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func viewFixture(t *testing.T, name string) Event {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(contractDir(), "valid", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	e, err := ParseEvent(b)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestViewOwnsEveryMutableInputAndReturnedCollection(t *testing.T) {
	st := viewFixture(t, "settings-memory-v1").Value.(*Status)
	reg := viewFixture(t, "switches").Value.(*Switches)
	models := viewFixture(t, "models").Value.(*Models)
	m := ModelsState{Catalogue: models.Catalogue, Scanned: models.Scanned, Listing: &models.Listing}
	input := ViewInput{Snapshot: Snapshot{State: Ready, Status: st, Rates: &Rates{DecodeTPS: f64(8)}}, Registry: reg, Models: &m, Saved: Settings{ModelFolders: []string{"D:\\Models"}}, Logs: []string{"line"}}
	var builder ViewBuilder
	v := builder.Build(input)
	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	st.Capabilities.Profiles[0].Label = "changed"
	reg.Switches[0].Requires = append(reg.Switches[0].Requires, "changed")
	input.Logs[0] = "changed"
	input.Saved.ModelFolders[0] = "changed"
	*input.Snapshot.Rates.DecodeTPS = 77
	if after, _ := json.Marshal(v); string(after) != string(encoded) {
		t.Fatal("input mutation leaked into a returned View")
	}
	v.Switches[0].Allowed.Kind = "changed"
	v.Saved.ModelFolders[0] = "changed"
	*v.Rates.DecodeTPS = 99
	if *input.Snapshot.Rates.DecodeTPS != 77 || input.Registry.Switches[0].Allowed == nil {
		t.Fatal("view mutation escaped ownership")
	}
	// Concurrent callers share the builder but never share returned collections.
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); a := builder.Build(input); a.Logs[0] = "private" }()
	}
	wg.Wait()
	if input.Logs[0] != "changed" {
		t.Fatal("concurrent return mutation escaped ownership")
	}
}

func TestViewPreservesCLISettingsRegistryAndAbsentFacts(t *testing.T) {
	st := viewFixture(t, "settings-memory-v1").Value.(*Status)
	st.Effective["new-setting"] = json.RawMessage(`{"requested":3,"effective":2,"source":"cli","reason":"limited","allowed":{"minimum":1,"maximum":2}}`)
	reg := viewFixture(t, "switches").Value.(*Switches)
	reg.Switches[0].Label = "CLI label"
	var b ViewBuilder
	v := b.Build(ViewInput{Snapshot: Snapshot{State: Starting, Status: st}, Registry: reg})
	if v.State != Starting || v.Switches[0].Label != "CLI label" || len(v.Profiles) != len(st.Capabilities.Profiles) {
		t.Fatal("state or CLI rows lost")
	}
	var found bool
	for _, r := range v.Settings {
		if r.Name == "new-setting" {
			found = r.Requested.Text == "3" && r.Effective.Text == "2" && r.Source == "cli" && r.Allowed.Properties["maximum"].Text == "2"
		}
	}
	if !found {
		t.Fatal("additive effective setting lost")
	}
	if len(st.Instances) > 0 {
		a := b.Build(ViewInput{Snapshot: Snapshot{State: Ready, Status: st, Instance: &st.Instances[0]}})
		if len(a.Pending) != 0 {
			t.Fatal("absent overrides require a restart", a.Pending)
		}
	}
	if v.Endpoints != nil || v.APIKeyEnv != "" || v.Runtime.Kind != "" || v.Saved.Port != nil || v.Saved.Profile != nil || v.Saved.Context != nil || v.Saved.Sessions != nil {
		t.Fatal("invented an absent CLI fact or user override")
	}
	st.Endpoints = &Endpoints{OpenAIBaseURL: "from-cli", AnthropicBaseURL: "also-from-cli"}
	st.APIKeyEnv = "TEST_CLI_KEY"
	st.Paths["runtime_kind"] = json.RawMessage(`"test-cli-runtime"`)
	st.Paths["runtime"] = json.RawMessage(`"D:\\runtime"`)
	v = b.Build(ViewInput{Snapshot: Snapshot{State: Ready, Status: st}})
	if v.Endpoints.OpenAIBaseURL != "from-cli" || v.Endpoints.AnthropicBaseURL != "also-from-cli" || v.APIKeyEnv != "TEST_CLI_KEY" || v.Runtime.Kind != "test-cli-runtime" || v.Runtime.Paths["runtime"].Text != "D:\\runtime" {
		t.Fatal("CLI runtime or endpoint facts lost")
	}
}

func TestViewOmitsUnavailableGPUFieldsAndRetainsChipPower(t *testing.T) {
	var b ViewBuilder
	v := b.Build(ViewInput{Snapshot: Snapshot{State: Stopped, GPU: &GPUStats{TemperatureC: f64(51), ChipPowerW: f64(37.2)}}})
	raw, _ := json.Marshal(v.GPU)
	s := string(raw)
	if !strings.Contains(s, `"temperature_c":51`) || !strings.Contains(s, `"chip_power_w":37.2`) || strings.Contains(s, "use_pct") || strings.Contains(s, "gpu_power") {
		t.Fatal(s)
	}
}

func TestViewValuePreservesOpenJSONAndNumberSpelling(t *testing.T) {
	v := valueFromJSON(json.RawMessage(`{"future":[true,null,"text",18446744073709551615]}`))
	items := v.Properties["future"].Items
	if v.Kind != "object" || len(items) != 4 || items[0].Boolean == nil || !*items[0].Boolean || items[1].Kind != "null" || items[2].Text != "text" || items[3].Text != "18446744073709551615" {
		t.Fatalf("%+v", v)
	}
}

func TestViewPartialPreparationStaysPendingUntilApproval(t *testing.T) {
	var b ViewBuilder
	s := Snapshot{State: Preparing, PlanParts: []Plan{{Component: "model", Parts: 2, Part: 0, Digest: "fixture"}}}
	v := b.Build(ViewInput{Snapshot: s})
	if !v.PreparationPending || v.PlanError != nil {
		t.Fatal("in-flight plan reported failure", v.PlanError)
	}
	s.Approval = &Approval{Digest: "fixture", Components: []string{"model"}}
	v = b.Build(ViewInput{Snapshot: s})
	if v.PlanError == nil || v.PreparationPending {
		t.Fatal("incomplete approval reported success")
	}
	s.PlanParts = append(s.PlanParts, Plan{Component: "model", Parts: 2, Part: 1, Digest: "fixture"})
	v = b.Build(ViewInput{Snapshot: s})
	if v.PlanError != nil || v.PreparationPending || len(v.Preparation) != 1 {
		t.Fatal("complete plan not published", v.PlanError)
	}
}

func TestApprovalWaitDoesNotClaimFilesAreDownloading(t *testing.T) {
	var b ViewBuilder
	s := Snapshot{Initialized: true, State: Preparing, Status: &Status{}, Approval: &Approval{Digest: "binding", Components: []string{"model", "runtime"}}}
	v := b.Build(ViewInput{Snapshot: s})
	if v.Phase != "needs_engine" || v.StatusLabel != "Setup needed: download the engine files" {
		t.Fatalf("first-start approval claimed a transfer: %s %q", v.Phase, v.StatusLabel)
	}
	s.Status.Readiness.Runtime = true
	s.Approval.Components = []string{"model"}
	v = b.Build(ViewInput{Snapshot: s})
	if v.Phase != "needs_model" || v.StatusLabel != "Setup needed: download a model" {
		t.Fatalf("model approval claimed a transfer: %s %q", v.Phase, v.StatusLabel)
	}
}

func TestViewKeyEchoNeverAppearsInViewLogsOrStreams(t *testing.T) {
	store := newFakeStore()
	key := strings.Repeat("cd", 32)
	store.m[KeyTarget] = key
	s, c := lanSupervisor(t, "leak-key", true, store)
	var mu sync.Mutex
	var streams []string
	s.SetEventSink(func(e StreamEvent) {
		b, _ := json.Marshal(e)
		mu.Lock()
		streams = append(streams, string(b))
		mu.Unlock()
	})
	reply(t, s.Start())
	waitState(t, s, Ready)
	var b ViewBuilder
	v := b.Build(ViewInput{Snapshot: s.Snapshot(), Logs: s.Logs(), Key: KeyView{Present: true, Mask: KeyMask}})
	data, _ := json.Marshal(v)
	if strings.Contains(string(data), key) {
		t.Fatal("key leaked into View")
	}
	for name, text := range artifacts(t, s, c) {
		if strings.Contains(text, key) {
			t.Errorf("key leaked into %s", name)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(streams) == 0 {
		t.Fatal("test saw no ingest events")
	}
	for _, text := range streams {
		if strings.Contains(text, key) {
			t.Fatal("key leaked into stream")
		}
	}
}

func TestSnapshotCopyDoesNotAliasRawContractObjects(t *testing.T) {
	st := viewFixture(t, "settings-memory-v1").Value.(*Status)
	st.MemoryEstimate.ReferenceCoefficient = json.RawMessage(`{"test":1}`)
	in := Snapshot{State: Ready, Status: st, Discovered: st.Instances}
	a := copySnapshot(in)
	before, _ := json.Marshal(in)
	a.Status.Effective["--context"][0] = 'x'
	a.Status.MemoryEstimate.ReferenceCoefficient[0] = 'x'
	if after, _ := json.Marshal(in); string(before) != string(after) {
		t.Fatal("snapshot copy aliases input")
	}
}
