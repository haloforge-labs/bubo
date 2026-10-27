//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixtureEvent(t *testing.T, name string) Event {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "tests", "cli", "contract", "valid", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	e, err := ParseEvent(b)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return e
}

type registryBackend struct {
	fakeBackend
	reg *Switches
}

func (b *registryBackend) Registry() *Switches { return b.reg }

func readySnapshot(t *testing.T, fixture string) Snapshot {
	st := fixtureEvent(t, fixture).Value.(*Status)
	inst := st.Instances[0]
	return Snapshot{Initialized: true, State: Ready, Prepared: true, Instance: &inst, Status: st}
}

func TestGetViewPreservesCLIFieldsAndUnknowns(t *testing.T) {
	a := newTestApp(&fakeBackend{view: readySnapshot(t, "status-observed")}, DefaultSettings(), nil, "")
	v := a.GetView()
	if v.MemoryEstimate != nil || v.EstimateReason == nil || v.Rates != nil || v.PiSnippet != "" {
		t.Fatal("absent CLI facts were invented", v)
	}
	if len(v.Slots) != 1 || len(v.Requests) != 1 || v.ProcessMemory == nil || v.Queue == nil || *v.Queue != 0 {
		t.Fatal("CLI facts were lost", v)
	}
}

func TestGetViewCarriesRegistryAndRestartChanges(t *testing.T) {
	snap := readySnapshot(t, "settings-memory-v1")
	// This fixture predates the core context row; supply a named CLI observation.
	snap.Status.Effective["context"] = json.RawMessage(`{"effective":4096,"source":"test_fixture"}`)
	reg := fixtureEvent(t, "switches").Value.(*Switches)
	settings := DefaultSettings()
	settings.Context = 8192
	a := newTestApp(&registryBackend{fakeBackend: fakeBackend{view: snap}, reg: reg}, settings, nil, "")
	v := a.GetView()
	if v.MemoryEstimate == nil || v.Inference == nil || v.ThinkBias == nil || v.LoopBreaker == nil || len(v.Switches) != len(reg.Switches) {
		t.Fatal("typed CLI facts lost", v)
	}
	found := false
	for _, change := range v.Pending {
		if change.Key == "context" && change.Next == "8192" {
			found = true
		}
	}
	if !found {
		t.Fatal("saved context change missing", v.Pending)
	}
	plain := newTestApp(&fakeBackend{view: snap}, settings, nil, "").GetView()
	if len(plain.Switches) != 0 {
		t.Fatal("registry invented")
	}
}

// Privacy contract: a key must never show on screen. The CLI omits credentials;
// the View follows explicit CLI secret metadata, including synthetic secret flags.
func TestPlantedKeyNeverReachesTheView(t *testing.T) {
	const planted = "sk-PLANTED-SECRET-0123456789"
	snap := readySnapshot(t, "settings-memory-v1")
	snap.Status.Effective["--api-key"] = json.RawMessage(`{"requested":"` + planted + `","effective":"` + planted + `","source":"user","reason":null}`)
	snap.Status.Effective["GUFO_EVAL_API_KEY"] = json.RawMessage(`{"requested":"` + planted + `","effective":null,"source":"env","reason":null}`)
	snap.Instance.Effective = snap.Status.Effective
	snap.Status.Instances[0].Effective = snap.Status.Effective
	snap.Discovered = []Instance{*snap.Instance}
	snap.Status.EffectiveSwitches = append(snap.Status.EffectiveSwitches, EffectiveSwitch{Name: "--api-key", Effective: json.RawMessage(`"` + planted + `"`), Source: "user", Audience: "advanced", Group: "Server"})
	reg := fixtureEvent(t, "switches").Value.(*Switches)
	// This recording predates B5's flag-secret correction. Test the explicit
	// CLI marker synthetically; never infer a secret from its name.
	for i := range reg.Switches {
		if reg.Switches[i].Name == "--api-key" {
			reg.Switches[i].Secret = true
		}
	}

	b := &registryBackend{fakeBackend: fakeBackend{view: snap}, reg: reg}
	a := newTestApp(b, DefaultSettings(), nil, "")
	data, _ := json.Marshal(a.GetView())
	if strings.Contains(string(data), planted) {
		t.Fatal("a planted key reached the frame")
	}
	if !strings.Contains(string(data), "[hidden]") {
		t.Fatal("the hidden row was dropped instead of hidden")
	}
}
