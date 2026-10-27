//go:build windows

package supervisor

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func actionTestInput(state State) ViewInput {
	fit := true
	return ViewInput{Snapshot: Snapshot{State: state, Initialized: true, Prepared: true,
		Status: &Status{Readiness: Readiness{Runtime: true, Model: true}, MemoryEstimate: &MemoryEstimate{Fits: &fit}}}}
}

func TestLifecycleSingleRuleTrayAndView(t *testing.T) {
	states := []State{NotPrepared, Preparing, Starting, Ready, Stopping, Stopped, Failed, AttachedExternal}
	count := 0
	for _, state := range states {
		for _, managed := range []bool{false, true} {
			for _, supported := range []bool{false, true} {
				for _, readiness := range []string{"ready", "runtime", "model", "missing"} {
					for _, fit := range []bool{false, true} {
						t.Run(fmt.Sprintf("%s/%t/%t/%s/%t", state, managed, supported, readiness, fit), func(t *testing.T) {
							in := actionTestInput(state)
							in.Snapshot.Managed, in.Snapshot.ManagementSupported = managed, supported
							in.Snapshot.Status.MemoryEstimate.Fits = &fit
							if readiness == "runtime" {
								in.Snapshot.Status.Readiness.Runtime = false
							}
							if readiness == "model" {
								in.Snapshot.Status.Readiness.Model = false
							}
							if readiness == "missing" {
								in.Snapshot.Status = nil
							}
							v := (&ViewBuilder{}).Build(in)
							menu := buildTrayMenu(in.Snapshot, v.Actions)
							for _, item := range menu.Items {
								var want *ActionAvailability
								switch item.Action {
								case "start":
									want = &v.Actions.Start
								case "stop":
									want = &v.Actions.Stop
								case "restart":
									want = &v.Actions.Restart
								}
								if want != nil && item.Enabled != want.Allowed {
									t.Fatal("tray disagrees", item, want)
								}
							}
							if v.Actions.Start.Allowed && (state != Stopped && state != Failed || !fit || readiness != "ready") {
								t.Fatal("unsafe start", v.Actions)
							}
							if v.Actions.Restart.Allowed && (state != Ready && !(state == AttachedExternal && managed && supported) || state != Ready && !fit || readiness != "ready") {
								t.Fatal("unsafe restart", v.Actions)
							}
							if state == Ready && readiness == "ready" && (!v.Actions.Restart.Allowed || v.Actions.Restart.ReasonCode != "fit_unchecked" || !strings.Contains(v.Actions.Restart.Reason, "Memory is checked when it restarts")) {
								t.Fatal("owned running allocation blocked Restart", v.Actions)
							}
							for _, a := range []ActionAvailability{v.Actions.Start, v.Actions.Stop, v.Actions.Restart} {
								if !a.Allowed && (a.ReasonCode == "" || a.Reason == "") {
									t.Fatal("missing typed reason", a)
								}
							}
						})
						count++
					}
				}
			}
		}
	}
	t.Logf("lifecycle single-rule cases: %d passed", count)
}

func TestLifecycleSavedVerdict(t *testing.T) {
	in := actionTestInput(Stopped)
	in.Saved.Context = 777
	in.Snapshot.Status.MemoryEstimate.ContextTokens = 888
	if got := lifecycleActions(in); got.Start.Allowed || got.Start.ReasonCode != "estimate_unavailable" {
		t.Fatal("borrowed another setup's verdict", got)
	}
	fit := false
	reason := "CLI saved-setup refusal"
	in.SavedEstimate = &MemoryEstimate{Fits: &fit, FitReason: &reason}
	if got := lifecycleActions(in).Start; got.Allowed || got.Reason != reason {
		t.Fatal(got)
	}
	fit = true
	if got := lifecycleActions(in).Start; !got.Allowed {
		t.Fatal(got)
	}
	in.Window.Closing = true
	if got := lifecycleActions(in); got.Start.Allowed || got.Stop.Allowed || got.Restart.Allowed {
		t.Fatal(got)
	}
}

func TestLifecycleDispatchUsesPublishedAvailability(t *testing.T) {
	for _, state := range []State{NotPrepared, Preparing, Starting, Ready, Stopping, Stopped, Failed, AttachedExternal} {
		for _, action := range []string{"start", "stop", "restart"} {
			t.Run(string(state)+"/"+action, func(t *testing.T) {
				b := &fakeBackend{view: actionTestInput(state).Snapshot}
				a := newTestApp(b, DefaultSettings(), nil, "")
				v := a.GetView()
				availability := map[string]ActionAvailability{"start": v.Actions.Start, "stop": v.Actions.Stop, "restart": v.Actions.Restart}[action]
				err := a.runAction(action, "")
				if availability.Allowed {
					if err != nil || len(b.actions) != 1 || b.actions[0] != action {
						t.Fatal("allowed action not delegated", err, b.actions)
					}
				} else if result := ActionResultFromError(err); result.Error == nil || result.Error.Code != availability.ReasonCode || len(b.actions) != 0 {
					t.Fatal("blocked action bypassed View rule", result, b.actions)
				}
			})
		}
	}
}

func TestOwnedRowsCarryPreferenceWriterAndCLILabel(t *testing.T) {
	registry := fixtureEvent(t, "switches").Value.(*Switches)
	prefs := map[string]bool{}
	typ := reflect.TypeOf(PreferenceValues{})
	for i := 0; i < typ.NumField(); i++ {
		prefs[strings.Split(typ.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	in := actionTestInput(Ready)
	in.Registry = registry
	in.Snapshot.Instance = &Instance{}
	in.Saved = Settings{Port: 19321, Profile: "future-profile", Context: 999, Sessions: 7, NPUPrefill: "future-npu", LanAccess: true}
	in.Snapshot.Status.Endpoints = &Endpoints{OpenAIBaseURL: "http://127.0.0.1:19320/from-cli"}
	for _, arg := range serveArguments(in.Saved, requestedHost(in.Saved), in.Saved.Port) {
		if arg.preference == "" {
			continue
		}
		if !prefs[arg.preference] {
			t.Fatal("unknown preference owner", arg)
		}
		for _, row := range registry.Switches {
			if row.Name == arg.name && arg.value != "" {
				in.Snapshot.Status.EffectiveSwitches = append(in.Snapshot.Status.EffectiveSwitches, EffectiveSwitch{Name: row.Name, Effective: json.RawMessage(`"different"`)})
			}
		}
	}
	v := (&ViewBuilder{}).Build(in)
	for _, arg := range serveArguments(in.Saved, requestedHost(in.Saved), in.Saved.Port) {
		for _, row := range v.SwitchOverrides {
			if row.Name == arg.name && arg.preference != "" && (row.WriteVia != "preference" || row.PreferenceField != arg.preference) {
				t.Fatal(row)
			}
		}
	}
	for _, change := range v.Pending {
		found := false
		for _, arg := range serveArguments(in.Saved, requestedHost(in.Saved), in.Saved.Port) {
			for _, row := range registry.Switches {
				if arg.preference == change.Key && arg.name == row.Name {
					found = true
					if change.Label != row.Label {
						t.Fatal("label not from CLI", change, row.Label)
					}
				}
			}
		}
		if !found {
			t.Fatal("pending change lacks registry owner", change)
		}
	}
	if len(v.Pending) < 6 {
		t.Fatal("lost preference restart notices", v.Pending)
	}
	if len(v.Saved.SwitchOverrides) != 0 {
		t.Fatal("duplicated preference as override", v.Saved.SwitchOverrides)
	}
}

func TestVisibilityUsesOnlyRegistrySecretAndEnvEcho(t *testing.T) {
	for _, tc := range []struct {
		name, kind         string
		secret, echo, hide bool
	}{
		{"--max-tokens", "flag", false, false, false},
		{"--ordinary", "flag", true, true, true},
		{"FUTURE_ENV", "env", false, false, true},
		{"PASSWORD_TOKEN", "env", false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const value = "unique-visible-or-hidden-value"
			row := Switch{Name: tc.name, Kind: tc.kind, Secret: tc.secret, Echo: tc.echo, GUIControl: true}
			raw := json.RawMessage(`{"requested":"` + value + `","effective":"` + value + `","source":"test"}`)
			effective := map[string]json.RawMessage{row.Name: raw}
			in := actionTestInput(Ready)
			in.Registry = &Switches{Switches: []Switch{row}}
			in.Saved.SwitchOverrides = map[string]string{row.Name: value}
			in.Snapshot.Instance = &Instance{Effective: effective}
			in.Snapshot.Discovered = []Instance{*in.Snapshot.Instance}
			in.Snapshot.Status.Effective = effective
			in.Snapshot.Status.EffectiveSwitches = []EffectiveSwitch{{Name: row.Name, Effective: json.RawMessage(`"` + value + `"`)}}
			data, _ := json.Marshal((&ViewBuilder{}).Build(in))
			if strings.Contains(string(data), value) == tc.hide {
				t.Fatalf("visibility policy mismatch: hidden=%t", tc.hide)
			}
			if tc.hide && !strings.Contains(string(data), "[hidden]") {
				t.Fatal("missing redaction")
			}
		})
	}
	removed := Switch{Name: "FUTURE_REMOVED", Kind: "env", Class: "REMOVED", Echo: true, GUIControl: true}
	registry := &Switches{Switches: []Switch{removed}}
	if _, err := ApplySwitchOverrides(nil, SwitchOverridesPatch{Set: []SwitchOverride{{Name: removed.Name, Value: "x"}}}, registry); err == nil {
		t.Fatal("removed row writable")
	}
	if _, _, err := switchCommand(map[string]string{removed.Name: "x"}, registry); err == nil {
		t.Fatal("removed row launched")
	}
	contextTokens := uint64(1234)
	registry.Switches = []Switch{{Name: "--context", Class: "REMOVED"}}
	b := &registryBackend{reg: registry}
	a := newTestApp(b, DefaultSettings(), nil, "")
	if result := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{Context: &contextTokens}}); result.Error == nil || a.GetView().Saved.Context != nil {
		t.Fatal("removed preference row writable", result)
	}
}

func TestLANAddressesRequireCLIEndpointAndUseItsPortAndPath(t *testing.T) {
	adapter := netAdapter{Name: "synthetic", Up: true, HasGateway: true, IPv4: []net.IP{net.ParseIP("192.168.7.8")}}
	for _, endpoint := range []string{"", "http://0.0.0.0:0/cli-path"} {
		if got := buildAddresses([]netAdapter{adapter}, endpoint); len(got) != 0 {
			t.Fatal("invented bound port", got)
		}
	}
	got := buildAddresses([]netAdapter{adapter}, "https://0.0.0.0:19437/cli-path")
	if len(got) != 1 || got[0].URL != "https://192.168.7.8:19437/cli-path" {
		t.Fatal(got)
	}
	a := newTestApp(&fakeBackend{}, Settings{LanAccess: true}, nil, "")
	a.adapters = fakeAdapters{list: []netAdapter{adapter}}
	if got := a.GetView().Key; len(got.Addresses) != 0 || got.AddressesUnavailable == "" {
		t.Fatal("stopped setup invented an address", got)
	}
	listener := "http://0.0.0.0:19437"
	snapshot := Snapshot{Instance: &Instance{Endpoint: &listener}, Status: &Status{Endpoints: &Endpoints{OpenAIBaseURL: "http://127.0.0.1:19437/sdk"}}}
	lan := a.lanState(snapshot, Settings{LanAccess: true})
	if !lan.LanActive || len(lan.Addresses) != 1 || lan.Addresses[0].URL != "http://192.168.7.8:19437/sdk" {
		t.Fatal("SDK host replaced bind fact", lan)
	}
}

func TestSavedEstimateCacheKeepsSavedRequestsSeparateFromDrafts(t *testing.T) {
	m := newModelsApp(t, "")
	// A recorded package and a recorded request (estimate-flash-c65536-s1.json);
	// the draft is another recorded request of the same package
	// (estimate-flash-c262144-s2.json). The fake has no answer for a folder or a
	// request that the candidate was not asked.
	folder := e21Folder(t, e21Flash)
	saved := Settings{ManifestPath: filepath.Join(folder, "bubo-package.json"), Context: 65536, Sessions: 1}
	st := actionTestInput(Stopped).Snapshot.Status
	st.MemoryEstimate.ContextTokens = 65536
	st.MemoryEstimate.Sessions = 1
	value, err := m.a.savedEstimate(saved, st)
	if value != nil || err != nil {
		t.Fatal("estimate did not start asynchronously", value, err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		value, err = m.a.savedEstimate(saved, st)
		if value != nil || err != nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if value == nil || err != nil {
		t.Fatal("saved estimate unavailable", err)
	}
	data, readErr := os.ReadFile(filepath.Join(m.dir, "models-argv-1.json"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	var args []string
	if json.Unmarshal(data, &args) != nil || !reflect.DeepEqual(args, []string{"models", "estimate", "--model", folder, "--json", "--context", "65536", "--sessions", "1"}) {
		t.Fatal("saved inputs were not passed to CLI", string(data))
	}
	if result := m.a.Estimate(folder, 262144, 2); result.Error != nil {
		t.Fatal(result)
	}
	if cached, err := m.a.savedEstimate(saved, st); err != nil || cached != value {
		t.Fatal("draft replaced saved verdict", err)
	}
	m.a.mu.Lock()
	m.a.estimate.at = time.Now().Add(-2 * savedEstimateMaxAge)
	m.a.mu.Unlock()
	if cached, _ := m.a.savedEstimate(saved, st); cached != nil {
		t.Fatal("expired machine-budget verdict reused")
	}
	saved.Profile = "unapplied-profile"
	if _, err := m.a.savedEstimate(saved, st); err == nil || err.Code != "estimate_needs_restart" {
		t.Fatal("profile unsupported by CLI estimate was silently accepted", err)
	}
}

func TestSavedSetupNeverBorrowsAnEstimateAcrossRuntimeOrClearedRequests(t *testing.T) {
	in := actionTestInput(Stopped)
	in.Saved.RuntimePath = `C:\new-runtime`
	in.Snapshot.Status.Paths = map[string]json.RawMessage{"runtime": json.RawMessage(`"C:\\previous-runtime"`)}
	if got := lifecycleActions(in).Start; got.Allowed {
		t.Fatal("borrowed old runtime verdict", got)
	}
	in.Saved.RuntimePath = ""
	in.Snapshot.RunningSwitchOverrides = map[string]string{"FUTURE_ENV": "old"}
	if got := lifecycleActions(in).Start; !got.Allowed {
		t.Fatal("borrowed verdict after override clear", got)
	}
	in.Snapshot.RunningSwitchOverrides = nil
	b := &fakeBackend{view: in.Snapshot}
	a := newTestApp(b, DefaultSettings(), nil, "")
	a.appliedSettings = &Settings{Context: 65536}
	a.appliedGeneration = in.Snapshot.Generation
	if got := a.GetView().Actions.Start; !got.Allowed {
		t.Fatal("borrowed verdict after preference clear", got)
	}
	in.Saved.NPUPrefill = "next"
	in.Snapshot.Status.EffectiveSwitches = []EffectiveSwitch{{Name: "--npu-prefill", Effective: json.RawMessage(`"previous"`)}}
	if got := lifecycleActions(in).Start; got.Allowed {
		t.Fatal("borrowed prior NPU setup verdict", got)
	}
}

func TestAutostartWaitsForSavedSetupEstimate(t *testing.T) {
	m := newModelsApp(t, "")
	m.b.view = actionTestInput(Stopped).Snapshot
	m.b.view.Status.MemoryEstimate.ContextTokens = 65536
	m.a.settings = Settings{StartServerWithGUI: true, ModelPath: filepath.Join(m.dir, "saved.gguf"), Context: 32768, Sessions: 1}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.a.Startup(ctx)
	defer m.a.Shutdown(ctx)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		m.b.mu.Lock()
		started := len(m.b.actions) == 1 && m.b.actions[0] == "start"
		m.b.mu.Unlock()
		if started {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("autostart discarded the pending saved-setup estimate", m.a.GetView().Notice)
}
