//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bubo/gui/updater"
)

func TestLaunchEstimateFailureDoesNotStartServer(t *testing.T) {
	b := backendFor(t, "split")
	saved := b.settings
	saved.ManifestPath = filepath.Join(e21Folder(t, e21Flash), "bubo-package.json")
	saved.Context, saved.Sessions = 200000, 1
	if err := b.Save(saved); err != nil {
		t.Fatal(err)
	}
	t.Setenv("BUBO_FAKE_MODE", "runtime-missing")
	reply(t, b.Action("start", saved, ""))
	v := waitBackend(t, b, func(v Snapshot) bool { return v.State == Failed })
	if v.LastError == nil || v.LastError.Code != "estimate_unavailable" || !strings.Contains(v.LastError.Remedy, "could not check memory") {
		t.Fatalf("failed estimate did not explain refusal: %+v", v.LastError)
	}
	if len(serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))) != 0 || v.Instance != nil {
		t.Fatal("estimate failure launched a server")
	}
}

// Failure cases, written before implementation: Ready bypasses a changed Tight
// verdict; consent for the old context leaks into the replacement; a direct or
// automatic restart launches before a fresh, post-stop verdict; no-fit bypasses
// the checkbox; a saved Tight verdict is hidden as "In use".
func TestTightRestartRequiresSetupConsent(t *testing.T) {
	st := e21Events(t, "status-w-idle.json")[0].Value.(*Status)
	est := e21Events(t, "estimate-flash-c262144-s2.json")[0].Value.(*ModelEstimate).MemoryEstimate
	reg := e21Events(t, "switches.json")[0].Value.(*Switches)
	saved := DefaultSettings()
	saved.ManifestPath = filepath.Join(e21Folder(t, e21Flash), "bubo-package.json")
	saved.Context, saved.Sessions = uint64(est.ContextTokens), uint32(est.Sessions)
	b := &registryBackend{fakeBackend: fakeBackend{view: Snapshot{Initialized: true, Prepared: true, State: Ready, Status: st}}, reg: reg}
	a := newTestApp(b, saved, nil, "")
	a.estimate = savedEstimateState{key: "unused", value: est, at: time.Now()}
	in := ViewInput{Snapshot: b.view, Registry: reg, Saved: saved, SavedEstimate: est}
	v := new(ViewBuilder).Build(in)
	if v.Actions.Restart.Allowed || v.Actions.Restart.ReasonCode != "memory_tight" || v.Fit.Class != "memory_tight" {
		t.Fatalf("changed Tight setup bypassed Ready admission: action=%+v fit=%+v", v.Actions.Restart, v.Fit)
	}
	yes := true
	if r := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{AllowTightSetup: &yes}}); r.Error != nil {
		t.Fatal(r.Error)
	}
	in.Saved = a.settings
	if v = new(ViewBuilder).Build(in); !v.Actions.Restart.Allowed || !v.Saved.AllowTightSetup {
		t.Fatal("saved setup consent did not admit restart", v.Actions.Restart)
	}
	changed := uint64(200000) // current default, captured by the merged CLI
	if r := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{Context: &changed}}); r.Error != nil {
		t.Fatal(r.Error)
	}
	if a.settings.AllowTightSetup || a.settings.TightSetupFingerprint != "" {
		t.Fatal("changed context retained old consent")
	}
	in.Saved = a.settings
	if v = new(ViewBuilder).Build(in); v.Actions.Restart.Allowed {
		t.Fatal("changed setup started with the old checkbox")
	}
	noFit := e21Events(t, "estimate-flash-c262144-s4.json")[0].Value.(*ModelEstimate).MemoryEstimate
	in.SavedEstimate, in.Saved = noFit, saved
	in.Saved.Context, in.Saved.Sessions = uint64(noFit.ContextTokens), uint32(noFit.Sessions)
	in.Saved.AllowTightSetup = true
	in.Saved.TightSetupFingerprint, _ = tightSetupFingerprint(in.Saved, st, reg)
	if new(ViewBuilder).Build(in).Actions.Restart.Allowed {
		t.Fatal("no-fit Ready replacement accepted an override")
	}
}

func TestTightRestartChecksMemoryAfterStopping(t *testing.T) {
	for _, automatic := range []bool{false, true} {
		t.Run(map[bool]string{false: "restart", true: "automatic"}[automatic], func(t *testing.T) {
			b := backendFor(t, "split")
			// Run a real supervisor lifecycle, with the fake replaying the actual
			// CLI estimate. A running state and a GPU loss are arranged; the
			// replacement model/context/sessions and its memory numbers are real.
			saved := b.settings
			saved.ManifestPath = filepath.Join(e21Folder(t, e21Flash), "bubo-package.json")
			saved.Context, saved.Sessions = 262144, 2
			if err := b.Save(saved); err != nil {
				t.Fatal(err)
			}
			// The first launch has consent. Clear it after Ready to exercise the
			// launch-time check independently of AppService's action gate.
			st := e21Events(t, "status-main.json")[0].Value.(*Status)
			saved.AllowTightSetup = true
			saved.TightSetupFingerprint, _ = tightSetupFingerprint(saved, st, e21Events(t, "switches.json")[0].Value.(*Switches))
			if err := b.Save(saved); err != nil {
				t.Fatal(err)
			}
			reply(t, b.Action("start", saved, ""))
			waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
			if automatic {
				// Auto-restart retains its arguments, but Save has cleared the
				// saved consent. A historical pin must not resurrect that checkbox.
				saved.Context, saved.Sessions = 200000, 1
				if err := b.Save(saved); err != nil {
					t.Fatal(err)
				}
				script(t, os.Getenv("BUBO_FAKE_DIR"), "75 once")
				waitBackend(t, b, func(v Snapshot) bool {
					return v.State == Failed && v.LastError != nil && v.LastError.Code == "memory_tight"
				})
				if n := len(serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))); n != 1 {
					t.Fatalf("launch count %d", n)
				}
				return
			}
			saved.AllowTightSetup, saved.TightSetupFingerprint = false, ""
			if err := b.Save(saved); err != nil {
				t.Fatal(err)
			}
			// Restart acknowledges the queued Stop immediately. Admission is
			// asynchronous, after the old process exits; verify its terminal
			// state and actual serve launches rather than the queue's reply.
			reply(t, b.Action("restart", saved, ""))
			waitBackend(t, b, func(v Snapshot) bool {
				return v.State == Failed && v.LastError != nil && v.LastError.Code == "memory_tight"
			})
			if n := len(serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))); n != 1 {
				t.Fatalf("blocked restart launched %d times", n)
			}
			if b.Snapshot().Instance != nil {
				t.Fatal("blocked replacement still has an instance")
			}
		})
	}
}

func TestTightRestartSavedCheckboxWorkflow(t *testing.T) {
	b := backendFor(t, "split")
	saved := b.settings
	saved.ManifestPath = filepath.Join(e21Folder(t, e21Flash), "bubo-package.json")
	saved.Context, saved.Sessions = 200000, 1
	if err := b.Save(saved); err != nil {
		t.Fatal(err)
	}
	a := newTestApp(b, saved, nil, "")
	a.AttachModels(b.config.SettingsPath, fakeExe, "")
	t.Cleanup(a.models.Close)
	record := func(name string) View {
		t.Helper()
		v := waitSavedEstimate(t, a)
		want := e21Events(t, "status-main.json")[0].Value.(*Status).Versions
		if v.Product == nil || *v.Product != want {
			t.Fatalf("workflow product metadata is not the recorded CLI: %+v", v.Product)
		}
		if directory := os.Getenv("BUBO_TIGHT_RESTART_EVIDENCE"); directory != "" {
			if err := os.MkdirAll(directory, 0700); err != nil {
				t.Fatal(err)
			}
			raw, err := json.MarshalIndent(v, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(directory, name+".json"), raw, 0600); err != nil {
				t.Fatal(err)
			}
			launches, err := json.MarshalIndent(serveLaunches(t, os.Getenv("BUBO_FAKE_DIR")), "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(directory, name+"-launches.json"), launches, 0600); err != nil {
				t.Fatal(err)
			}
		}
		return v
	}
	record("stopped-original")
	if r := a.Start(); r.Error != nil {
		t.Fatal(r.Error)
	}
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
	record("ready-original")
	context, sessions := uint64(262144), uint32(2)
	if r := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{Context: &context, Sessions: &sessions}}); r.Error != nil {
		t.Fatal(r.Error)
	}
	v := record("ready-tight-blocked")
	if v.Actions.Restart.Allowed || v.Fit.Class != "memory_tight" || v.Saved.AllowTightSetup {
		t.Fatal("Ready Save bypasses Tight", v.Actions.Restart, v.Fit)
	}
	if r := a.Restart(); r.Error == nil || r.Error.Code != "memory_tight" {
		t.Fatal("Restart ignored the saved checkbox", r)
	}
	if n := len(serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))); n != 1 {
		t.Fatalf("blocked setup launched %d times", n)
	}
	yes := true
	if r := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{AllowTightSetup: &yes}}); r.Error != nil {
		t.Fatal(r.Error)
	}
	v = record("ready-tight-allowed")
	if !v.Actions.Restart.Allowed || !v.Saved.AllowTightSetup {
		t.Fatal("checkbox did not admit this setup", v.Actions.Restart)
	}
	if r := a.Restart(); r.Error != nil {
		t.Fatal(r.Error)
	}
	waitBackend(t, b, func(v Snapshot) bool { return v.State == Ready })
	record("ready-tight-restarted")
	if n := len(serveLaunches(t, os.Getenv("BUBO_FAKE_DIR"))); n != 2 {
		t.Fatalf("approved setup launch count %d", n)
	}
	loaded, warning := LoadSettings(b.config.SettingsPath)
	if warning != "" || !tightConsentActive(loaded, b.Snapshot().Status, b.Registry()) {
		t.Fatal("unchanged restart lost consent", warning)
	}
	context = 200000
	if r := a.PatchPreferences(PreferencesPatch{Set: PreferenceValues{Context: &context}}); r.Error != nil {
		t.Fatal(r.Error)
	}
	v = record("ready-changed-cleared")
	if v.Saved.AllowTightSetup {
		t.Fatal("changing the setup retained the checkbox")
	}
}

// A complete refresh-only plan must reach the update card as zero network
// bytes. Missing/mixed file states must never become a reuse promise.
func TestUpdateReusePlanFacts(t *testing.T) {
	var parts []Plan
	for _, e := range e21Events(t, "plan-runtime-refresh.jsonl") {
		if p, ok := e.Value.(*Plan); ok {
			parts = append(parts, *p)
		}
	}
	if len(parts) == 0 {
		t.Fatal("no captured refresh plan")
	}
	for _, state := range []UpdateState{UpdateAvailable, UpdateCurrent} {
		for _, unknown := range []bool{false, true} {
			candidate := append([]Plan(nil), parts...)
			if unknown {
				for i := range candidate {
					candidate[i].FileStates = nil
				}
			}
			st := e21Events(t, "status-main.json")[0].Value.(*Status)
			current := "0.1.0"
			if state == UpdateCurrent {
				current = "0.1.1"
			}
			v := new(ViewBuilder).Build(ViewInput{Snapshot: Snapshot{PlanParts: candidate, Status: st}, Update: Update{State: state, CurrentVersion: current, AvailableVersion: "0.1.1", RuntimeFollowupRequired: state == UpdateCurrent}})
			data, _ := json.Marshal(v.Update)
			var update map[string]json.RawMessage
			_ = json.Unmarshal(data, &update)
			value, present := update["runtime_download_bytes"]
			if unknown && present || !unknown && (!present || string(value) != "0") {
				t.Fatalf("CLI reuse provenance lost or invented: unknown=%v update=%s", unknown, data)
			}
		}
	}
}

func TestUpdateReuseReleaseFacts(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", "release-update", "bubo-release.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest updater.Manifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	for _, changed := range []bool{false, true} {
		manifest.Runtime.Changed, manifest.Models.Changed = changed, changed
		client := &stubUpdateClient{check: updater.CheckResult{Status: updater.Available, Available: &updater.Release{Manifest: manifest, Asset: manifest.Assets[0], Version: manifest.Version, Size: manifest.Assets[0].Size, Notes: manifest.Notes}}}
		u := NewUpdateService(UpdateServiceConfig{Updater: updater.Config{InstalledVersion: "0.1.0"}, Snapshot: saneUpdateSnapshot, Enabled: func() bool { return false }})
		u.factory = func(updater.Config) (updateClient, *updater.Failure) { return client, nil }
		u.Start(readyWindow())
		u.wg.Wait()
		if r := u.CheckUpdate(); r.Error != nil {
			t.Fatal(r.Error)
		}
		if offer := u.State(); offer.CurrentVersion != "0.1.0" || offer.AvailableVersion != "0.1.1" || offer.State != UpdateAvailable {
			t.Fatalf("not a real offer: %+v", offer)
		}
		view, _ := json.Marshal(u.State())
		var fields map[string]json.RawMessage
		_ = json.Unmarshal(view, &fields)
		for _, name := range []string{"runtime_download_bytes", "models_download_bytes"} {
			value, present := fields[name]
			if !changed && (!present || string(value) != "0") || changed && present && string(value) == "0" {
				t.Fatalf("unchanged facts missing or changed facts promised reuse: changed=%v %s", changed, view)
			}
		}
		u.Close()
	}
}
