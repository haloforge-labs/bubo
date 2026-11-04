//go:build windows

package supervisor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func waitSavedEstimate(t *testing.T, a *AppService) View {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		v := a.GetView()
		if (v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error != nil) && (v.Models == nil || !v.Models.Refreshing) {
			return v
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("saved setup estimate did not finish")
	return View{}
}

func recordedEstimates(t *testing.T, dir string) [][]string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "models-argv-*.json"))
	if err != nil {
		t.Fatal(err)
	}
	var estimates [][]string
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var args []string
		if err := json.Unmarshal(data, &args); err != nil {
			t.Fatal(err)
		}
		if len(args) > 1 && args[1] == "estimate" {
			estimates = append(estimates, args)
		}
	}
	return estimates
}

func TestSavedEstimateBeforeServerUsesTheActivePackageFolderAndCLIDefaults(t *testing.T) {
	m := newModelsApp(t, "")
	st := actionTestInput(Stopped).Snapshot
	st.Status.MemoryEstimate = nil
	reason := "No active server allocation settings are available."
	st.Status.MemoryEstimateReason = &reason
	// The fake answers for the packages of the recording only (E21).
	manifest := filepath.Join(e21Folder(t, e21Flash), "bubo-package.json")
	other := filepath.Join(m.dir, "other", "package.json")
	raw, _ := json.Marshal(manifest)
	st.Status.Paths = map[string]json.RawMessage{"manifest": raw}
	st.Status.PreparedModels = []PreparedModel{
		{ID: "wrong-first-model", ManifestPath: &other},
		{ID: "cli-prepared-model", ManifestPath: &manifest},
	}
	m.b.view = st
	m.a.settings = Settings{}
	v := waitSavedEstimate(t, m.a)
	if v.MemoryEstimate != nil || v.SavedEstimate.MemoryEstimate == nil || v.SavedEstimate.Error != nil || !v.Actions.Start.Allowed {
		t.Fatal("saved CLI verdict was not shared with setup and Start", v.SavedEstimate, v.Actions)
	}
	estimates := recordedEstimates(t, m.dir)
	if len(estimates) != 1 || !reflect.DeepEqual(estimates[0], []string{"models", "estimate", "--model", filepath.Dir(manifest), "--json"}) {
		t.Fatal("the package folder or the CLI defaults were replaced", estimates)
	}
	for range 3 {
		m.a.GetView()
	}
	if estimates := recordedEstimates(t, m.dir); len(estimates) != 1 {
		t.Fatal("setup projection made a second estimate request", estimates)
	}
	// The generated View copy must own the nested saved-estimate allocation.
	v.SavedEstimate.MemoryEstimate.ContextTokens = 1
	fresh := m.a.GetView()
	if fresh.SavedEstimate.MemoryEstimate.ContextTokens == 1 {
		t.Fatal("saved estimate projection aliases another View")
	}
}

func TestSavedEstimatePublishesCLIUnavailableReasonAndSameActionVerdict(t *testing.T) {
	m := newModelsApp(t, "")
	m.b.view = actionTestInput(Stopped).Snapshot
	m.b.view.Status.MemoryEstimate = nil
	// A Gemma 4 package: the CLI has no memory estimate for it yet (recorded).
	m.a.settings = Settings{ManifestPath: filepath.Join(e21Folder(t, e21Hauhau), "bubo-package.json")}
	v := waitSavedEstimate(t, m.a)
	want := e21Events(t, "estimate-hauhau-default.json")[0].Value.(*ModelEstimate).Reason
	if want == nil || v.SavedEstimate.Reason == nil || *v.SavedEstimate.Reason != *want || v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error == nil {
		t.Fatal("CLI no-estimate reason was replaced", v.SavedEstimate, want)
	}
	// An unknown verdict warns but never blocks: serve judges admission (B1).
	if !v.Actions.Start.Allowed || !strings.HasPrefix(v.Actions.Start.Reason, *want) {
		t.Fatal("setup and Start used different verdicts", v.SavedEstimate, v.Actions.Start)
	}
}

func TestSavedEstimateDoesNotClaimUnsupportedSavedPreferences(t *testing.T) {
	for _, saved := range []Settings{
		{ModelPath: "cli-model", Profile: "not-applied"},
		{ModelPath: "cli-model", ManifestPath: `C:\different\package.json`},
		{ModelPath: "cli-model", NPUPrefill: "not-applied"},
		{ModelPath: "cli-model", Effort: "not-applied"},
	} {
		m := newModelsApp(t, "")
		m.b.view = actionTestInput(Stopped).Snapshot
		m.b.view.Status.MemoryEstimate = nil
		m.a.settings = saved
		v := m.a.GetView()
		if v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error == nil || v.SavedEstimate.Error.Code != "estimate_needs_restart" || v.SavedEstimate.Reason == nil {
			t.Fatal("unsupported saved setup borrowed a fit verdict", saved, v.SavedEstimate, v.Actions)
		}
		// G-ENG round 2 B1: a saved setting the CLI cannot estimate must still
		// be appliable; Start carries the warning and serve judges admission.
		if !v.Actions.Start.Allowed {
			t.Fatal("saved setting locked out of Start", saved, v.Actions.Start)
		}
	}
}

// The owner's walk-through: a stopped server with a saved q8 cache and fast
// prefill. The page must show the fit at launch, with nothing edited.
func TestSavedCacheTypeAndPrefillModeStillGetAFitAtLaunch(t *testing.T) {
	m := newModelsApp(t, "")
	st := actionTestInput(Stopped).Snapshot
	st.Status.MemoryEstimate = nil
	st.Status.EffectiveSwitches = nil
	m.b.view = st
	m.a.settings = Settings{
		ManifestPath:    filepath.Join(e21Folder(t, e21Flash), "bubo-package.json"),
		SwitchOverrides: map[string]string{"--kv-cache-type": "q8_0", "--prefill-mode": "fast"},
	}
	v := waitSavedEstimate(t, m.a)
	if v.SavedEstimate.Error != nil || v.SavedEstimate.MemoryEstimate == nil || v.Fit.Class == "fit_unknown" || !v.Actions.Start.Allowed {
		t.Fatalf("a saved q8 cache and fast prefill left the launch without a fit: %+v %+v %+v", v.SavedEstimate.Error, v.Fit, v.Actions.Start)
	}
}

// Every saved serve override reaches the estimate, as `--switch NAME=VALUE` in name
// order, so the fit is the one the launch will have: the cache type, the side files,
// a bare flag and an environment row alike. None asks for a restart to be judged.
func TestSavedEstimateAsksForEverySavedOverride(t *testing.T) {
	for _, tc := range []struct {
		name      string
		overrides map[string]string
		want      []string
	}{
		{"q8 cache and fast prefill", map[string]string{"--prefill-mode": "fast", "--kv-cache-type": "q8_0"},
			[]string{"--switch", "--kv-cache-type=q8_0", "--switch", "--prefill-mode=fast"}},
		{"f16 is sent too", map[string]string{"--kv-cache-type": "f16"},
			[]string{"--switch", "--kv-cache-type=f16"}},
		{"side files, a bare flag and an environment row", map[string]string{
			"--mtp-model": `C:mmtp.gguf`, "--mmproj": `C:mmmproj.gguf`, "--no-cache": "", "GUFO_MAX_IMAGE_TOKENS": "1024"},
			[]string{"--switch", `--mmproj=C:mmmproj.gguf`, "--switch", `--mtp-model=C:mmtp.gguf`, "--switch", "--no-cache=", "--switch", "GUFO_MAX_IMAGE_TOKENS=1024"}},
		{"the think-marker placeholder is not a file", map[string]string{"GUFO_THINK_MARKER_LIST": "none"}, nil},
		{"nothing saved", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newModelsApp(t, "")
			st := actionTestInput(Stopped).Snapshot
			st.Status.MemoryEstimate = nil
			st.Status.EffectiveSwitches = nil
			m.b.view = st
			folder := e21Folder(t, e21Flash)
			m.a.settings = Settings{ManifestPath: filepath.Join(folder, "bubo-package.json"), SwitchOverrides: tc.overrides}
			v := waitSavedEstimate(t, m.a)
			if v.SavedEstimate.Error != nil && v.SavedEstimate.Error.Code == "estimate_needs_restart" {
				t.Errorf("a saved override asked for a restart to be estimated: %+v", v.SavedEstimate.Error)
			}
			estimates := recordedEstimates(t, m.dir)
			want := append([]string{"models", "estimate", "--model", folder, "--json"}, tc.want...)
			if len(estimates) != 1 || !reflect.DeepEqual(estimates[0], want) {
				t.Errorf("estimate arguments %v, want %v", estimates, want)
			}
		})
	}
}

// The recorded answer of `models estimate --kv-cache-type q8_0` parses, says which
// cache type its numbers assume, and needs less memory than the f16 answer.
func TestRecordedQ8EstimateNamesItsCacheTypeAndNeedsLess(t *testing.T) {
	read := func(file string) *ModelEstimate {
		raw, err := os.ReadFile(filepath.Join("..", "tests", "scenarios", "recordings", "v12", file))
		if err != nil {
			t.Fatal(err)
		}
		event, err := ParseEvent(raw)
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		return event.Value.(*ModelEstimate)
	}
	q8, f16 := read("estimate-flash-catalogue-id-q8.json"), read("estimate-flash-catalogue-id.json")
	if string(q8.FitInputs["kv_cache_type"]) != `"q8_0"` {
		t.Fatalf("fit_inputs: %s", q8.FitInputs)
	}
	if q8.MemoryEstimate == nil || f16.MemoryEstimate == nil || q8.MemoryEstimate.TotalRange.Max.Bytes >= f16.MemoryEstimate.TotalRange.Max.Bytes {
		t.Fatal("a q8 cache must need less memory than the f16 answer")
	}
}

// G-ENG round 2 B1: a saved NPU/profile/effort/override request is applied by
// Restart even though the CLI estimate cannot cover it; only fits:false blocks.
func TestSavedUnestimatedSettingCanBeAppliedByRestart(t *testing.T) {
	m := newModelsApp(t, "")
	m.b.view = actionTestInput(Ready).Snapshot
	m.b.view.Status.MemoryEstimate = nil
	m.a.settings = Settings{ModelPath: e21Folder(t, e21Flash), NPUPrefill: "on"}
	v := m.a.GetView()
	if !v.Actions.Restart.Allowed || v.Actions.Restart.ReasonCode != "estimate_unavailable" || v.Actions.Restart.Reason == "" {
		t.Fatal("saved setting locked out of Restart", v.Actions.Restart)
	}
}

func TestSavedEstimateErrorNeverPublishesPriorStatusFit(t *testing.T) {
	in := actionTestInput(Stopped)
	in.SavedEstimateError = &CLIError{Code: "estimate_needs_restart", Remedy: "This saved setup cannot be estimated."}
	v := new(ViewBuilder).Build(in)
	if v.SavedEstimate.MemoryEstimate != nil || v.SavedEstimate.Error == nil || v.SavedEstimate.Reason == nil || *v.SavedEstimate.Reason != in.SavedEstimateError.Remedy || !v.Actions.Start.Allowed {
		t.Fatal("unsupported saved setup published the prior allocation", v.SavedEstimate, v.Actions)
	}
	// Both the reason and typed error belong to this View.
	v.SavedEstimate.Error.Remedy = "mutated"
	if in.SavedEstimateError.Remedy == "mutated" || *v.SavedEstimate.Reason == "mutated" {
		t.Fatal("saved estimate error aliases the input")
	}
}

func TestLifecycleMissingFilesBlockStartAndRestartButAllowStop(t *testing.T) {
	for _, code := range []string{"runtime_missing", "model_missing", "source_missing"} {
		for _, state := range []State{Failed, Ready} {
			for _, statusFailure := range []bool{false, true} {
				in := actionTestInput(state)
				remedy := "The CLI says to prepare the missing file."
				if statusFailure {
					in.Snapshot.Status.LastError = &CLIError{Code: code, Remedy: remedy}
				} else {
					in.Snapshot.LastError = &Failure{Code: code, Remedy: remedy}
				}
				actions := new(ViewBuilder).Build(in).Actions
				blocked := actions.Start
				if state == Ready {
					blocked = actions.Restart
				}
				if blocked.Allowed || blocked.ReasonCode != code || blocked.Reason != remedy || actions.Stop.Allowed != (state == Ready) {
					t.Fatal("missing files borrowed stale readiness or blocked Stop", code, state, statusFailure, actions)
				}
			}
		}
	}
}
