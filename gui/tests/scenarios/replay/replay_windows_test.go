//go:build windows

package replay

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"bubo/gui/supervisor"
)

func TestF6RecordedRunningViewAllowsRestartWithMemoryWarning(t *testing.T) {
	v, err := Build(filepath.Join("..", "recordings", "ready-idle"))
	if err != nil {
		t.Fatal(err)
	}
	if !v.Actions.Restart.Allowed || v.Actions.Restart.ReasonCode != "fit_unchecked" || !strings.Contains(v.Actions.Restart.Reason, "Memory is checked when it restarts") || !v.Actions.Stop.Allowed {
		t.Fatalf("recorded running allocation blocked Restart: %+v", v.Actions)
	}
}

func TestScenariosUseRealBuilderAndMatchCommittedViews(t *testing.T) {
	dirs, err := filepath.Glob(filepath.Join("..", "recordings", "*", "scenario.json"))
	if err != nil || len(dirs) != 43 {
		t.Fatalf("scenario count %d: %v", len(dirs), err)
	}
	for _, manifest := range dirs {
		t.Run(filepath.Base(filepath.Dir(manifest)), func(t *testing.T) {
			folder := filepath.Dir(manifest)
			m, _, err := Read(folder)
			if err != nil {
				t.Fatal(err)
			}
			v, err := Build(folder)
			if err != nil {
				t.Fatal(err)
			}
			b, err := json.MarshalIndent(v, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(filepath.Join("..", "..", "..", "wails", "frontend", "src", "scenarios", filepath.Base(folder)+".json"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(append(b, '\n'), want) {
				t.Fatal("scenario is stale; cd gui/tests; go run ./scenarios")
			}
			if v.Source != "recorded" && v.Source != "fixture" {
				t.Fatal("missing provenance")
			}
			if filepath.Base(folder) == "ready-busy" && (v.Sessions == nil || *v.Sessions != 1 || len(v.Slots) != 1 || v.Slots[0].State != "generating") {
				t.Fatal("busy scenario did not reach builder")
			}
			if filepath.Base(folder) == "stalled" && v.Stall == nil {
				t.Fatal("stall scenario missing warning")
			}
			if filepath.Base(folder) == "preparing" && (v.Progress == nil || v.Progress.Component != "model" || v.Runtime.Readiness.Model || !v.Runtime.Readiness.Runtime || v.Prepared) {
				t.Fatal("active model transfer contradicts preparation readiness")
			}
			if filepath.Base(folder) == "failed-with-remedy" && (v.Failure == nil || v.Remedy == "") {
				t.Fatal("failure remedy lost")
			}
			if filepath.Base(folder) == "models-download-in-progress" && (v.Models == nil || v.Models.Download == nil || v.Models.Download.BytesDone == 0) {
				t.Fatal("download progress lost")
			}
			for _, kind := range []string{"status", "switches", "models"} {
				if _, ok := m.Records[kind]; !ok {
					continue
				}
				events, err := Command(folder, kind)
				if err != nil {
					t.Fatal(err)
				}
				for _, e := range events {
					b, _ := json.Marshal(e)
					if _, err := supervisor.ParseEvent(b); err != nil {
						t.Fatalf("%s replay: %v", kind, err)
					}
				}
			}
		})
	}
}

func TestRecordedRunningScenariosPreserveRealCLIState(t *testing.T) {
	for _, name := range []string{"ready-idle", "ready-busy", "external-server", "stopped"} {
		t.Run(name, func(t *testing.T) {
			folder := filepath.Join("..", "recordings", name)
			m, records, err := Read(folder)
			if err != nil {
				t.Fatal(err)
			}
			wantSource := "fixture"
			if name == "stopped" {
				wantSource = "recorded"
			}
			if m.Source != wantSource {
				t.Fatal("status provenance differs from raw/derived inputs")
			}
			original := records["status"][0].Value.(*supervisor.Status)
			in, err := Inputs(folder)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(original, in.Snapshot.Status) {
				t.Fatal("recorded CLI facts changed during replay")
			}
			if len(original.Instances) > 0 && in.Snapshot.Instance == nil {
				t.Fatal("recorded instance not selected")
			}
		})
	}
}

func TestRecordedScenarioRejectsSyntheticOverlays(t *testing.T) {
	folder := t.TempDir()
	b, _ := json.Marshal(Manifest{Source: "recorded", Busy: true})
	if err := os.WriteFile(filepath.Join(folder, "scenario.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Read(folder); err == nil {
		t.Fatal("synthetic recorded scenario accepted")
	}
}

func TestWriteContractScenarios(t *testing.T) {
	v, err := Build(filepath.Join("..", "recordings", "switch-overrides-pending"))
	if err != nil {
		t.Fatal(err)
	}
	states := map[string]supervisor.SwitchRestartState{}
	for _, row := range v.SwitchOverrides {
		states[row.Name] = row.RestartState
	}
	if states["GUFO_LOOP_BREAKER"] != supervisor.SwitchPending || states["GUFO_CACHE_HOST_FLOOR_MB"] != supervisor.SwitchUnknownUntilRestart {
		t.Fatal("pending fixture lost known/unknown state", states)
	}
	v, err = Build(filepath.Join("..", "recordings", "switch-override-rejected"))
	if err != nil {
		t.Fatal(err)
	}
	if v.Source != "fixture" || v.Failure == nil || v.Failure.Code != "invalid_arguments" || v.Failure.Component != "serve" || v.Failure.Switch != "--think-marker-bias" || v.Saved.SwitchOverrides[v.Failure.Switch] != "1" {
		t.Fatal("rejection fixture lost failure or clearable saved request")
	}
	in, err := Inputs(filepath.Join("..", "recordings", "switch-override-rejected"))
	if err != nil {
		t.Fatal(err)
	}
	remaining, err := supervisor.ApplySwitchOverrides(in.Saved.SwitchOverrides, supervisor.SwitchOverridesPatch{Clear: []string{v.Failure.Switch}}, in.Registry)
	if err != nil || len(remaining) != 1 || remaining["GUFO_LOOP_BREAKER"] != "0" {
		t.Fatal("culprit clear changed an unrelated override", remaining, err)
	}
}

func TestNPURuntimeFixtureScenarios(t *testing.T) {
	for _, state := range []supervisor.NPUPrefillState{supervisor.NPUPrefillOn, supervisor.NPUPrefillFaulted, supervisor.NPUPrefillOff} {
		t.Run(string(state), func(t *testing.T) {
			folder := filepath.Join("..", "recordings", "npu-prefill-"+string(state))
			m, records, err := Read(folder)
			if err != nil {
				t.Fatal(err)
			}
			v, err := Build(folder)
			if err != nil || m.Source != "fixture" || v.Source != "fixture" || v.State != supervisor.Ready || v.NPUPrefillRuntime == nil || v.NPUPrefillRuntime.State != state {
				t.Fatal("NPU fixture state lost", v.NPUPrefillRuntime, err)
			}
			original := records["npu_runtime"][0].Value.(*supervisor.Status).NPUPrefillRuntime
			if !reflect.DeepEqual(v.NPUPrefillRuntime, original) {
				t.Fatal("NPU fixture values changed")
			}
		})
	}
}

// ready-idle comes from walk W's release-exe status (recordings/w), taken at
// Ready before any request, so only ready-busy still merges two recordings.
func TestReadyIdleIsTheReleaseExeStatus(t *testing.T) {
	m, _, err := Read(filepath.Join("..", "recordings", "ready-idle"))
	if err != nil {
		t.Fatal(err)
	}
	if m.Source != "fixture" || !strings.Contains(filepath.ToSlash(m.Records["status"]), "v12/status-w-idle.json") {
		t.Fatal("ready-idle is not walk W's release-exe status", m.Records["status"])
	}
}

func TestMergeTwoRecordingsKeepLiveRatesAndContext(t *testing.T) {
	for _, name := range []string{"ready-busy"} {
		t.Run(name, func(t *testing.T) {
			folder := filepath.Join("..", "recordings", name)
			m, records, err := Read(folder)
			if err != nil {
				t.Fatal(err)
			}
			if m.Source != "fixture" || !strings.Contains(m.Records["status"], "v12/status-lrel8-busy.json") {
				t.Fatal("merge-2 provenance lost")
			}
			st := records["status"][0].Value.(*supervisor.Status)
			v, err := Build(folder)
			if err != nil {
				t.Fatal(err)
			}
			if st.LiveThroughput == nil || st.LiveThroughput.DecodeTPS == nil || st.LiveThroughput.PrefillTPS == nil || v.Rates == nil || !reflect.DeepEqual(v.Rates.DecodeTPS, st.LiveThroughput.DecodeTPS) || !reflect.DeepEqual(v.Rates.PrefillTPS, st.LiveThroughput.PrefillTPS) {
				t.Fatal("recorded live rates lost")
			}
			if len(v.Slots) != 1 || v.Slots[0].ContextTokens == nil || *v.Slots[0].ContextTokens == 0 || !reflect.DeepEqual(v.Slots[0].ContextTokens, st.Slots[0].ContextTokens) {
				t.Fatal("recorded resolved slot context lost")
			}
			if v.Endpoints == nil || !reflect.DeepEqual(v.Endpoints, st.Endpoints) {
				t.Fatal("recorded endpoints lost")
			}
		})
	}
}

// The owner-memory scenario shows a recorded estimate of a model the CLI cannot
// check (a startup lower bound): the View carries the CLI's numbers unchanged.
func TestOwnerMemoryScenarioCarriesTheRecordedEstimateUnchanged(t *testing.T) {
	folder := filepath.Join("..", "recordings", "owner-memory-fit")
	_, records, err := Read(folder)
	if err != nil {
		t.Fatal(err)
	}
	want := records["estimate"][0].Value.(*supervisor.ModelEstimate).MemoryEstimate
	v, err := Build(folder)
	if err != nil {
		t.Fatal(err)
	}
	m := v.MemoryEstimate
	if v.Source != "fixture" || m == nil || m.GPU == nil || m.Host == nil || m.GPU.Available == nil || m.Host.Available == nil || m.Fits == nil || !*m.Fits {
		t.Fatal("owner scenario lost the CLI budgets or the fit")
	}
	if !reflect.DeepEqual(m.GPU.Available, want.GPU.Available) || !reflect.DeepEqual(m.Host.Available, want.Host.Available) || m.ContextTokens != want.ContextTokens || m.Sessions != want.Sessions {
		t.Fatal("owner scenario no longer matches the recorded estimate")
	}
	if m.FitVerified || m.FitScope != "startup_lower_bound" || m.CacheMode != "automatic" {
		t.Fatal("planning limitation was lost")
	}
}

func TestEveryUpdateStateReachesGeneratedView(t *testing.T) {
	for _, state := range []supervisor.UpdateState{supervisor.UpdateNone, supervisor.UpdateCurrent, supervisor.UpdateAvailable, supervisor.UpdateInProgress, supervisor.UpdateReady, supervisor.UpdateFailed, supervisor.UpdateRollbackPending} {
		t.Run(string(state), func(t *testing.T) {
			v, err := Build(filepath.Join("..", "recordings", "update-"+string(state)))
			if err != nil {
				t.Fatal(err)
			}
			if v.Update.State != state || v.Product == nil {
				t.Fatal("update state or CLI installed version lost")
			}
			if state == supervisor.UpdateRollbackPending {
				// The new GUI is still running while the previous release awaits recovery.
				if v.Product.Version != "0.1.1" || v.Update.CurrentVersion != "0.1.0" || v.Update.PreviousVersion != "0.1.0" {
					t.Fatal("rollback lost the current CLI and previous published release versions")
				}
			} else if v.Update.CurrentVersion != v.Product.Version {
				t.Fatal("update state or CLI installed version lost")
			}
			if state == supervisor.UpdateInProgress && (v.Update.Progress == nil || v.Update.Progress.BytesTotal == nil || v.Update.Progress.BytesDone == 0) {
				t.Fatal("update progress lost")
			}
			if state == supervisor.UpdateRollbackPending && (v.Update.Error == nil || v.Update.LastResultNotice == "") {
				t.Fatal("rollback notice lost")
			}
			if state == supervisor.UpdateReady && (!v.Actions.Start.Allowed || v.SavedEstimate.MemoryEstimate == nil) {
				t.Fatal("update-ready lost the saved setup and allowed Start")
			}
			if state == supervisor.UpdateRollbackPending && (v.Actions.Start.Allowed || v.Actions.Start.ReasonCode != "rollback_pending" || !v.Actions.Recover.Allowed || v.SavedEstimate.MemoryEstimate == nil) {
				t.Fatal("rollback pending has inconsistent saved setup or actions")
			}
		})
	}
}

func TestAddedGoInputScenariosKeepDerivedFacts(t *testing.T) {
	model, err := Build(filepath.Join("..", "recordings", "runtime-ready-model-missing"))
	if err != nil {
		t.Fatal(err)
	}
	if model.Phase != "needs_model" || model.StatusLabel != "Setup needed: download a model" || !model.Actions.DownloadDefault.Allowed || model.Actions.Start.ReasonCode != "model_missing" {
		t.Fatal("runtime-ready model-missing scenario has stale derived facts")
	}
	fit, err := Build(filepath.Join("..", "recordings", "saved-fit-unchecked"))
	if err != nil {
		t.Fatal(err)
	}
	if fit.Phase != "stopped" || !fit.Actions.Start.Allowed || fit.SavedEstimate.Error == nil || fit.SavedEstimate.MemoryEstimate != nil {
		t.Fatal("saved fit unchecked scenario lacks its real Go input")
	}
	for _, name := range []string{"ready-idle", "ready-busy"} {
		v, err := Build(filepath.Join("..", "recordings", name))
		if err != nil || len(v.Logs) < 2 {
			t.Fatalf("%s lost recorded ring log lines: %v", name, err)
		}
	}
}
