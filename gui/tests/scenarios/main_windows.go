//go:build windows

// Run from gui/tests: go run ./scenarios. View JSON is never hand authored.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"bubo/gui/supervisor"
	"bubo/gui/tests/scenarios/replay"
	"bubo/gui/updater"
)

func main() {
	if err := generate(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func generate() error {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		return err
	}
	base := filepath.Join(root, "gui", "tests", "scenarios", "recordings")
	fixture := func(name string) string {
		return filepath.Join(root, "tests", "cli", "contract", "valid", name+".json")
	}
	recording := func(name string) string { return filepath.Join(base, filepath.FromSlash(name)) }
	type spec struct {
		name                             string
		state                            supervisor.State
		source, status, models           string
		stall, lan, download             bool
		update                           *supervisor.Update
		installedProduct                 *supervisor.Product
		savedOverrides                   map[string]string
		overrideFailure                  *supervisor.CLIError
		autoRestarted                    bool
		failure                          *supervisor.CLIError
		failureAgain                     bool
		npuState                         string
		estimate, savedEstimate          string // a recorded estimate of the e21 folder
		savedEstimateError               *supervisor.CLIError
		noVerdictFrom                    string // a recorded estimate whose reason is the saved fit error
		savedModelID                     string
		firstStart, cliMissing, ringLogs bool
		noRowFits                        bool // hand-written overlay: no recording has a model that fits at no context size
	}



	const (
		mainStatus  = "v12/status-main.json"
		emptyStatus = "v12/status-empty.json"
		fullList    = "v12/models-list-gui.json"
		emptyList   = "v12/models-list-empty.json"
		flashPlan   = "v12/plan-huihui-qwen38-flash-next-ud-q4-k-xl-v1-fresh.jsonl"
		flashFit    = "v12/estimate-flash-default.json"
	)
	idle := "v12/status-w-idle.json"
	// The release exe reports every switch's effective value, so it cannot show
	// a switch whose running value is unknown; the pending fixture keeps the
	// older status that lacks some.
	partialIdle := "v12/status-gate-idle.json"
	busy := "v12/status-gate-busy.json"
	specs := []spec{
		{name: "first-start", state: supervisor.NotPrepared, firstStart: true},
		{name: "cli-missing", state: supervisor.Failed, cliMissing: true},
		// The engine-missing state has no E21 recording: the b4 status stays.
		{name: "not-prepared", state: supervisor.NotPrepared, source: "fixture", status: "v12/status-engine-missing.json", models: emptyList},
		{name: "runtime-ready-model-missing", state: supervisor.Stopped, source: "recorded", status: emptyStatus, models: emptyList},
		{name: "onboarding-96-64", state: supervisor.Stopped, source: "fixture", status: "v12-96gb/status-96-64.json", models: "v12-96gb/models-96-64.json"},
		{name: "onboarding-96-72", state: supervisor.Stopped, source: "fixture", status: "v12-96gb/status-96-72.json", models: "v12-96gb/models-96-72.json"},
		{name: "onboarding-128-96", state: supervisor.Stopped, source: "fixture", status: "v12-96gb/status-128-96.json", models: "v12-96gb/models-128-96.json"},
		{name: "preparing", state: supervisor.Preparing, status: emptyStatus, models: emptyList},
		{name: "stopped", state: supervisor.Stopped, source: "recorded", status: mainStatus},
		{name: "starting", state: supervisor.Starting, status: mainStatus},
		{name: "ready-idle", state: supervisor.Ready, source: "fixture", status: idle, ringLogs: true},
		{name: "ready-busy", state: supervisor.Ready, source: "fixture", status: "v12/status-lrel8-busy.json", ringLogs: true},
		{name: "stalled", state: supervisor.Ready, status: busy, stall: true},
		{name: "failed-with-remedy", state: supervisor.Failed, status: mainStatus},
		{name: "external-server", state: supervisor.AttachedExternal, source: "fixture", status: idle},
		{name: "models-download-in-progress", state: supervisor.Stopped, status: mainStatus, download: true},
		{name: "lan-on", state: supervisor.Ready, status: idle, lan: true},
		// A model with no verdict (startup lower bound): Qwen3.8 27B at 262144 x 1.
		{name: "owner-memory-fit", state: supervisor.Stopped, status: mainStatus, estimate: "v12/estimate-q27-c262144-s1.json"},
		{name: "stopped-saved-estimate", state: supervisor.Stopped, status: mainStatus, savedEstimate: flashFit},
		// Gemma 4: the CLI has no memory estimate for the Huihui QAT package.
		{name: "saved-no-verdict", state: supervisor.Stopped, status: mainStatus, noVerdictFrom: "v12/estimate-hauhau-default.json", savedModelID: "gemma4-26b-a4b-uncensored-hauhaucs-qat-q4-k-m"},
		{name: "saved-fit-unchecked", state: supervisor.Stopped, status: mainStatus, savedEstimateError: &supervisor.CLIError{Code: "estimate_needs_restart", Remedy: "The saved setup cannot be estimated before the next start."}},
		// Memory facts are the CLI's: Flash Next at 2 x 262144 fits but fails
		// the recommendation headroom checks; at 8 x 262144 both fit checks fail.
		// A host-floor-only failure did not come out of the recorded grid.
		// S20 (W6b): CLI text at the Models and NPU sites: the recorded list with the
		// contract fixture's scan warning and unusable files (e23 fixture metadata),
		// and a faulted NPU runtime.
		{name: "models-warnings", state: supervisor.Stopped, status: mainStatus, savedEstimate: "v12/estimate-flash-c262144-s2.json", models: "v12/models-with-warnings.json", npuState: "faulted"},
		{name: "memory-gpu-headroom", state: supervisor.Stopped, status: mainStatus, savedEstimate: "v12/estimate-flash-c262144-s2.json"},
		{name: "memory-default-setup", state: supervisor.Stopped, status: mainStatus, estimate: "v12/estimate-flash-c262144-s8.json"},
		{name: "no-fit", state: supervisor.Stopped, status: mainStatus, savedEstimate: "v12/estimate-flash-c262144-s4.json", noRowFits: true},
		{name: "memory-smaller-context", state: supervisor.Stopped, status: mainStatus, estimate: "v12/estimate-flash-c262144-s4.json"},
		{name: "switch-overrides-pending", state: supervisor.Ready, status: partialIdle, savedOverrides: map[string]string{"GUFO_LOOP_BREAKER": "0", "GUFO_CACHE_HOST_FLOOR_MB": "4321"}},
		// Exit status 75 (GPU lost): restarted once by the supervisor, and the second 75 stopped.
		{name: "gpu-restarted", state: supervisor.Ready, source: "fixture", status: idle, autoRestarted: true},
		{name: "gpu-lost-stopped", state: supervisor.Failed, status: mainStatus, savedEstimate: flashFit, failure: &supervisor.CLIError{Code: "device_lost", Remedy: "Review the local logs and retry on request."}, failureAgain: true},
		{name: "switch-override-rejected", state: supervisor.Failed, status: mainStatus, savedOverrides: map[string]string{"--think-marker-bias": "1", "GUFO_LOOP_BREAKER": "0"}, overrideFailure: &supervisor.CLIError{Code: "invalid_arguments", Component: "serve", Stage: "startup", Remedy: "Check the settings and model files.", Switch: "--think-marker-bias"}},
		{name: "cli-page-facts-loopback", state: supervisor.Ready, status: "v12/status-page-facts-false.json"},
		{name: "cli-page-facts-authenticated", state: supervisor.Ready, status: "v12/status-page-facts-true.json"},
	}
	for _, state := range []string{"on", "faulted", "off"} {
		specs = append(specs, spec{name: "npu-prefill-" + state, state: supervisor.Ready, status: idle, npuState: state})
	}
	// Real package facts replace the old 64 MB / future 0.1.2 placeholders.
	var release updater.Manifest
	raw, err := os.ReadFile(recording("release-update/bubo-release.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &release); err != nil {
		return err
	}
	var published struct {
		Product struct {
			Version  string `json:"product_version"`
			Revision string `json:"revision"`
			Engine   string `json:"engine_version"`
		} `json:"product"`
	}
	raw, err = os.ReadFile(recording("release-update/published-build-manifest.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &published); err != nil {
		return err
	}
	oldProduct := &supervisor.Product{Name: release.Product, Version: published.Product.Version, Revision: published.Product.Revision, EngineVersion: published.Product.Engine}
	notes := release.Notes.Summary
	for _, state := range []supervisor.UpdateState{supervisor.UpdateNone, supervisor.UpdateCurrent, supervisor.UpdateAvailable, supervisor.UpdateInProgress, supervisor.UpdateReady, supervisor.UpdateFailed, supervisor.UpdateRollbackPending} {
		u := &supervisor.Update{State: state, Ready: true}
		if state != supervisor.UpdateNone && state != supervisor.UpdateCurrent {
			u.CurrentVersion, u.AvailableVersion = oldProduct.Version, release.Version
			u.Notes = notes
			u.SizeBytes = ptr(uint64(release.Assets[0].Size))
			u.RuntimeDownloadBytes, u.ModelsDownloadBytes = supervisor.ReleaseReuseFacts(release)
		}
		if state == supervisor.UpdateInProgress {
			// An arranged transfer point, bounded by the actual package size.
			u.Progress = &supervisor.UpdateProgress{Phase: "download", BytesDone: uint64(release.Assets[0].Size / 4), BytesTotal: ptr(uint64(release.Assets[0].Size))}
		}
		if state == supervisor.UpdateFailed {
			u.Error = &supervisor.CLIError{Code: "network", Remedy: "The recorded release feed could not be reached.", Retryable: true}
		}
		if state == supervisor.UpdateRollbackPending {
			u.Error = &supervisor.CLIError{Code: "rollback_pending", Component: "update", Remedy: "new GUI is still running; .previous is intact and rollback must wait for its exit close the new GUI and its server, then invoke Recover; no processes were killed", Retryable: true}
			u.LastResultNotice = "Update rollback is pending. Install retries recovery after stopping CLI work. new GUI is still running; .previous is intact and rollback must wait for its exit"
			u.PreviousVersion = "0.1.0"
		}
		sp := spec{name: "update-" + string(state), state: supervisor.Stopped, status: mainStatus, update: u}
		if state != supervisor.UpdateNone && state != supervisor.UpdateCurrent && state != supervisor.UpdateRollbackPending {
			sp.installedProduct = oldProduct
		}
		if state == supervisor.UpdateReady || state == supervisor.UpdateRollbackPending {
			sp.savedEstimate = flashFit
		}
		specs = append(specs, sp)
	}
	// The updater could not start (the cause the real first start showed with a
	// refused manifest address): no update was started, so no header pill.
	specs = append(specs, spec{name: "update-check-failed", state: supervisor.Stopped, status: mainStatus, update: &supervisor.Update{State: supervisor.UpdateFailed, Ready: false, CheckFailed: true, Error: &supervisor.CLIError{Code: "url_refused", Component: "update", Remedy: "manifest must use latest or an exact stable release tag", Retryable: true}}})
	out := filepath.Join(root, "gui", "wails", "frontend", "src", "scenarios")
	if err = os.MkdirAll(out, 0755); err != nil {
		return err
	}
	generated := 0
	for _, s := range specs {
		if only := os.Getenv("BUBO_SCENARIO_ONLY"); only != "" && only != s.name {
			continue
		}
		dir := filepath.Join(base, s.name)
		if err = os.MkdirAll(dir, 0755); err != nil {
			return err
		}
		if s.source == "" {
			s.source = "fixture"
		}
		m := replay.Manifest{Source: s.source, State: s.state, Stall: s.stall, LAN: s.lan, Download: s.download, Update: s.update, SavedOverrides: s.savedOverrides, OverrideFailure: s.overrideFailure, AutoRestarted: s.autoRestarted, Failure: s.failure, FailureAgain: s.failureAgain, SavedModelID: s.savedModelID, NoRowFits: s.noRowFits, SavedEstimateError: s.savedEstimateError, FirstStart: s.firstStart, CLIMissing: s.cliMissing, Records: map[string]string{}}
		m.InstalledProduct = s.installedProduct
		if s.noVerdictFrom != "" {
			// The saved fit error carries the CLI's own reason of the recorded estimate.
			reason, err := recordedReason(recording(s.noVerdictFrom))
			if err != nil {
				return err
			}
			m.SavedEstimateError = &supervisor.CLIError{Code: "estimate_no_verdict", Remedy: reason}
		}
		if s.ringLogs {
			logPath := filepath.Join(root, "gui", "tests", "fakecli", "recorded-server-info.log")
			rel, err := filepath.Rel(dir, logPath)
			if err != nil {
				return err
			}
			m.LogRecord = filepath.ToSlash(rel)
		}
		paths := map[string]string{"status": recording(mainStatus), "switches": recording("v12/switches.json"), "models": recording(fullList)}
		if s.firstStart || s.cliMissing {
			clear(paths)
		}
		if s.models != "" {
			paths["models"] = recording(s.models)
		}
		if s.status != "" {
			paths["status"] = recording(s.status)
		}
		if s.state == supervisor.Preparing {
			paths["prepare"] = recording(flashPlan)
			paths["progress"] = recording("v12/progress-model.json")
		}
		if s.download {
			paths["progress"] = recording("v12/progress-model.json")
		}
		if s.state == supervisor.Failed && !s.cliMissing {
			paths["failure"] = fixture("error-runtime_missing")
		}
		if s.estimate != "" {
			paths["estimate"] = recording(s.estimate)
		}
		if s.savedEstimate != "" {
			paths["saved_estimate"] = recording(s.savedEstimate)
		}
		if s.npuState != "" {
			paths["npu_runtime"] = fixture("status-npu-" + s.npuState)
		}
		for k, p := range paths {
			r, err := filepath.Rel(dir, p)
			if err != nil {
				return err
			}
			m.Records[k] = filepath.ToSlash(r)
		}
		b, err := json.MarshalIndent(m, "", "  ")
		if err != nil {
			return err
		}
		if err = os.WriteFile(filepath.Join(dir, "scenario.json"), append(b, '\n'), 0644); err != nil {
			return err
		}
		v, err := replay.Build(dir)
		if err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}
		b, err = json.MarshalIndent(v, "", "  ")
		if err != nil {
			return err
		}
		if err = os.WriteFile(filepath.Join(out, s.name+".json"), append(b, '\n'), 0644); err != nil {
			return err
		}
		fmt.Printf("%s: %d bytes (%s)\n", s.name, len(b)+1, m.Source)
		generated++
	}
	fmt.Printf("scenarios: %d passed, 0 failed\n", generated)
	return runtimePlanFixture(root, base)
}

// The engine consent fixture is projected by the same ViewBuilder as the app,
// from the real CLI preview; no file sizes, licences or paths are invented.
func runtimePlanFixture(root, base string) error {
	for _, item := range []struct{ file, output, id string }{
		{"plan-runtime-fresh.jsonl", "runtime-plan.json", ""},
		{"plan-huihui-qwen38-flash-next-lowmemory-v1-fresh.jsonl", "lower-memory-plan.json", "huihui-qwen38-flash-next-lowmemory-v1"},
		{"plan-lowmemory-short.jsonl", "lower-memory-short-plan.json", "huihui-qwen38-flash-next-lowmemory-v1"},
		{"plan-gemma4-26b-a4b-huihui-abliterated-qat-q4-k-fresh.jsonl", "gemma-qat-plan.json", "gemma4-26b-a4b-huihui-abliterated-qat-q4-k"},
		{"plan-qwen38-flash-next-base-ud-q4-k-xl-v1-fresh.jsonl", "original-plan.json", "qwen38-flash-next-base-ud-q4-k-xl-v1"},
		{"plan-flash-rootC-fresh.jsonl", "flash-short-plan.json", "huihui-qwen38-flash-next-ud-q4-k-xl-v1"},
	} {
		if err := writePlanFixture(root, base, item.file, item.output, item.id); err != nil {
			return err
		}
	}
	frontend := filepath.Join(root, "gui", "wails", "frontend")
	command := exec.Command("node", filepath.Join(frontend, "node_modules", "prettier", "bin", "prettier.cjs"), "--write",
		filepath.Join(frontend, "tests", "runtime-plan.json"), filepath.Join(frontend, "tests", "lower-memory-plan.json"),
		filepath.Join(frontend, "tests", "lower-memory-short-plan.json"), filepath.Join(frontend, "tests", "gemma-qat-plan.json"),
		filepath.Join(frontend, "tests", "original-plan.json"), filepath.Join(frontend, "tests", "flash-short-plan.json"))
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("format recorded plans: %w: %s", err, output)
	}
	return writeMemory96EstimateFixture(root, base)
}

func writeMemory96EstimateFixture(root, base string) error {
	raw, err := os.ReadFile(filepath.Join(base, "v12-96gb", "estimate-128-96-s1.json"))
	if err != nil {
		return err
	}
	event, err := supervisor.ParseEvent(raw)
	if err != nil {
		return err
	}
	estimate, ok := event.Value.(*supervisor.ModelEstimate)
	if !ok {
		return fmt.Errorf("memory96 fixture is not a model estimate")
	}
	view := (&supervisor.ViewBuilder{}).Build(supervisor.ViewInput{SavedEstimate: estimate.MemoryEstimate})
	reply := struct {
		Estimate *supervisor.ModelEstimate `json:"estimate"`
		Fit      supervisor.FitVerdict     `json:"fit"`
		Error    *supervisor.CLIError      `json:"error"`
	}{Estimate: estimate, Fit: view.Fit}
	encoded, err := json.MarshalIndent(reply, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(root, "gui", "wails", "frontend", "src", "generated", "memory96-estimate-reply.json"), append(encoded, '\n'), 0644)
}

func writePlanFixture(root, base, file, output, id string) error {
	raw, err := os.ReadFile(filepath.Join(base, "v12", file))
	if err != nil {
		return err
	}
	in := supervisor.ViewInput{Source: "fixture", Snapshot: supervisor.Snapshot{State: supervisor.NotPrepared, Initialized: true}}
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		e, err := supervisor.ParseEvent([]byte(line))
		if err != nil {
			return err
		}
		switch value := e.Value.(type) {
		case *supervisor.Plan:
			in.Snapshot.PlanParts = append(in.Snapshot.PlanParts, *value)
		case *supervisor.Approval:
			in.Snapshot.Approval = value
		}
	}
	v := (&supervisor.ViewBuilder{}).Build(in)
	if v.PlanError != nil || len(v.Preparation) != 1 || id == "" && v.Approval == nil {
		return fmt.Errorf("%s cannot form a consent fixture: %+v", file, v.PlanError)
	}
	if id != "" {
		raw, err := os.ReadFile(filepath.Join(base, "v12", "models-list-empty.json"))
		if err != nil {
			return err
		}
		event, err := supervisor.ParseEvent(raw)
		if err != nil {
			return err
		}
		entry, ok := event.Value.(*supervisor.Models).Entry(id)
		if !ok {
			return fmt.Errorf("missing catalogue id %s", id)
		}
		v.Preparation[0].ID = id
		v.Preparation[0].Kind = "download"
		v.Preparation[0].Licence = &entry.Licence
	}
	b, err := json.MarshalIndent(struct {
		Plan     supervisor.ModelPlan `json:"plan"`
		Approval *supervisor.Approval `json:"approval"`
	}{v.Preparation[0], v.Approval}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(root, "gui", "wails", "frontend", "tests", output), append(b, '\n'), 0644)
}

func ptr[T any](v T) *T { return &v }

// recordedReason is the reason a recorded model estimate gives.
func recordedReason(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var e struct {
		Payload struct {
			Reason *string `json:"reason"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(b, &e); err != nil || e.Payload.Reason == nil {
		return "", fmt.Errorf("%s has no reason", path)
	}
	return *e.Payload.Reason, nil
}
