//go:build windows

package main

// The Models page calls of the fake CLI: `models list`, `models estimate` and
// `prepare model --catalogue-id`. Every record is a recording of the candidate
// exe (recorded_windows.go, gui/tests/scenarios/recordings/v12). The fake adds
// nothing to a recorded answer. A field written by hand is a finding.
//
// Modes (BUBO_FAKE_MODE):
//
//	runtime-missing     models list/estimate: runtime_missing, exit 11
//	models-garbage      models list: a models event that breaks the contract
//	model-plan-changed  the download run offers a different plan digest
//	model-stall         the run sends one progress record, then stays silent
//	model-error         the run fails with storage_failed (exit 7) after approval
//	model-no-approval   the run never asks for approval and finishes (a bad CLI)
//	model-plan-verification  a plan verifies existing files with >400 progress records
//
// BUBO_FAKE_MODELS_RECORDING names another recorded `models list` file of the
// v12 folder (default: models-list-gui, or models-list-empty while the ui
// modes have not prepared a model).
//
// Files in BUBO_FAKE_DIR:
//
//	models-argv-N.json   the arguments of the Nth models call
//	prepare-argv-N.json  the arguments of the Nth catalogue prepare call
//	volume-short         present: the recorded plan volume reports less free space
//	                     than it needs (hand-edited field: no recording of a short
//	                     volume exists)
//	model-steps          progress steps of a run (default 4)

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// The CLI binds approval to the component digests, rather than echoing a
// component's plan_digest. Keep the fake's wire answer distinct as well.
func approvalDigest(components map[string]string) string {
	data, err := json.Marshal(components)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

func runtimeMissingForModels() {
	cliError("runtime_missing", "engine", "bootstrap", false, "Run bubo prepare runtime. Reinstall the Bubo package.", 11)
}

func allFlags(name string) []string {
	var out []string
	for i, a := range os.Args {
		if a == name && i+1 < len(os.Args) {
			out = append(out, os.Args[i+1])
		}
	}
	return out
}

func recordArgv(prefix string) {
	for n := 1; ; n++ {
		path := filepath.Join(dir, fmt.Sprintf("%s-argv-%d.json", prefix, n))
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if os.IsExist(err) {
			continue
		}
		if err != nil {
			panic(err)
		}
		b, _ := json.Marshal(os.Args[1:])
		if _, err := file.Write(b); err != nil {
			file.Close()
			panic(err)
		}
		file.Close()
		return
	}
}

func modelsCommand() {
	recordArgv("models")
	if len(os.Args) < 3 {
		cliError("invalid_arguments", "models", "arguments", false, "Use models list or models estimate.", 2)
	}
	if mode == "runtime-missing" || ((mode == "ui" || mode == "ui-fail") && !fakeRuntimePrepared()) {
		runtimeMissingForModels()
	}
	switch os.Args[2] {
	case "list":
		if packagesMode() {
			kind, payload := packagesModelsList()
			event(kind, payload)
			return
		}
		file := recordedListFile()
		if mode == "models-garbage" {
			kind, payload := recordedOne(file)
			delete(payload, "catalogue")
			event(kind, payload)
			return
		}
		// The recorded bytes, unchanged: the folders the GUI passed are not
		// echoed, as the recording holds the folders of the recording run.
		printRecording(file)
	case "estimate":
		if packagesMode() {
			packagesEstimate()
			return
		}
		subject := flag("--model")
		if httpMode == "mvp" {
			// MVP smoke: the allocation comes from the smoke's own recording
			// (mvp_windows.go; hand-edited fields, see the E21 report).
			_, payload := recordedOne("estimate-flash-default.json")
			payload["subject"] = subject
			n, _ := strconv.Atoi(flag("--sessions"))
			mvpEstimate(payload, n)
			event("model_estimate", payload)
			return
		}
		printRecording(recordedEstimateFile(subject, flag("--context"), flag("--sessions")))
	default:
		cliError("invalid_arguments", "models", "arguments", false, "Use models list or models estimate.", 2)
	}
}

// recordedPlan reads the recorded plan of one catalogue id. A refusal of the
// CLI (a download-disabled entry, an unknown id) comes with its exit code.
func recordedPlan(id string) (lines []recordedLine, exit int) {
	file := recordedPlanFile(id)
	return recordedLines(file), recordedExitCode(file)
}

// planDigestOf is the digest the recorded plan records carry.
func planDigestOf(lines []recordedLine) string {
	for _, l := range lines {
		if l.Type == "plan" {
			if d, ok := l.Payload["plan_digest"].(string); ok {
				return d
			}
		}
	}
	return ""
}

// emitRecordedPlan writes the recorded records of a plan up to, but not
// including, its result record. With a runDigest other than the recorded one
// every plan record names that digest (mode model-plan-changed). adjust may
// change a plan record; it is nil unless a test needs a different volume.
func emitRecordedPlan(lines []recordedLine, runDigest string, adjust func(payload map[string]any)) {
	recorded := planDigestOf(lines)
	for _, l := range lines {
		if l.Type == "result" {
			continue
		}
		if l.Type == "plan" {
			if runDigest != recorded {
				l.Payload["plan_digest"] = runDigest
			}
			if adjust != nil {
				adjust(l.Payload)
			}
		}
		event(l.Type, l.Payload)
		if l.Type == "preflight" && mode == "model-plan-verification" && hasFlag("--plan") {
			for n := 0; n < 500; n++ {
				event("progress", map[string]any{"component": "model", "phase": "verify", "file_id": "existing", "bytes_done": n, "bytes_total": 500, "resume_offset": 0, "verification_method": "sha256"})
			}
		}
	}
}

// recordedPlanResult is the recorded result record of a plan-only run.
func recordedPlanResult(lines []recordedLine) {
	for _, l := range lines {
		if l.Type == "result" {
			event("result", l.Payload)
			return
		}
	}
}

// shortVolume makes the plan volumes report less free space than they need.
// Hand-edited field (volume-short): no recording of a short volume exists.
func shortVolume(payload map[string]any) {
	if _, err := os.Stat(filepath.Join(dir, "volume-short")); err != nil {
		return
	}
	volumes, _ := payload["volumes"].([]any)
	for _, v := range volumes {
		if volume, ok := v.(map[string]any); ok {
			if required, ok := volume["required_bytes"].(float64); ok && required > 1 {
				volume["free_bytes"] = required - 1
			}
		}
	}
}

// catalogue serves `prepare model --catalogue-id ID` in every variant.
func catalogue() {
	if packagesMode() {
		cataloguePackages()
		return
	}
	recordArgv("prepare")
	id := flag("--catalogue-id")
	lines, exit := recordedPlan(id)
	if exit != 0 {
		// The CLI refuses: its records, then its exit code.
		emitRecordedPlan(lines, planDigestOf(lines), nil)
		os.Exit(exit)
	}
	digest := planDigestOf(lines)
	runDigest := digest
	if mode == "model-plan-changed" && !hasFlag("--plan") {
		runDigest = strings.Repeat("d", 64)
	}
	emitRecordedPlan(lines, runDigest, shortVolume)
	if hasFlag("--plan") {
		recordedPlanResult(lines)
		return
	}
	offline := hasFlag("--offline")
	if !offline && mode != "model-no-approval" {
		binding := approvalDigest(map[string]string{"model": runDigest})
		event("approval_required", map[string]any{"plan_digest": binding, "components": []string{"model"}})
		if !approve(binding) {
			return
		}
	}
	// A cancel line may arrive at any time after approval.
	cancelled := make(chan struct{})
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		for sc.Scan() {
			var in map[string]any
			if json.Unmarshal(sc.Bytes(), &in) == nil && in["action"] == "cancel" {
				close(cancelled)
				return
			}
		}
	}()
	if mode == "model-error" {
		cliError("storage_failed", "model", "download", false, "The drive is full. Free space and start again.", 7)
	}
	if fault := os.Getenv("BUBO_FAKE_MODEL_FAILURE"); fault != "" {
		// _source: fixture, tests/cli/contract/valid/error-<fault>.json.
		switch fault {
		case "transport_failed", "integrity_failed", "storage_failed":
			kind, payload := replay("error-" + fault)
			event(kind, payload)
			os.Exit(7)
		default:
			panic("unknown BUBO_FAKE_MODEL_FAILURE")
		}
	}
	steps := 4
	if b, err := os.ReadFile(filepath.Join(dir, "model-steps")); err == nil {
		steps, _ = strconv.Atoi(strings.TrimSpace(string(b)))
	}
	pause := 20 * time.Millisecond
	pause = progressPause(pause)
	phase := "download"
	if offline || hasFlag("--verify") {
		phase = "verify"
	}
	for n := 0; n <= steps; n++ {
		event("progress", map[string]any{"component": "model", "phase": phase, "file_id": "main", "bytes_done": n, "bytes_total": steps, "resume_offset": 0, "verification_method": "sha256"})
		if (mode == "model-stall" || os.Getenv("BUBO_FAKE_MODEL_STALL") == "1") && n == 0 {
			select {
			case <-cancelled:
				os.WriteFile(filepath.Join(dir, "cancel-received"), []byte("cancel"), 0600)
				cliError("cancelled", "model", "download", false, "Cancelled.", 3)
			case <-time.After(30 * time.Second):
				return
			}
		}
		select {
		case <-cancelled:
			if os.Getenv("BUBO_FAKE_MODEL_CANCEL_MODE") == "late-progress" {
				event("progress", map[string]any{"component": "model", "phase": phase, "file_id": "main", "bytes_done": n + 1, "bytes_total": steps, "resume_offset": 0, "verification_method": "sha256"})
				time.Sleep(time.Second)
			}
			if os.Getenv("BUBO_FAKE_MODEL_CANCEL_MODE") == "exit-one" {
				os.Exit(1)
			}
			cliError("cancelled", "model", "download", false, "Cancelled.", 3)
		case <-time.After(pause):
		}
	}
	event("result", map[string]any{"exit_code": 0, "manifest_path": "C:/test/bubo-package.json", "prepared": true})
}
